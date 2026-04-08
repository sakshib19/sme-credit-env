"""
server/app.py — SME Credit Risk RL Environment — FastAPI Server
===============================================================
Exposes the LoanEnvironment over HTTP.

Endpoints:
  GET  /health         — Liveness probe {"status": "healthy"}
  POST /reset          — Start a new episode. Body: {"task_id": "easy_01"} or {}
  POST /step           — Take one action
  GET  /state          — Full internal state (includes ground truth)
  GET  /tasks          — List all 50 task IDs
  GET  /tasks/{tid}    — Get one task's public metadata
  POST /grade          — Grade a completed episode
  GET  /docs           — Swagger UI

CRITICAL: reset() and step() return LoanObservation dataclass instances.
We serialise with dataclasses.asdict() — NOT obs.model_dump().
LoanObservation is a plain @dataclass, not a Pydantic model.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from models import (
    LoanAction,
    LoanObservation,
    LoanState,
    VALID_ACTIONS,
    REVEALABLE_FACTORS,
    ACTION_TO_FACTOR,
)
from env.environment import LoanEnvironment, _load_tasks
from env.graders import grade


# ---------------------------------------------------------------------------
# Build FastAPI app — try openenv-core first, fall back to plain FastAPI
# ---------------------------------------------------------------------------
_OPENENV_WIRED = False
try:
    from openenv.core.env_server import create_fastapi_app as _create
    app = _create(LoanEnvironment)
    _OPENENV_WIRED = True
except Exception:
    app = FastAPI(
        title="SME Credit Risk RL Environment",
        description=(
            "Multi-step RL environment for SME loan underwriting. "
            "The agent reveals financial factors one at a time then makes a "
            "final approve / reject / refer decision."
        ),
        version="1.0.0",
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------

def _obs_to_dict(obs) -> dict:
    """
    Serialise a LoanObservation to a plain dict.
    LoanObservation is a @dataclass — use dataclasses.asdict().
    Never call .model_dump() on it; that is a Pydantic method.
    """
    if isinstance(obs, dict):
        return obs
    if dataclasses.is_dataclass(obs):
        return dataclasses.asdict(obs)
    # Pydantic fallback (should not be needed)
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    return dict(obs)


# ---------------------------------------------------------------------------
# Shared environment instance (one per worker)
# ---------------------------------------------------------------------------
_env = LoanEnvironment()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "easy_01"   # validator may send {} — this default handles it


class StepRequest(BaseModel):
    action_type:    str
    application_id: str


class GradeRequest(BaseModel):
    action_log:   list
    ground_truth: str
    task_id:      str


# ---------------------------------------------------------------------------
# Core endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness probe — must return 200."""
    return {"status": "healthy", "openenv_wired": _OPENENV_WIRED, "tasks": 50}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    """
    Start a new episode.
    Accepts: {"task_id": "easy_01"} OR empty body {}.
    Returns LoanObservation as JSON dict.
    """
    try:
        obs = _env.reset(req.task_id)
        return _obs_to_dict(obs)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"reset() failed: {str(e)}")


@app.post("/step")
def step(req: StepRequest):
    """
    Take one action in the current episode.
    Returns LoanObservation as JSON dict.
    """
    if req.action_type not in VALID_ACTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown action_type '{req.action_type}'. Valid: {sorted(VALID_ACTIONS)}",
        )
    try:
        action = LoanAction(
            action_type=req.action_type,
            application_id=req.application_id,
        )
        obs = _env.step(action)
        return _obs_to_dict(obs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"step() failed: {str(e)}")


@app.get("/state")
def state():
    """Full internal state — includes ground truth (for graders/debugging)."""
    return dataclasses.asdict(_env.state)


# ---------------------------------------------------------------------------
# Extra endpoints
# ---------------------------------------------------------------------------

@app.get("/tasks")
def list_tasks():
    all_tasks = _load_tasks()
    grouped: dict[str, list] = {"easy": [], "medium": [], "hard": []}
    for t in all_tasks:
        tier = t.get("task_id", "unknown")
        if tier in grouped:
            grouped[tier].append({
                "application_id": t["application_id"],
                "business_name":  t.get("business_name", ""),
                "sector":         t.get("sector", ""),
                "loan_amount":    t.get("loan_amount", 0.0),
                "task_id":        tier,
            })
    return {
        "total":              len(all_tasks),
        "tasks":              grouped,
        "valid_actions":      sorted(VALID_ACTIONS),
        "revealable_factors": REVEALABLE_FACTORS,
        "action_to_factor":   ACTION_TO_FACTOR,
    }


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    all_tasks = _load_tasks()
    index = {t["application_id"]: t for t in all_tasks}
    if task_id not in index:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    t = index[task_id]
    return {
        "application_id": t["application_id"],
        "business_name":  t.get("business_name", ""),
        "sector":         t.get("sector", ""),
        "loan_amount":    t.get("loan_amount", 0.0),
        "task_id":        t.get("task_id", ""),
        "explanation":    t.get("explanation", ""),
    }


@app.post("/grade")
def grade_episode(req: GradeRequest):
    if req.task_id not in ("easy", "medium", "hard"):
        raise HTTPException(
            status_code=400,
            detail=f"task_id must be easy/medium/hard. Got: '{req.task_id}'"
        )
    if req.ground_truth not in ("approve", "reject", "refer"):
        raise HTTPException(
            status_code=400,
            detail=f"ground_truth must be approve/reject/refer. Got: '{req.ground_truth}'"
        )

    score = grade(req.action_log, req.ground_truth, req.task_id)

    n_reveals = sum(1 for e in req.action_log
                    if e.get("action_type", "").startswith("assess_") and e.get("valid", True))
    n_invalid  = sum(1 for e in req.action_log if not e.get("valid", True))
    final_entry = next(
        (e for e in reversed(req.action_log) if e.get("action_type", "").startswith("decide_")),
        None,
    )
    decision = final_entry["action_type"].replace("decide_", "") if final_entry else None

    return {
        "score":        score,
        "task_id":      req.task_id,
        "ground_truth": req.ground_truth,
        "decision":     decision,
        "correct":      decision == req.ground_truth,
        "n_reveals":    n_reveals,
        "n_invalid":    n_invalid,
        "n_steps":      len(req.action_log),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)