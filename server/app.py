"""
server/app.py — SME Credit Risk RL Environment — FastAPI Server
===============================================================
Exposes the LoanEnvironment over HTTP + WebSocket using the OpenEnv
server pattern from module-4.

Endpoints (all created automatically by create_fastapi_app):
  WS  /ws              — Primary WebSocket interface (reset / step / state)
  POST /reset          — HTTP reset (pass task_id in body)
  POST /step           — HTTP step  (pass LoanAction fields in body)
  GET  /state          — HTTP state snapshot
  GET  /health         — Liveness probe  {"status": "healthy"}
  GET  /docs           — Swagger / OpenAPI UI
  GET  /web            — Browser-friendly environment viewer

Extra endpoints added here:
  GET  /tasks          — List all available task IDs grouped by difficulty
  POST /grade          — Grade a completed episode from its action_log
  GET  /tasks/{tid}    — Get one task's public metadata (no ground truth)

Run locally:
  uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

Or from the repo root:
  uvicorn server.app:app --reload --port 7860
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap — makes `models` and `env` importable whether the server is
# run from the repo root or from inside server/
# ---------------------------------------------------------------------------
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
    ASSESS_ACTIONS,
    DECIDE_ACTIONS,
    REVEALABLE_FACTORS,
    ACTION_TO_FACTOR,
)
from env.environment import LoanEnvironment, _load_tasks
from env.graders import grade


# ---------------------------------------------------------------------------
# Try to wire up via OpenEnv's create_fastapi_app.
# If openenv-core is not installed we fall back to a plain FastAPI app that
# still exposes all the same endpoints over plain HTTP — useful for local dev
# and hackathon judges running the code without the full framework.
# ---------------------------------------------------------------------------
try:
    from openenv import create_fastapi_app as _create
    app = _create(LoanEnvironment)
    _OPENENV_WIRED = True
    print("✅ OpenEnv wired successfully")
except Exception as e:
    print("❌ OpenEnv wiring failed:", e)

    app = FastAPI(
        title="SME Credit Risk RL Environment",
        description="Fallback FastAPI (OpenEnv not wired)",
        version="1.0.0",
    )
    _OPENENV_WIRED = False


# ---------------------------------------------------------------------------
# CORS — allow any origin so the HF Spaces web UI and local notebook can talk
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Shared environment instance (one per worker process).
# For concurrent usage the OpenEnv WebSocket layer creates per-session
# instances automatically; this shared instance is used only by the plain
# HTTP fallback endpoints below.
# ---------------------------------------------------------------------------
_env = LoanEnvironment()


# ---------------------------------------------------------------------------
# Request / response models for the plain HTTP endpoints
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "easy_01"


class StepRequest(BaseModel):
    action_type: str
    application_id: str


class GradeRequest(BaseModel):
    action_log: list
    ground_truth: str
    task_id: str   # "easy" | "medium" | "hard"


# ---------------------------------------------------------------------------
# Plain HTTP endpoints (work even without openenv-core installed)
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness probe — returns 200 when the server is up."""
    return {"status": "healthy", "openenv_wired": _OPENENV_WIRED}


@app.post("/reset")
def reset(req: ResetRequest):
    """
    Start a new episode.

    Body: { "task_id": "easy_01" }
    Returns: LoanObservation as JSON.
    """
    import dataclasses
    try:
        obs = _env.reset(req.task_id)
        return dataclasses.asdict(obs)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/step")
def step(req: StepRequest):
    """
    Take one action in the current episode.

    Body: { "action_type": "assess_credit_score", "application_id": "easy_01" }
    Returns: LoanObservation as JSON.
    """
    import dataclasses
    if req.action_type not in VALID_ACTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown action_type '{req.action_type}'. Valid: {sorted(VALID_ACTIONS)}",
        )
    action = LoanAction(action_type=req.action_type, application_id=req.application_id)
    obs = _env.step(action)
    return dataclasses.asdict(obs)


@app.get("/state")
def state():
    """
    Return the full internal state of the current episode.
    Includes ground truth — intended for graders and debugging only.
    """
    import dataclasses
    return dataclasses.asdict(_env.state)


# ---------------------------------------------------------------------------
# Extra endpoints
# ---------------------------------------------------------------------------

@app.get("/tasks")
def list_tasks():
    """
    List all available task IDs grouped by difficulty.
    Reveals only public fields (no financial factors, no ground truth).
    """
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
        "total": len(all_tasks),
        "tasks": grouped,
        "valid_actions": sorted(VALID_ACTIONS),
        "revealable_factors": REVEALABLE_FACTORS,
        "action_to_factor": ACTION_TO_FACTOR,
    }


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    """
    Get public metadata for one task by its application_id.
    Does NOT return financial factors or ground truth.
    """
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
        "explanation":    t.get("explanation", ""),  # human-readable, no raw values
    }


@app.post("/grade")
def grade_episode(req: GradeRequest):
    """
    Grade a completed episode.

    Body:
      {
        "action_log":   [...],           # from LoanState.action_log
        "ground_truth": "approve",       # from LoanState.ground_truth_decision
        "task_id":      "hard"           # "easy" | "medium" | "hard"
      }

    Returns:
      { "score": 0.85, "task_id": "hard", "n_reveals": 3, "decision": "approve" }
    """
    if req.task_id not in ("easy", "medium", "hard"):
        raise HTTPException(
            status_code=400,
            detail=f"task_id must be 'easy', 'medium', or 'hard'. Got: '{req.task_id}'"
        )
    if req.ground_truth not in ("approve", "reject", "refer"):
        raise HTTPException(
            status_code=400,
            detail=f"ground_truth must be approve/reject/refer. Got: '{req.ground_truth}'"
        )

    score = grade(req.action_log, req.ground_truth, req.task_id)

    # Extract summary stats from the log for the response
    n_reveals  = sum(1 for e in req.action_log if e.get("action_type", "").startswith("assess_") and e.get("valid", True))
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


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------
def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()