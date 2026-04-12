"""
server/app.py — SME Credit Risk RL Environment — FastAPI Server
===============================================================
Uses openenv.core.env_server.http_server.create_app() exactly as the
reference repo (harsh063423/my_env) does. This wires up:
  - POST /reset
  - POST /step
  - GET  /state
  - GET  /schema
  - WS   /ws          ← WebSocket for Playground UI
  - GET  /health       ← added manually after create_app()

DO NOT define /reset /step /state manually — create_app() owns those.
"""

from __future__ import annotations
import sys
from pathlib import Path

# Path bootstrap — ensures local packages resolve regardless of cwd
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# OpenEnv wiring — EXACT pattern from reference repo
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv-core is required. Install with: pip install 'openenv-core[core]'"
    ) from e

try:
    from ..models import LoanAction, LoanObservation, LoanState
    from .loan_environment import LoanEnvironment
except (ModuleNotFoundError, ImportError):
    from models import LoanAction, LoanObservation, LoanState
    from server.loan_environment import LoanEnvironment

# create_app() registers /reset /step /state /schema /ws automatically.
# Do NOT add @app.post('/reset') etc — that causes duplicate route errors.
app = create_app(
    LoanEnvironment,
    LoanAction,
    LoanObservation,
    env_name="sme-credit-env",
    max_concurrent_envs=10,
)

# ---------------------------------------------------------------------------
# Additional middleware and endpoints — added AFTER create_app()
# ---------------------------------------------------------------------------
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from tasks.graders import grade as _grade
from tasks.environment import _load_tasks
from models import VALID_ACTIONS, REVEALABLE_FACTORS, ACTION_TO_FACTOR


@app.get("/health")
def health():
    """Liveness probe — validator and HF health check ping this."""
    return {"status": "healthy", "env": "sme-credit-env", "tasks": 50}


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


class GradeRequest(BaseModel):
    action_log:   list
    ground_truth: str
    task_id:      str


@app.post("/grade")
def grade_episode(req: GradeRequest):
    if req.task_id not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400,
            detail=f"task_id must be easy/medium/hard. Got: '{req.task_id}'")
    if req.ground_truth not in ("approve", "reject", "refer"):
        raise HTTPException(status_code=400,
            detail=f"ground_truth must be approve/reject/refer. Got: '{req.ground_truth}'")

    score = _grade(req.action_log, req.ground_truth, req.task_id)
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


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()