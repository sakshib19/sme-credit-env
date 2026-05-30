"""
server/app.py — SME Credit Risk RL Environment — FastAPI Server
===============================================================
Uses openenv.core.env_server.http_server.create_app() to ensure 100% 
compliance with the OpenEnv hackathon validator.

This automatically wires up:
  - POST /reset
  - POST /step   <-- (Requires payload: {"action": {"action_type": ...}})
  - GET  /state
  - GET  /schema
  - WS   /ws     <-- Required for OpenEnv UI

Added custom /recommend and /health endpoints manually.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

# Path bootstrap
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 1. OpenEnv spec compliance (MANDATORY FOR GRADER)
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv-core is required. pip install 'openenv-core[core]'") from e

try:
    from ..models import LoanAction, LoanObservation, LoanState
    from .loan_environment import LoanEnvironment
except (ModuleNotFoundError, ImportError):
    from models import LoanAction, LoanObservation, LoanState
    from server.loan_environment import LoanEnvironment

# Do NOT define /reset, /step, or /state manually. create_app() owns them.
app = create_app(
    LoanEnvironment,
    LoanAction,
    LoanObservation,
    env_name="sme-credit-env",
    max_concurrent_envs=10,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from tasks.graders import grade as _grade
from tasks.environment import _load_tasks
from models import VALID_ACTIONS, REVEALABLE_FACTORS, ACTION_TO_FACTOR

# ---------------------------------------------------------------------------
# 2. Hackathon & HF Health Probes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Liveness probe — validator and HF health check ping this."""
    return {"status": "healthy", "env": "sme-credit-env", "tasks": 50}

# ---------------------------------------------------------------------------
# 3. Task Management & Grading Endpoints
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

class GradeRequest(BaseModel):
    action_log:   list
    ground_truth: str
    task_id:      str

@app.post("/grade")
def grade_episode(req: GradeRequest):
    if req.task_id not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail=f"task_id must be easy/medium/hard. Got: '{req.task_id}'")
    if req.ground_truth not in ("approve", "reject", "refer"):
        raise HTTPException(status_code=400, detail=f"ground_truth must be approve/reject/refer. Got: '{req.ground_truth}'")

    score = _grade(req.action_log, req.ground_truth, req.task_id)
    n_reveals = sum(1 for e in req.action_log if e.get("action_type", "").startswith("assess_") and e.get("valid", True))
    n_invalid  = sum(1 for e in req.action_log if not e.get("valid", True))
    final_entry = next((e for e in reversed(req.action_log) if e.get("action_type", "").startswith("decide_")), None)
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
# 4. Custom Testing Endpoint (Direct Risk Simulation)
# ---------------------------------------------------------------------------
class RecommendRequest(BaseModel):
    loan_amount:          float = Field(..., description="Loan amount in GBP")
    credit_score:         Optional[int]   = Field(None, description="Credit score 300-850")
    dti:                  Optional[float] = Field(None, description="Debt-to-income ratio 0.0-1.0")
    annual_revenue:       Optional[float] = Field(None, description="Annual revenue in GBP")
    collateral_value:     Optional[float] = Field(None, description="Collateral value in GBP")
    business_age_years:   Optional[float] = Field(None, description="Business age in years")
    cash_flow_volatility: Optional[float] = Field(None, description="Cash flow volatility 0.0-1.0")

def _compute_risk_score(loan_amount, credit_score, dti, annual_revenue, collateral_value, business_age_years, cash_flow_volatility) -> dict:
    UNKNOWN = 0.5

    def credit_p(s): return max(0.0, min(1.0, (850 - s) / 550))
    def dti_p(d):
        if d <= 0.30: return 0.0
        if d <= 0.45: return 0.2
        if d <= 0.60: return 0.5
        return min(1.0, 0.85 + (d - 0.60) * 0.75)
    def ltr_p(l):
        if l <= 0.25: return 0.0
        if l <= 0.50: return 0.15
        if l <= 1.00: return 0.40
        if l <= 2.00: return 0.70
        return 1.0
    def age_p(a):
        if a >= 5.0: return 0.0
        if a >= 3.0: return 0.15
        if a >= 2.0: return 0.35
        if a >= 1.0: return 0.60
        return 0.90

    hard_floor = None
    if credit_score is not None and credit_score < 500: hard_floor = f"credit_score={credit_score} < 500"
    if dti is not None and dti > 0.80: hard_floor = f"dti={dti:.2f} > 0.80"

    cp = credit_p(credit_score) if credit_score is not None else UNKNOWN
    dp = dti_p(dti) if dti is not None else UNKNOWN
    ltr = loan_amount / annual_revenue if (annual_revenue and annual_revenue > 0) else None
    lp = ltr_p(ltr) if ltr is not None else UNKNOWN
    ap = age_p(business_age_years) if business_age_years is not None else UNKNOWN
    vp = min(1.0, cash_flow_volatility * 1.2) if cash_flow_volatility is not None else UNKNOWN

    if collateral_value is not None:
        coverage = collateral_value / max(loan_amount, 1)
        discount = 1.0 - 0.20 * min(coverage, 2.0)
    else:
        coverage, discount = None, 1.0

    raw = cp * 0.30 + dp * 0.25 + lp * 0.20 + ap * 0.15 + vp * 0.10
    risk = round(raw * discount, 4)

    if hard_floor:
        decision, reason = "reject", f"Hard floor triggered: {hard_floor}"
    elif risk < 0.35:
        decision, reason = "approve", f"Risk score {risk:.4f} below 0.35 approval threshold"
    elif risk > 0.65:
        decision, reason = "reject", f"Risk score {risk:.4f} exceeds 0.65 rejection threshold"
    else:
        decision, reason = "refer", f"Risk score {risk:.4f} falls in borderline band (0.35–0.65)"

    known = sum(1 for v in [credit_score, dti, annual_revenue, collateral_value, business_age_years, cash_flow_volatility] if v is not None)
    confidence = "HIGH" if known >= 5 else "MEDIUM" if known >= 3 else "LOW"

    return {
        "decision": decision,
        "reason": reason,
        "confidence": confidence,
        "factors_known": f"{known}/6",
        "risk_score": risk,
        "breakdown": {
            "raw_risk": round(raw, 4),
            "collateral_discount": round(discount, 4)
        }
    }

@app.post("/recommend")
def recommend(req: RecommendRequest):
    return _compute_risk_score(
        req.loan_amount, req.credit_score, req.dti, req.annual_revenue, 
        req.collateral_value, req.business_age_years, req.cash_flow_volatility
    )

def main():
    import uvicorn
    # PORT 7860 is strictly required for Hugging Face Spaces compatibility
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()