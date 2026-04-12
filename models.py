"""
models.py — SME Credit Risk RL — typed models
==============================================
CRITICAL: LoanAction, LoanObservation, LoanState MUST inherit from
openenv.core.env_server.types Action/Observation/State.

create_app() calls:
  isinstance(env_instance, Environment)      ← needs Environment base
  action_cls.__fields__                      ← needs Pydantic (from openenv types)
  observation_cls.model_validate(payload)    ← needs Pydantic .model_validate()

Plain @dataclass has none of these → TypeError on startup.
The openenv base types are Pydantic v2 models. All our fields just work.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from openenv.core.env_server.types import Action, Observation, State


# ---------------------------------------------------------------------------
# Constants — unchanged from original
# ---------------------------------------------------------------------------

REVEALABLE_FACTORS = [
    "credit_score",
    "dti",
    "annual_revenue",
    "collateral_value",
    "business_age_years",
    "cash_flow_volatility",
]

ASSESS_ACTIONS = {
    "assess_credit_score",
    "assess_dti",
    "assess_revenue",
    "assess_collateral",
    "assess_business_age",
    "assess_cash_flow",
}

DECIDE_ACTIONS = {
    "decide_approve",
    "decide_reject",
    "decide_refer",
}

VALID_ACTIONS = ASSESS_ACTIONS | DECIDE_ACTIONS

ACTION_TO_FACTOR: Dict[str, str] = {
    "assess_credit_score":   "credit_score",
    "assess_dti":            "dti",
    "assess_revenue":        "annual_revenue",
    "assess_collateral":     "collateral_value",
    "assess_business_age":   "business_age_years",
    "assess_cash_flow":      "cash_flow_volatility",
}


# ---------------------------------------------------------------------------
# Models — inherit from openenv types (Pydantic v2 under the hood)
# ---------------------------------------------------------------------------

class LoanAction(Action):
    """
    Action sent by the agent.
    Inherits Action (Pydantic BaseModel) — create_app() uses .model_dump()
    and field introspection on this class.
    """
    action_type:    str   # e.g. "assess_credit_score" or "decide_approve"
    application_id: str   # must match current episode's application_id


class LoanObservation(Observation):
    """
    Observation returned by reset() and step().
    Inherits Observation (Pydantic BaseModel) — create_app() uses
    .model_validate() to deserialise from JSON and .model_dump() to serialise.

    All fields must have defaults so that empty construction is possible
    (create_app() may construct an empty observation for schema generation).
    """
    done:                bool           = False
    reward:              float          = 0.0
    metadata:            Dict[str, Any] = {}

    application_id:      str            = ""
    task_id:             str            = ""
    business_name:       str            = ""
    sector:              str            = ""
    loan_amount:         float          = 0.0

    # Financial factors — None means not yet revealed
    loan_to_revenue:     Optional[float] = None
    annual_revenue:      Optional[float] = None
    credit_score:        Optional[int]   = None
    dti:                 Optional[float] = None
    collateral_value:    Optional[float] = None
    business_age_years:  Optional[float] = None
    cash_flow_volatility: Optional[float] = None

    factors_assessed:    List[str]      = []
    factors_remaining:   List[str]      = []
    step_count:          int            = 0
    max_steps:           int            = 8
    cumulative_reward:   float          = 0.0
    last_action_valid:   bool           = True
    feedback:            str            = ""
    final_decision:      Optional[str]  = None


class LoanState(State):
    """
    Full internal state — includes ground truth. Used by /state endpoint.
    Inherits State (Pydantic BaseModel).
    """
    episode_id:              str            = ""
    step_count:              int            = 0
    metadata:                Dict[str, Any] = {}
    task_id:                 str            = ""
    application_id:          str            = ""
    ground_truth_decision:   str            = ""
    ground_truth_risk_score: float          = 0.0
    factor_directions:       Dict[str, str] = {}
    factors_assessed:        List[str]      = []
    cumulative_reward:       float          = 0.0
    final_decision:          Optional[str]  = None
    penalty_total:           float          = 0.0
    correct_assessments:     int            = 0
    action_log:              List[Dict[str, Any]] = []