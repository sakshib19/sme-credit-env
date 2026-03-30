"""
models.py — SME Credit Risk Assessment Environment
====================================================
Type-safe contracts for the OpenEnv interface.

Architecture
------------
The real openenv-core package ships Observation and Action as Pydantic v2
BaseModel subclasses, and State as a plain dataclass.  Inheriting from
a Pydantic model using @dataclass is not supported in Pydantic v2 — it
raises AttributeError: '__pydantic_extra__' at runtime.

Solution used here:
  • LoanAction      → plain @dataclass  (standalone, no inheritance)
  • LoanObservation → plain @dataclass  (standalone, no inheritance)
  • LoanState       → plain @dataclass  (standalone, no inheritance)

app.py passes these classes to create_fastapi_app() which handles
the OpenEnv registration.  The environment itself just needs the
three dataclasses to carry the right fields — it does not need to
inherit from openenv base classes at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Constants — action strings
# ---------------------------------------------------------------------------

ASSESS_ACTIONS = {
    "assess_revenue",        # reveals annual_revenue
    "assess_credit_score",   # reveals credit_score
    "assess_dti",            # reveals dti
    "assess_collateral",     # reveals collateral_value
    "assess_business_age",   # reveals business_age_years
    "assess_cash_flow",      # reveals cash_flow_volatility
}

DECIDE_ACTIONS = {
    "decide_approve",
    "decide_reject",
    "decide_refer",
}

VALID_ACTIONS = ASSESS_ACTIONS | DECIDE_ACTIONS

REVEALABLE_FACTORS = [
    "annual_revenue",
    "credit_score",
    "dti",
    "collateral_value",
    "business_age_years",
    "cash_flow_volatility",
]

ACTION_TO_FACTOR: dict[str, str] = {
    "assess_revenue":      "annual_revenue",
    "assess_credit_score": "credit_score",
    "assess_dti":          "dti",
    "assess_collateral":   "collateral_value",
    "assess_business_age": "business_age_years",
    "assess_cash_flow":    "cash_flow_volatility",
}


# ---------------------------------------------------------------------------
# LoanAction — what the agent sends
# ---------------------------------------------------------------------------

@dataclass
class LoanAction:
    """
    One agent action per step.

    action_type : str
        One of VALID_ACTIONS.
        assess_*  → reveal a hidden financial factor.
        decide_*  → make the final approve / reject / refer decision.

    application_id : str
        Must match the application_id in the current observation.
        A mismatch triggers a penalty in environment.step().
    """
    action_type:    str = "assess_revenue"
    application_id: str = ""


# ---------------------------------------------------------------------------
# LoanObservation — what the agent receives back
# ---------------------------------------------------------------------------

@dataclass
class LoanObservation:
    """
    Agent's view after each step.

    Base fields (match OpenEnv Observation contract)
    -------------------------------------------------
    done     : bool         — True when the episode is complete.
    reward   : float | None — Step reward earned.
    metadata : dict         — Reserved for framework use.

    Always visible
    --------------
    application_id, task_id, business_name, sector, loan_amount,
    loan_to_revenue (auto-computed once annual_revenue is known)

    Revealable factors (None = hidden until the agent calls assess_*)
    -----------------------------------------------------------------
    annual_revenue, credit_score, dti, collateral_value,
    business_age_years, cash_flow_volatility

    Progress / feedback
    -------------------
    factors_assessed, factors_remaining, step_count, max_steps,
    cumulative_reward, last_action_valid, feedback, final_decision
    """

    # OpenEnv base contract fields
    done:     bool          = False
    reward:   Optional[float] = None
    metadata: dict          = field(default_factory=dict)

    # Always visible
    application_id: str   = ""
    task_id:        str   = "easy"
    business_name:  str   = ""
    sector:         str   = ""
    loan_amount:    float = 0.0
    loan_to_revenue: Optional[float] = None

    # Revealable factors (None = hidden)
    annual_revenue:       Optional[float] = None
    credit_score:         Optional[int]   = None
    dti:                  Optional[float] = None
    collateral_value:     Optional[float] = None
    business_age_years:   Optional[float] = None
    cash_flow_volatility: Optional[float] = None

    # Progress
    factors_assessed: list  = field(default_factory=list)
    factors_remaining: list = field(default_factory=list)
    step_count:        int  = 0
    max_steps:         int  = 8
    cumulative_reward: float = 0.0

    # Feedback
    last_action_valid: bool         = True
    feedback:          str          = ""
    final_decision:    Optional[str] = None


# ---------------------------------------------------------------------------
# LoanState — full server-side snapshot (GET /state)
# ---------------------------------------------------------------------------

@dataclass
class LoanState:
    """
    Complete internal environment state — returned by GET /state.
    Contains ground truth which is hidden from the agent mid-episode.

    Base fields (match OpenEnv State contract)
    ------------------------------------------
    episode_id : str  — unique ID set on each reset().
    step_count : int  — incremented each valid step().
    metadata   : dict — reserved for framework use.
    """

    # OpenEnv base contract fields
    episode_id: str  = ""
    step_count: int  = 0
    metadata:   dict = field(default_factory=dict)

    # Identification
    task_id:        str = ""
    application_id: str = ""

    # Ground truth (server-side only — never shown to agent mid-episode)
    ground_truth_decision:   str   = ""
    ground_truth_risk_score: float = 0.0
    factor_directions:       dict  = field(default_factory=dict)

    # Progress
    factors_assessed:  list         = field(default_factory=list)
    cumulative_reward: float        = 0.0
    final_decision:    Optional[str] = None

    # Penalty accounting
    penalty_total:       float = 0.0
    correct_assessments: int   = 0

    # Full action log (passed to graders at episode end)
    action_log: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Self-test — python models.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import dataclasses

    print("=" * 56)
    print("  models.py — self-test")
    print("=" * 56)

    a = LoanAction(action_type="assess_credit_score", application_id="easy_01")
    assert a.action_type in VALID_ACTIONS
    print(f"  LoanAction     action_type={a.action_type}  ✓")

    o = LoanObservation(application_id="easy_01", task_id="easy",
                        loan_amount=80_000, done=False, reward=0.0)
    assert o.credit_score is None
    assert o.done is False
    d = dataclasses.asdict(o)
    assert "credit_score" in d and "done" in d
    print(f"  LoanObservation credit_score=None  done=False  asdict() ✓")

    s = LoanState(episode_id="ep_001", task_id="easy",
                  ground_truth_decision="approve")
    assert s.episode_id == "ep_001"
    sd = dataclasses.asdict(s)
    assert "ground_truth_decision" in sd
    print(f"  LoanState      episode_id={s.episode_id}  asdict() ✓")

    assert len(VALID_ACTIONS) == 9
    assert len(ACTION_TO_FACTOR) == 6
    print(f"  9 valid actions, 6 factor mappings  ✓")
    print("\n  All assertions passed.\n")