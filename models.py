"""
models.py — SME Credit Risk Assessment Environment
====================================================
Type-safe contracts for the OpenEnv interface.

Three dataclasses following the exact OpenEnv spec:
  LoanAction      — what the agent sends each step
  LoanObservation — what the agent receives back
  LoanState       — full internal snapshot (state() endpoint)

Import pattern (confirmed from OpenEnv official docs):
  from dataclasses import dataclass, field
  from openenv.core.env_server import Action, Observation, State

Base class fields (inherited, do NOT redeclare):
  Action       → no required base fields
  Observation  → done: bool, reward: float | None, metadata: dict
  State        → episode_id: str, step_count: int, metadata: dict

All custom fields must have default values so the dataclass
can be instantiated with no arguments (required by OpenEnv validator).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from openenv.core.env_server import Action, Observation, State


# ---------------------------------------------------------------------------
# Action type enum — kept as plain strings for JSON serialisation simplicity.
# The full set of valid values is documented below each field.
# ---------------------------------------------------------------------------

# assess_*  actions reveal one hidden factor to the agent
# decide_*  actions make the final loan decision (ends the episode)
ASSESS_ACTIONS = {
    "assess_revenue",         # reveals annual_revenue
    "assess_credit_score",    # reveals credit_score
    "assess_dti",             # reveals debt-to-income ratio
    "assess_collateral",      # reveals collateral_value
    "assess_business_age",    # reveals business_age_years
    "assess_cash_flow",       # reveals cash_flow_volatility
}

DECIDE_ACTIONS = {
    "decide_approve",         # approve the loan application
    "decide_reject",          # reject the loan application
    "decide_refer",           # refer to senior underwriter
}

VALID_ACTIONS = ASSESS_ACTIONS | DECIDE_ACTIONS

# Factors the agent can reveal, in the order they appear in tasks.json
REVEALABLE_FACTORS = [
    "annual_revenue",
    "credit_score",
    "dti",
    "collateral_value",
    "business_age_years",
    "cash_flow_volatility",
]

# Mapping from action string → factor key it reveals
ACTION_TO_FACTOR: dict[str, str] = {
    "assess_revenue":      "annual_revenue",
    "assess_credit_score": "credit_score",
    "assess_dti":          "dti",
    "assess_collateral":   "collateral_value",
    "assess_business_age": "business_age_years",
    "assess_cash_flow":    "cash_flow_volatility",
}


# ---------------------------------------------------------------------------
# LoanAction — one action per step
# ---------------------------------------------------------------------------

@dataclass
class LoanAction(Action):
    """
    A single agent action on the current loan application.

    The agent chooses ONE of:
      Assess actions  → reveal a hidden financial factor
      Decide actions  → make the final approve/reject/refer decision

    Fields
    ------
    action_type : str
        One of the VALID_ACTIONS strings above.
        E.g. "assess_credit_score" or "decide_approve".

    application_id : str
        ID of the application being acted on. Must match the
        application_id in the current observation. Validated by
        environment.step() — mismatch returns a penalty.

    Examples
    --------
    # Reveal the applicant's credit score
    LoanAction(action_type="assess_credit_score", application_id="easy_03")

    # Make final decision
    LoanAction(action_type="decide_approve", application_id="easy_03")
    """

    action_type: str = "assess_revenue"
    application_id: str = ""


# ---------------------------------------------------------------------------
# LoanObservation — returned by reset() and step()
# ---------------------------------------------------------------------------

@dataclass
class LoanObservation(Observation):
    """
    The agent's view of the environment after each action.

    Inherits from Observation:
      done   : bool        — True when episode is complete
      reward : float|None  — step reward (partial or terminal)
      metadata : dict      — reserved for framework use

    All factor values start as None (hidden).
    They are revealed one at a time as the agent calls assess_* actions.

    Fields
    ------
    application_id : str
        ID of the current application being assessed.

    task_id : str
        Which task is running: "easy" | "medium" | "hard".

    business_name : str
        Display name of the SME (cosmetic, for readability).

    sector : str
        Industry sector of the business (cosmetic).

    loan_amount : float
        The loan amount requested (GBP). Always visible — the agent
        needs to know what they're evaluating.

    loan_to_revenue : float | None
        Computed ratio loan_amount / annual_revenue.
        Revealed automatically when both loan_amount and
        annual_revenue are known.

    -- Revealable factors (None = hidden, float = revealed) --

    annual_revenue : float | None
        Annual revenue in GBP. Revealed by "assess_revenue".

    credit_score : int | None
        Credit score 300–850. Revealed by "assess_credit_score".

    dti : float | None
        Debt-to-income ratio 0.0–1.0. Revealed by "assess_dti".

    collateral_value : float | None
        Collateral value in GBP (0 = unsecured). Revealed by
        "assess_collateral".

    business_age_years : float | None
        Years the business has been operating. Revealed by
        "assess_business_age".

    cash_flow_volatility : float | None
        Cash flow stability 0.0 (stable) → 1.0 (volatile).
        Revealed by "assess_cash_flow".

    -- Progress tracking --

    factors_assessed : list[str]
        Names of factors revealed so far this episode.

    factors_remaining : list[str]
        Names of factors not yet revealed.

    step_count : int
        Number of steps taken in this episode so far.

    max_steps : int
        Maximum steps allowed before episode auto-terminates.

    cumulative_reward : float
        Total reward accumulated so far this episode.

    -- Feedback --

    last_action_valid : bool
        False if the last action was structurally invalid
        (wrong application_id, duplicate assess, unknown action_type).

    feedback : str
        Human-readable feedback on the last action, explaining
        the reward earned and any issues. Useful for debugging
        and for including in the agent's system prompt.

    final_decision : str | None
        Set to "approve" | "reject" | "refer" when episode ends,
        None during assessment phase.
    """

    # Always visible
    application_id: str = ""
    task_id: str = "easy"
    business_name: str = ""
    sector: str = ""
    loan_amount: float = 0.0
    loan_to_revenue: Optional[float] = None

    # Revealable factors (None = hidden)
    annual_revenue: Optional[float] = None
    credit_score: Optional[int] = None
    dti: Optional[float] = None
    collateral_value: Optional[float] = None
    business_age_years: Optional[float] = None
    cash_flow_volatility: Optional[float] = None

    # Progress
    factors_assessed: list = field(default_factory=list)
    factors_remaining: list = field(default_factory=list)
    step_count: int = 0
    max_steps: int = 8
    cumulative_reward: float = 0.0

    # Feedback
    last_action_valid: bool = True
    feedback: str = ""
    final_decision: Optional[str] = None


# ---------------------------------------------------------------------------
# LoanState — full internal snapshot (state() endpoint)
# ---------------------------------------------------------------------------

@dataclass
class LoanState(State):
    """
    Complete internal state of the environment.

    Returned by the GET /state endpoint.
    Contains everything in LoanObservation PLUS server-side fields
    that are hidden from the agent during the episode (ground truth).

    Inherits from State:
      episode_id : str   — unique ID for this episode
      step_count : int   — number of steps taken
      metadata   : dict  — reserved for framework use

    Fields
    ------
    task_id : str
        Current task: "easy" | "medium" | "hard".

    application_id : str
        ID of the current application.

    -- Ground truth (server-side only, hidden from agent) --

    ground_truth_decision : str
        The correct decision: "approve" | "reject" | "refer".
        Computed deterministically by the risk formula in
        generate_dataset.py.

    ground_truth_risk_score : float
        The numeric risk score 0.0–1.0 used to derive the decision.

    factor_directions : dict
        Per-factor signal used by graders:
          "positive" → factor supports approval
          "negative" → factor supports rejection
          "neutral"  → borderline signal

    -- Episode progress --

    factors_assessed : list[str]
        Factor keys revealed so far.

    cumulative_reward : float
        Total reward accumulated this episode.

    final_decision : str | None
        Agent's final decision, set when episode ends.

    -- Penalty tracking --

    penalty_total : float
        Sum of all penalties incurred. Useful for debugging
        whether an agent is making avoidable mistakes.

    correct_assessments : int
        Count of assess_* actions where the factor direction
        matched what the agent's decision implied.

    -- Grader inputs --

    action_log : list[dict]
        Full ordered log of every action taken this episode.
        Passed to graders at episode end for final scoring.
    """

    # Task identification
    task_id: str = "easy"
    application_id: str = ""

    # Ground truth (hidden from agent, used by graders)
    ground_truth_decision: str = ""
    ground_truth_risk_score: float = 0.0
    factor_directions: dict = field(default_factory=dict)

    # Progress
    factors_assessed: list = field(default_factory=list)
    cumulative_reward: float = 0.0
    final_decision: Optional[str] = None

    # Penalty accounting
    penalty_total: float = 0.0
    correct_assessments: int = 0

    # Full action log for graders
    action_log: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# __init__.py exports (copy into __init__.py)
# ---------------------------------------------------------------------------
# from .models import LoanAction, LoanObservation, LoanState, VALID_ACTIONS
# from .models import ACTION_TO_FACTOR, ASSESS_ACTIONS, DECIDE_ACTIONS


# ---------------------------------------------------------------------------
# Quick self-test — run with: python models.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import dataclasses

    print("=" * 60)
    print("  SME Credit Risk — OpenEnv Models Self-Test")
    print("=" * 60)

    # --- LoanAction ---
    action = LoanAction(
        action_type="assess_credit_score",
        application_id="easy_01",
    )
    print(f"\nLoanAction:")
    print(f"  action_type      : {action.action_type}")
    print(f"  application_id   : {action.application_id}")
    assert action.action_type in VALID_ACTIONS, "Invalid action_type"

    # --- LoanObservation (initial state — all factors hidden) ---
    obs = LoanObservation(
        application_id="easy_01",
        task_id="easy",
        business_name="Redwood Fabricators Ltd",
        sector="manufacturing",
        loan_amount=80_000.0,
        factors_remaining=list(ACTION_TO_FACTOR.values()),
        max_steps=8,
        feedback="Episode started. Assess factors before deciding.",
        done=False,
        reward=0.0,
    )
    print(f"\nLoanObservation (initial):")
    print(f"  application_id   : {obs.application_id}")
    print(f"  loan_amount      : £{obs.loan_amount:,.0f}")
    print(f"  credit_score     : {obs.credit_score}  (hidden)")
    print(f"  annual_revenue   : {obs.annual_revenue}  (hidden)")
    print(f"  factors_remaining: {obs.factors_remaining}")
    print(f"  done             : {obs.done}")
    print(f"  reward           : {obs.reward}")
    assert obs.done is False
    assert obs.credit_score is None
    assert obs.annual_revenue is None

    # --- LoanObservation (after one reveal) ---
    obs_revealed = LoanObservation(
        application_id="easy_01",
        task_id="easy",
        business_name="Redwood Fabricators Ltd",
        sector="manufacturing",
        loan_amount=80_000.0,
        credit_score=780,
        factors_assessed=["credit_score"],
        factors_remaining=[f for f in REVEALABLE_FACTORS if f != "credit_score"],
        step_count=1,
        cumulative_reward=0.08,
        feedback="Credit score revealed: 780 (good). +0.08 reward.",
        done=False,
        reward=0.08,
    )
    print(f"\nLoanObservation (after assess_credit_score):")
    print(f"  credit_score     : {obs_revealed.credit_score}  (revealed)")
    print(f"  annual_revenue   : {obs_revealed.annual_revenue}  (still hidden)")
    print(f"  cumulative_reward: {obs_revealed.cumulative_reward}")
    print(f"  feedback         : {obs_revealed.feedback}")
    assert obs_revealed.credit_score == 780
    assert obs_revealed.annual_revenue is None

    # --- LoanState ---
    state = LoanState(
        episode_id="ep_abc123",
        task_id="easy",
        application_id="easy_01",
        ground_truth_decision="approve",
        ground_truth_risk_score=0.042,
        factor_directions={
            "credit_score": "positive",
            "dti": "positive",
            "loan_to_revenue": "positive",
            "business_age": "positive",
            "cash_flow_volatility": "positive",
            "collateral": "positive",
        },
        factors_assessed=["credit_score"],
        cumulative_reward=0.08,
        action_log=[
            {"step": 1, "action_type": "assess_credit_score",
             "application_id": "easy_01", "reward": 0.08}
        ],
    )
    print(f"\nLoanState:")
    print(f"  episode_id            : {state.episode_id}")
    print(f"  ground_truth_decision : {state.ground_truth_decision}")
    print(f"  ground_truth_risk     : {state.ground_truth_risk_score}")
    print(f"  factor_directions     : {state.factor_directions}")
    print(f"  action_log entries    : {len(state.action_log)}")
    assert state.ground_truth_decision == "approve"
    assert state.step_count == 0  # base State default

    # --- Serialisation check (dataclasses.asdict) ---
    action_dict = dataclasses.asdict(action)
    obs_dict    = dataclasses.asdict(obs)
    state_dict  = dataclasses.asdict(state)
    assert "action_type" in action_dict
    assert "credit_score" in obs_dict
    assert "ground_truth_decision" in state_dict

    # --- Valid action set check ---
    print(f"\nValid action types ({len(VALID_ACTIONS)}):")
    for a in sorted(VALID_ACTIONS):
        print(f"  {a}")

    print(f"\nASSESS_ACTIONS → factor mapping:")
    for act, factor in ACTION_TO_FACTOR.items():
        print(f"  {act:28s} → {factor}")

    print("\n  All assertions passed. Models are valid.\n")
