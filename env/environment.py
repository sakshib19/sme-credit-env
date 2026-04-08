"""
env/environment.py — SME Credit Risk RL Environment
=====================================================
Implements the core RL loop for the SME loan decision environment.

Agent flow:
  1. reset(task_id)  → loads one application, hides all factors
  2. step(action)    → reveal a factor OR make a final decision
  3. state           → full internal snapshot (ground truth included)

Reward design:
  Assess actions (reveal):
    +0.10  informative factor (strong signal direction)
    +0.05  neutral / non-informative reveal
    -0.05  duplicate reveal (factor already assessed)
    -0.10  wrong application_id or unknown action_type

  Decide actions (terminal):
    +1.00  correct decision   (easy / medium)
    +1.20  correct decision   (hard — harder task, higher reward)
    +0.30  refer decision     (partial credit; reasonable caution)
    -0.50  wrong decision     (costly error)
    -0.20  timeout            (max steps reached without deciding)

  Efficiency bonus (applied at episode end with the decide reward):
    +0.05 × (number of unrevealed factors at decision time)
    Fewer reveals = higher total reward.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Resolve tasks.json — checked in priority order
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent

_TASKS_CANDIDATES = [
    _HERE.parent / "data" / "tasks.json",   # sme-credit-env/data/tasks.json  ← primary
    _HERE.parent / "tasks.json",            # sme-credit-env/tasks.json (root copy)
    _HERE / "tasks.json",                   # sme-credit-env/env/tasks.json
    Path("tasks.json"),                     # cwd fallback
]


def _load_tasks() -> list[dict]:
    """
    Load tasks.json and return a flat list of application dicts.

    Handles both output formats of generate_dataset.py:
      Nested  → {"metadata": {...}, "tasks": {"easy": [...], ...}}
      Flat    → [{...}, ...]   (legacy / test format)

    For the nested format each app dict gets a "task_id" field
    stamped from its bucket key ("easy" / "medium" / "hard").
    """
    for candidate in _TASKS_CANDIDATES:
        if candidate.exists():
            with open(candidate) as fh:
                raw = json.load(fh)

            if isinstance(raw, dict) and "tasks" in raw:
                flat: list[dict] = []
                for tier, apps in raw["tasks"].items():
                    for app in apps:
                        app = dict(app)
                        app.setdefault("task_id", tier)
                        flat.append(app)
                return flat

            if isinstance(raw, list):
                return raw

            raise ValueError(
                f"Unrecognised tasks.json structure in {candidate}. "
                "Expected a list or a dict with a 'tasks' key."
            )

    raise FileNotFoundError(
        "tasks.json not found. Searched:\n"
        + "\n".join(f"  {p}" for p in _TASKS_CANDIDATES)
    )


# ---------------------------------------------------------------------------
# Import models — try direct import, fall back to inserting repo root
# ---------------------------------------------------------------------------
try:
    from models import (
        LoanAction,
        LoanObservation,
        LoanState,
        ACTION_TO_FACTOR,
        ASSESS_ACTIONS,
        DECIDE_ACTIONS,
        REVEALABLE_FACTORS,
        VALID_ACTIONS,
    )
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(_HERE.parent))
    from models import (
        LoanAction,
        LoanObservation,
        LoanState,
        ACTION_TO_FACTOR,
        ASSESS_ACTIONS,
        DECIDE_ACTIONS,
        REVEALABLE_FACTORS,
        VALID_ACTIONS,
    )


# ---------------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------------
_R_REVEAL_INFORMATIVE   = +0.10
_R_REVEAL_NEUTRAL       = +0.05
_R_REVEAL_DUPLICATE     = -0.05
_R_INVALID_ACTION       = -0.10

_R_DECIDE_CORRECT       = +1.00   # easy / medium
_R_DECIDE_CORRECT_HARD  = +1.20   # hard
_R_DECIDE_REFER         = +0.30   # always partial credit
_R_DECIDE_WRONG         = -0.50

_R_EFFICIENCY_PER_SAVED = +0.05   # per unrevealed factor at decision time
_R_TIMEOUT_PENALTY      = -0.20

_MAX_STEPS = 8


# ---------------------------------------------------------------------------
# Factor direction helper
# ---------------------------------------------------------------------------

def _compute_factor_directions(app: dict) -> dict[str, str]:
    """
    Return per-factor direction signals keyed by REVEALABLE_FACTORS names.

    generate_dataset.py stores factor_directions with keys:
        credit_score, dti, loan_to_revenue, business_age,
        cash_flow_volatility, collateral

    We map them to REVEALABLE_FACTORS keys:
        credit_score, dti, annual_revenue,
        business_age_years, cash_flow_volatility, collateral_value

    Falls back to computing from raw values if factor_directions absent.
    """
    raw: dict = app.get("factor_directions", {})

    if raw:
        return {
            "credit_score":         raw.get("credit_score",         "neutral"),
            "dti":                  raw.get("dti",                  "neutral"),
            "annual_revenue":       raw.get("loan_to_revenue",      "neutral"),
            "collateral_value":     raw.get("collateral",           "neutral"),
            "business_age_years":   raw.get("business_age",         "neutral"),
            "cash_flow_volatility": raw.get("cash_flow_volatility", "neutral"),
        }

    # Fallback: compute from raw values
    def _dir(p: float) -> str:
        if p < 0.25: return "positive"
        if p > 0.55: return "negative"
        return "neutral"

    ltr = app.get("loan_amount", 0) / max(app.get("annual_revenue", 1), 1)
    cov = app.get("collateral_value", 0) / max(app.get("loan_amount", 1), 1)

    def _cp(s): return max(0.0, min(1.0, (850 - s) / 550))
    def _dp(d): return 0.0 if d<=0.3 else 0.2 if d<=0.45 else 0.5 if d<=0.6 else min(1.0,0.85+(d-0.6)*0.75)
    def _lp(l): return 0.0 if l<=0.25 else 0.15 if l<=0.5 else 0.4 if l<=1.0 else 0.7 if l<=2.0 else 1.0
    def _ap(a): return 0.0 if a>=5 else 0.15 if a>=3 else 0.35 if a>=2 else 0.6 if a>=1 else 0.9
    def _vp(v): return min(1.0, v*1.2)

    return {
        "credit_score":         _dir(_cp(app.get("credit_score", 300))),
        "dti":                  _dir(_dp(app.get("dti", 1.0))),
        "annual_revenue":       _dir(_lp(ltr)),
        "collateral_value":     "positive" if cov>=0.8 else ("neutral" if cov>=0.4 else "negative"),
        "business_age_years":   _dir(_ap(app.get("business_age_years", 0))),
        "cash_flow_volatility": _dir(_vp(app.get("cash_flow_volatility", 1.0))),
    }


# ---------------------------------------------------------------------------
# LoanEnvironment
# ---------------------------------------------------------------------------

class LoanEnvironment:
    """
    Core RL environment for SME loan decisions.

    Usage
    -----
    env = LoanEnvironment()
    obs = env.reset("easy_01")         # or env.reset("easy")

    while not obs.done:
        action = LoanAction(
            action_type="assess_credit_score",
            application_id=obs.application_id,
        )
        obs = env.step(action)

    full_state = env.state             # includes ground truth
    """

    def __init__(self, tasks_path: Optional[str] = None):
        self._tasks:        list[dict]       = []
        self._tasks_index:  dict[str, dict]  = {}
        self._tasks_path                     = tasks_path
        self._loaded                         = False

        # Episode state — populated / reset by reset()
        self._app:                 dict       = {}
        self._episode_id:          str        = ""
        self._task_id:             str        = ""
        self._application_id:      str        = ""
        self._revealed:            dict       = {}
        self._step_count:          int        = 0
        self._cumulative_reward:   float      = 0.0
        self._done:                bool       = False
        self._final_decision:      Optional[str] = None
        self._action_log:          list       = []
        self._penalty_total:       float      = 0.0
        self._correct_assessments: int        = 0
        self._factor_directions:   dict       = {}
        self._ground_truth:        str        = ""
        self._risk_score:          float      = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self, task_id: str) -> LoanObservation:
        """
        Start a new episode.

        Parameters
        ----------
        task_id : str
            Full application_id ("easy_01", "hard_07") OR a difficulty
            bucket name ("easy" / "medium" / "hard") which returns the
            first application in that bucket.

        Returns
        -------
        LoanObservation with all financial factors hidden.
        """
        self._ensure_loaded()
        app = self._resolve_task(task_id)

        self._app               = app
        self._episode_id        = str(uuid.uuid4())
        self._task_id           = app["task_id"]
        self._application_id    = app["application_id"]
        self._revealed          = {}
        self._step_count        = 0
        self._cumulative_reward = 0.0
        self._done              = False
        self._final_decision    = None
        self._action_log        = []
        self._penalty_total     = 0.0
        self._correct_assessments = 0
        self._ground_truth      = app.get("decision", "")
        self._risk_score        = app.get("risk_score", 0.0)
        self._factor_directions = _compute_factor_directions(app)

        return self._make_obs(
            reward=0.0,
            feedback="Episode started. All factors hidden. Assess before deciding.",
            valid=True,
        )

    def step(self, action: LoanAction) -> LoanObservation:
        """
        Execute one agent action.

        Parameters
        ----------
        action : LoanAction
            action_type must be in VALID_ACTIONS.
            application_id must match current episode.

        Returns
        -------
        Updated LoanObservation with step reward and feedback.
        """
        if self._done:
            return self._make_obs(
                reward=0.0,
                feedback="Episode already done. Call reset() to start a new episode.",
                valid=False,
            )

        # ── Validate application_id ──────────────────────────────────
        if action.application_id != self._application_id:
            r = _R_INVALID_ACTION
            self._apply_reward(r)
            self._penalty_total += abs(r)
            self._log(action, r, valid=False)
            return self._make_obs(
                reward=r,
                feedback=(
                    f"Wrong application_id '{action.application_id}'. "
                    f"Expected '{self._application_id}'. Penalty: {r}."
                ),
                valid=False,
            )

        # ── Validate action_type ─────────────────────────────────────
        if action.action_type not in VALID_ACTIONS:
            r = _R_INVALID_ACTION
            self._apply_reward(r)
            self._penalty_total += abs(r)
            self._log(action, r, valid=False)
            return self._make_obs(
                reward=r,
                feedback=(
                    f"Unknown action_type '{action.action_type}'. "
                    f"Valid: {sorted(VALID_ACTIONS)}. Penalty: {r}."
                ),
                valid=False,
            )

        self._step_count += 1

        # ── Dispatch ─────────────────────────────────────────────────
        if action.action_type in ASSESS_ACTIONS:
            obs = self._handle_assess(action)
        else:
            obs = self._handle_decide(action)

        # ── Auto-timeout ─────────────────────────────────────────────
        if not self._done and self._step_count >= _MAX_STEPS:
            r = _R_TIMEOUT_PENALTY
            self._apply_reward(r)
            self._penalty_total += abs(r)
            self._done = True
            timeout_action = LoanAction(
                action_type="__timeout__",
                application_id=self._application_id,
            )
            self._log(timeout_action, r, valid=False)
            return self._make_obs(
                reward=r,
                feedback=(
                    f"Step budget ({_MAX_STEPS}) reached without a decision. "
                    f"Episode ended. Penalty: {r}."
                ),
                valid=False,
            )

        return obs

    @property
    def state(self) -> LoanState:
        """Full server-side state snapshot (includes ground truth)."""
        return LoanState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            metadata={},
            task_id=self._task_id,
            application_id=self._application_id,
            ground_truth_decision=self._ground_truth,
            ground_truth_risk_score=self._risk_score,
            factor_directions=self._factor_directions,
            factors_assessed=list(self._revealed.keys()),
            cumulative_reward=round(self._cumulative_reward, 4),
            final_decision=self._final_decision,
            penalty_total=round(self._penalty_total, 4),
            correct_assessments=self._correct_assessments,
            action_log=list(self._action_log),
        )

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_assess(self, action: LoanAction) -> LoanObservation:
        factor_key = ACTION_TO_FACTOR[action.action_type]

        # Duplicate
        if factor_key in self._revealed:
            r = _R_REVEAL_DUPLICATE
            self._apply_reward(r)
            self._penalty_total += abs(r)
            self._log(action, r, valid=False)
            return self._make_obs(
                reward=r,
                feedback=f"'{factor_key}' already assessed. Duplicate penalty: {r}.",
                valid=False,
            )

        # Reveal
        value = self._app.get(factor_key)
        self._revealed[factor_key] = value

        direction = self._factor_directions.get(factor_key, "neutral")
        if direction in ("positive", "negative"):
            r = _R_REVEAL_INFORMATIVE
            self._correct_assessments += 1
            tag = f"[{direction.upper()}]"
        else:
            r = _R_REVEAL_NEUTRAL
            tag = "[NEUTRAL]"

        self._apply_reward(r)
        self._log(action, r, valid=True)

        note = ""
        if factor_key == "annual_revenue" and value and value > 0:
            ltr = round(self._app["loan_amount"] / value, 4)
            self._revealed["loan_to_revenue"] = ltr
            note = f" | loan_to_revenue = {ltr:.3f}"

        return self._make_obs(
            reward=r,
            feedback=f"Revealed {factor_key} = {value} {tag}. Reward: +{r:.2f}.{note}",
            valid=True,
        )

    def _handle_decide(self, action: LoanAction) -> LoanObservation:
        decision_map = {
            "decide_approve": "approve",
            "decide_reject":  "reject",
            "decide_refer":   "refer",
        }
        agent_decision = decision_map[action.action_type]
        self._final_decision = agent_decision

        # Efficiency bonus
        n_unrevealed = len(REVEALABLE_FACTORS) - len(
            [k for k in self._revealed if k in REVEALABLE_FACTORS]
        )
        efficiency_bonus = round(_R_EFFICIENCY_PER_SAVED * n_unrevealed, 4)

        if agent_decision == "refer":
            base = _R_DECIDE_REFER
            outcome = "REFER (partial credit)"
        elif agent_decision == self._ground_truth:
            base = _R_DECIDE_CORRECT_HARD if self._task_id == "hard" else _R_DECIDE_CORRECT
            outcome = f"CORRECT ✓ (gt: {self._ground_truth})"
        else:
            base = _R_DECIDE_WRONG
            self._penalty_total += abs(_R_DECIDE_WRONG)
            outcome = f"WRONG ✗ (agent={agent_decision}, gt={self._ground_truth})"

        total = base + efficiency_bonus
        self._apply_reward(total)
        self._done = True
        self._log(action, total, valid=True)

        return self._make_obs(
            reward=total,
            feedback=(
                f"Decision: {agent_decision.upper()}. {outcome}. "
                f"Base: {base:+.2f}, efficiency bonus: +{efficiency_bonus:.2f}, "
                f"step reward: {total:+.2f}. "
                f"Cumulative: {self._cumulative_reward:.4f}."
            ),
            valid=True,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_obs(
        self,
        reward: float,
        feedback: str,
        valid: bool,
    ) -> LoanObservation:
        assessed  = [k for k in REVEALABLE_FACTORS if k in self._revealed]
        remaining = [k for k in REVEALABLE_FACTORS if k not in self._revealed]
        return LoanObservation(
            done=self._done,
            reward=round(reward, 4),
            metadata={},
            application_id=self._application_id,
            task_id=self._task_id,
            business_name=self._app.get("business_name", ""),
            sector=self._app.get("sector", ""),
            loan_amount=self._app.get("loan_amount", 0.0),
            loan_to_revenue=self._revealed.get("loan_to_revenue"),
            annual_revenue=self._revealed.get("annual_revenue"),
            credit_score=self._revealed.get("credit_score"),
            dti=self._revealed.get("dti"),
            collateral_value=self._revealed.get("collateral_value"),
            business_age_years=self._revealed.get("business_age_years"),
            cash_flow_volatility=self._revealed.get("cash_flow_volatility"),
            factors_assessed=assessed,
            factors_remaining=remaining,
            step_count=self._step_count,
            max_steps=_MAX_STEPS,
            cumulative_reward=round(self._cumulative_reward, 4),
            last_action_valid=valid,
            feedback=feedback,
            final_decision=self._final_decision,
        )

    def _apply_reward(self, r: float) -> None:
        self._cumulative_reward = round(self._cumulative_reward + r, 4)

    def _log(self, action: LoanAction, reward: float, valid: bool) -> None:
        self._action_log.append({
            "step":           self._step_count,
            "action_type":    action.action_type,
            "application_id": action.application_id,
            "reward":         round(reward, 4),
            "valid":          valid,
            "cumulative":     self._cumulative_reward,
        })

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self._tasks_path:
            with open(self._tasks_path) as fh:
                raw = json.load(fh)
            if isinstance(raw, dict) and "tasks" in raw:
                flat: list[dict] = []
                for tier, apps in raw["tasks"].items():
                    for app in apps:
                        a = dict(app)
                        a.setdefault("task_id", tier)
                        flat.append(a)
                self._tasks = flat
            else:
                self._tasks = raw
        else:
            self._tasks = _load_tasks()
        self._tasks_index = {t["application_id"]: t for t in self._tasks}
        self._loaded = True

    def _resolve_task(self, task_id: str) -> dict:
        """Accept full application_id OR a difficulty bucket name."""
        if task_id in self._tasks_index:
            return self._tasks_index[task_id]
        for task in self._tasks:
            if task.get("task_id") == task_id:
                return task
        raise ValueError(
            f"Task '{task_id}' not found. "
            f"Use a full application_id (e.g. 'easy_01') or a bucket "
            f"name ('easy' / 'medium' / 'hard'). "
            f"Available: {sorted(self._tasks_index.keys())}"
        )


# ---------------------------------------------------------------------------
# Smoke-test — python env/environment.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 56)
    print("  LoanEnvironment — smoke test")
    print("=" * 56)

    env   = LoanEnvironment()
    tasks = _load_tasks()
    tid   = tasks[0]["application_id"]
    print(f"\nTask: {tid}")

    obs = env.reset(tid)
    assert not obs.done and obs.credit_score is None
    print(f"  reset()  loan=₹{obs.loan_amount:,.0f}  credit_score=None ✓")

    obs = env.step(LoanAction(action_type="assess_credit_score", application_id=tid))
    assert obs.credit_score is not None and obs.reward > 0
    print(f"  assess_credit_score  credit={obs.credit_score}  reward={obs.reward:+.2f} ✓")

    obs = env.step(LoanAction(action_type="assess_credit_score", application_id=tid))
    assert obs.reward < 0
    print(f"  duplicate assess  reward={obs.reward:+.2f}  (penalty) ✓")

    obs = env.step(LoanAction(action_type="decide_approve", application_id=tid))
    assert obs.done and obs.final_decision == "approve"
    print(f"  decide_approve  done=True  final={obs.final_decision} ✓")

    st = env.state
    assert st.ground_truth_decision in ("approve", "reject", "refer")
    print(f"  state()  gt={st.ground_truth_decision}  cumul={st.cumulative_reward} ✓")
    print("\n  All assertions passed.\n")