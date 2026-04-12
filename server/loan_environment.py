"""
server/loan_environment.py — LoanEnvironment for OpenEnv
=========================================================
Mirrors the reference pattern exactly:
  - Inherits from openenv.core.env_server.Environment
  - reset() uses (self, seed=None, episode_id=None, **kwargs) signature
  - task_id extracted from **kwargs, NOT as a positional parameter
  - step() and state follow the same typed interface

This file is the server-side environment class, analogous to:
    reference: server/aitrade_environment.py

Your existing business logic (factor revealing, reward shaping, graders)
is unchanged — only the class signature and inheritance are fixed.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Optional

from openenv.core.env_server import Environment

import sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models import (
    LoanAction, LoanObservation, LoanState,
    ACTION_TO_FACTOR, ASSESS_ACTIONS, DECIDE_ACTIONS,
    REVEALABLE_FACTORS, VALID_ACTIONS,
)
from tasks.environment import _load_tasks, _compute_factor_directions

# ---------------------------------------------------------------------------
# Reward constants — unchanged from your original
# ---------------------------------------------------------------------------
_R_REVEAL_INFORMATIVE   = +0.10
_R_REVEAL_NEUTRAL       = +0.05
_R_REVEAL_DUPLICATE     = -0.05
_R_INVALID_ACTION       = -0.10
_R_DECIDE_CORRECT       = +1.00
_R_DECIDE_CORRECT_HARD  = +1.20
_R_DECIDE_REFER         = +0.30
_R_DECIDE_WRONG         = -0.50
_R_EFFICIENCY_PER_SAVED = +0.05
_R_TIMEOUT_PENALTY      = -0.20
_MAX_STEPS = 8


class LoanEnvironment(Environment[LoanAction, LoanObservation, LoanState]):
    """
    OpenEnv-compatible SME loan underwriting environment.

    Inherits from Environment[Action, Observation, State] exactly as the
    reference AiTradeEnvironment does. This makes create_app() work.
    """

    # Allows multiple concurrent WebSocket sessions (one env per session)
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._tasks:             list[dict]    = []
        self._tasks_index:       dict[str, dict] = {}
        self._loaded:            bool          = False
        self._app:               dict          = {}
        self._episode_id:        str           = ""
        self._task_id:           str           = ""
        self._application_id:    str           = ""
        self._revealed:          dict          = {}
        self._step_count:        int           = 0
        self._cumulative_reward: float         = 0.0
        self._done:              bool          = False
        self._final_decision:    Optional[str] = None
        self._action_log:        list          = []
        self._penalty_total:     float         = 0.0
        self._correct_assessments: int         = 0
        self._factor_directions: dict          = {}
        self._ground_truth:      str           = ""
        self._risk_score:        float         = 0.0

    # ------------------------------------------------------------------
    # OpenEnv required interface — signatures MUST match exactly
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> LoanObservation:
        """
        Reset to start of a new episode.

        task_id comes from **kwargs, matching OpenEnv's call convention:
            env.reset(**{"task_id": "easy_01"})   ← from POST /reset body
            env.reset()                            ← empty body, uses default
        """
        # Extract task_id from kwargs — this is the OpenEnv pattern
        task_id = kwargs.get("task_id") or "easy_01"

        self._ensure_loaded()
        app = self._resolve_task(task_id)

        self._app               = app
        self._episode_id        = episode_id or str(uuid.uuid4())
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
        """Execute one action. Returns LoanObservation."""
        if self._done:
            return self._make_obs(0.0, "Episode already done. Call reset().", False)

        if action.application_id != self._application_id:
            r = _R_INVALID_ACTION
            self._apply_reward(r)
            self._penalty_total += abs(r)
            self._log(action, r, valid=False)
            return self._make_obs(r,
                f"Wrong application_id '{action.application_id}'. "
                f"Expected '{self._application_id}'.", False)

        if action.action_type not in VALID_ACTIONS:
            r = _R_INVALID_ACTION
            self._apply_reward(r)
            self._penalty_total += abs(r)
            self._log(action, r, valid=False)
            return self._make_obs(r,
                f"Unknown action '{action.action_type}'. Valid: {sorted(VALID_ACTIONS)}", False)

        self._step_count += 1

        if action.action_type in ASSESS_ACTIONS:
            obs = self._handle_assess(action)
        else:
            obs = self._handle_decide(action)

        if not self._done and self._step_count >= _MAX_STEPS:
            r = _R_TIMEOUT_PENALTY
            self._apply_reward(r)
            self._penalty_total += abs(r)
            self._done = True
            self._log(LoanAction(action_type="__timeout__",
                                 application_id=self._application_id), r, False)
            return self._make_obs(r,
                f"Step budget ({_MAX_STEPS}) reached without a decision.", False)

        return obs

    @property
    def state(self) -> LoanState:
        """Full state snapshot including ground truth."""
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
    # Action handlers — business logic unchanged
    # ------------------------------------------------------------------

    def _handle_assess(self, action: LoanAction) -> LoanObservation:
        factor_key = ACTION_TO_FACTOR[action.action_type]

        if factor_key in self._revealed:
            r = _R_REVEAL_DUPLICATE
            self._apply_reward(r)
            self._penalty_total += abs(r)
            self._log(action, r, valid=False)
            return self._make_obs(r, f"'{factor_key}' already assessed. Penalty: {r}.", False)

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

        return self._make_obs(r,
            f"Revealed {factor_key} = {value} {tag}. Reward: +{r:.2f}.{note}", True)

    def _handle_decide(self, action: LoanAction) -> LoanObservation:
        decision_map = {"decide_approve": "approve",
                        "decide_reject": "reject", "decide_refer": "refer"}
        agent_decision = decision_map[action.action_type]
        self._final_decision = agent_decision

        n_unrevealed = len(REVEALABLE_FACTORS) - len(
            [k for k in self._revealed if k in REVEALABLE_FACTORS])
        efficiency_bonus = round(_R_EFFICIENCY_PER_SAVED * n_unrevealed, 4)

        if agent_decision == "refer":
            base, outcome = _R_DECIDE_REFER, "REFER (partial credit)"
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

        return self._make_obs(total,
            f"Decision: {agent_decision.upper()}. {outcome}. "
            f"Base: {base:+.2f}, efficiency: +{efficiency_bonus:.2f}, "
            f"step: {total:+.2f}. Cumulative: {self._cumulative_reward:.4f}.", True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_obs(self, reward: float, feedback: str, valid: bool) -> LoanObservation:
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
            "step": self._step_count, "action_type": action.action_type,
            "application_id": action.application_id,
            "reward": round(reward, 4), "valid": valid,
            "cumulative": self._cumulative_reward,
        })

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._tasks = _load_tasks()
        self._tasks_index = {t["application_id"]: t for t in self._tasks}
        self._loaded = True

    def _resolve_task(self, task_id: str) -> dict:
        if task_id in self._tasks_index:
            return self._tasks_index[task_id]
        for task in self._tasks:
            if task.get("task_id") == task_id:
                return task
        raise ValueError(
            f"Task '{task_id}' not found. "
            f"Use 'easy_01' or bucket name 'easy'/'medium'/'hard'. "
            f"Available: {sorted(self._tasks_index.keys())}"
        )