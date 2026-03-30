from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Optional

# ✅ LOAD ENV FILE
from dotenv import load_dotenv
load_dotenv()

# ✅ REQUIRED CLIENT
from openai import OpenAI

# =========================
# ENV VARIABLES (MANDATORY)
# =========================
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models import (
    LoanAction,
    LoanObservation,
    VALID_ACTIONS,
    ACTION_TO_FACTOR,
)
from env.environment import LoanEnvironment, _load_tasks
from env.graders import grade

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are an expert SME (Small and Medium Enterprise) credit underwriter.
You are evaluating a loan application step-by-step.

## YOUR TASK
Decide whether to APPROVE, REJECT, or REFER a business loan application.
You must gather financial evidence first, then make your decision.

## AVAILABLE ACTIONS
Assess actions (reveal one hidden factor):
  - assess_revenue        → reveals annual_revenue (GBP)
  - assess_credit_score   → reveals credit_score (300–850)
  - assess_dti            → reveals dti (debt-to-income ratio 0.0–1.0)
  - assess_collateral     → reveals collateral_value (GBP, 0 = unsecured)
  - assess_business_age   → reveals business_age_years
  - assess_cash_flow      → reveals cash_flow_volatility (0=stable, 1=volatile)

Decision actions (ENDS the episode — choose one):
  - decide_approve        → approve the loan
  - decide_reject         → reject the loan
  - decide_refer          → refer to senior underwriter (borderline cases)

## RISK FRAMEWORK (use this to decide)
APPROVE when risk is low:
  ✓ Credit score ≥ 650
  ✓ DTI ≤ 0.40
  ✓ Loan-to-revenue ratio ≤ 0.50
  ✓ Business age ≥ 2 years
  ✓ Cash flow volatility ≤ 0.40
  ✓ Strong collateral coverage

REJECT when risk is high:
  ✗ Credit score < 500 (hard floor — always reject)
  ✗ DTI > 0.80
  ✗ Business age < 1 year with no collateral and large loan
  ✗ Multiple severe negatives together

REFER when signals are mixed:
  - Some factors suggest approval, others suggest rejection
  - Risk is borderline (cannot clearly approve or reject)

## EFFICIENCY RULE
You earn a bonus for FEWER reveals. Don't assess all 6 factors if you can
decide confidently from fewer. A decisive agent scores higher than a
thorough-but-slow one.

## RESPONSE FORMAT
Respond with ONLY a JSON object — no prose, no markdown:
{
  "reasoning": "one sentence explaining your thinking",
  "action_type": "assess_credit_score"
}

The action_type must be exactly one of the valid actions listed above.
""".strip()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs_to_prompt(obs: LoanObservation) -> str:
    """Convert a LoanObservation into a concise text prompt for the LLM."""
    lines = [
        f"APPLICATION: {obs.application_id}  |  Business: {obs.business_name}",
        f"Sector: {obs.sector}  |  Loan requested: £{obs.loan_amount:,.0f}",
        f"Step: {obs.step_count}/{obs.max_steps}  |  Cumulative reward: {obs.cumulative_reward:.3f}",
        "",
        "REVEALED FACTORS:",
    ]

    factor_map = {
        "annual_revenue":       ("Annual Revenue",       lambda v: f"£{v:,.0f}"),
        "credit_score":         ("Credit Score",         lambda v: str(v)),
        "dti":                  ("DTI Ratio",            lambda v: f"{v:.1%}"),
        "collateral_value":     ("Collateral Value",     lambda v: f"£{v:,.0f}"),
        "business_age_years":   ("Business Age",         lambda v: f"{v:.1f} years"),
        "cash_flow_volatility": ("Cash Flow Volatility", lambda v: f"{v:.2f}"),
    }

    revealed_any = False
    for key, (label, fmt) in factor_map.items():
        val = getattr(obs, key, None)
        if val is not None:
            lines.append(f"  {label:28s}: {fmt(val)}")
            revealed_any = True

    if obs.loan_to_revenue is not None:
        lines.append(f"  {'Loan-to-Revenue':28s}: {obs.loan_to_revenue:.2f}x")

    if not revealed_any:
        lines.append("  (none revealed yet)")

    lines += [
        "",
        f"Factors still hidden: {obs.factors_remaining}",
        f"Last action valid: {obs.last_action_valid}",
    ]
    if obs.feedback:
        lines.append(f"Last feedback: {obs.feedback}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# UPDATED LLM CALL (OpenAI + HF compatible)
# ---------------------------------------------------------------------------

def _call_llm(prompt: str, api_key: str, model: str = None) -> dict:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=api_key)

        completion = client.chat.completions.create(
            model=model or MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=256
        )

        text = completion.choices[0].message.content.strip()

        # Clean markdown if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        return json.loads(text.strip())

    except Exception as e:
        print(f"[LLM error: {e}] → fallback")
        return {}

# ---------------------------------------------------------------------------
# HEURISTIC AGENT (UNCHANGED)
# ---------------------------------------------------------------------------
class HeuristicAgent:
    """
    Deterministic rule-based agent.
    Mirrors the risk formula from generate_dataset.py so it should achieve
    near-perfect accuracy with minimal reveals.
    """

    # Priority order for reveals — most informative factors first
    REVEAL_PRIORITY = [
        "assess_credit_score",   # hard floor check
        "assess_dti",            # second most predictive
        "assess_revenue",        # needed for LTR
        "assess_business_age",   # hard floor check
        "assess_collateral",     # discount factor
        "assess_cash_flow",      # least weight
    ]

    def choose_action(self, obs: LoanObservation) -> str:
        """Return an action_type string based on current observation."""

        # --- Hard-floor checks on revealed factors ---
        if obs.credit_score is not None and obs.credit_score < 500:
            return "decide_reject"

        if obs.dti is not None and obs.dti > 0.80:
            return "decide_reject"

        if (
            obs.business_age_years is not None
            and obs.business_age_years < 1.0
            and obs.collateral_value is not None
            and obs.collateral_value < obs.loan_amount * 0.5
        ):
            return "decide_reject"

        # --- Count strong signals if we have ≥ 3 factors ---
        n_revealed = len(obs.factors_assessed)

        if n_revealed >= 3:
            positive_signals = 0
            negative_signals = 0

            if obs.credit_score is not None:
                if obs.credit_score >= 650:   positive_signals += 1
                elif obs.credit_score < 550:  negative_signals += 1

            if obs.dti is not None:
                if obs.dti <= 0.40:   positive_signals += 1
                elif obs.dti > 0.60:  negative_signals += 1

            if obs.loan_to_revenue is not None:
                if obs.loan_to_revenue <= 0.50:   positive_signals += 1
                elif obs.loan_to_revenue > 1.0:   negative_signals += 1

            if obs.business_age_years is not None:
                if obs.business_age_years >= 3:   positive_signals += 1
                elif obs.business_age_years < 1:  negative_signals += 1

            if obs.cash_flow_volatility is not None:
                if obs.cash_flow_volatility <= 0.40:  positive_signals += 1
                elif obs.cash_flow_volatility > 0.60: negative_signals += 1

            if obs.collateral_value is not None:
                cov = obs.collateral_value / max(obs.loan_amount, 1)
                if cov >= 0.8:    positive_signals += 1
                elif cov < 0.2:   negative_signals += 1

            total_signals = positive_signals + negative_signals
            if total_signals > 0:
                pos_ratio = positive_signals / total_signals
                if pos_ratio >= 0.75 and n_revealed >= 3:
                    return "decide_approve"
                if pos_ratio <= 0.30 and n_revealed >= 3:
                    return "decide_reject"
                # Borderline — refer if we've seen enough
                if n_revealed >= 5:
                    return "decide_refer"

        # --- Reveal next hidden factor in priority order ---
        revealed = set(obs.factors_assessed)
        for action in self.REVEAL_PRIORITY:
            factor = ACTION_TO_FACTOR[action]
            if factor not in revealed:
                return action

        # --- All factors revealed — make best decision ---
        return "decide_refer"

# ---------------------------------------------------------------------------
# LLM Agent (wraps heuristic with LLM override)
# ---------------------------------------------------------------------------

class LLMAgent:
    """
    Uses Claude to pick actions. Falls back to heuristic on API errors.
    """

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model = model
        self._heuristic = HeuristicAgent()

    def choose_action(self, obs: LoanObservation) -> tuple[str, str]:
        """Returns (action_type, reasoning)."""
        prompt = _obs_to_prompt(obs)
        result = _call_llm(prompt, self.api_key, self.model)

        action_type = result.get("action_type", "")
        reasoning   = result.get("reasoning", "LLM fallback")

        # Validate — if LLM returned garbage, use heuristic
        if action_type not in VALID_ACTIONS:
            action_type = self._heuristic.choose_action(obs)
            reasoning = f"[heuristic fallback] {reasoning}"

        return action_type, reasoning


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode_local(
    env: LoanEnvironment,
    task_id: str,
    agent,
    verbose: bool = True,
) -> dict:
    """
    Run one full episode locally (no HTTP).
    Returns a result dict with score and episode metadata.
    """
    obs = env.reset(task_id)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Episode: {obs.application_id}  [{obs.task_id.upper()}]")
        print(f"  Business: {obs.business_name}  |  Sector: {obs.sector}")
        print(f"  Loan: £{obs.loan_amount:,.0f}")
        print(f"{'='*60}")

    while not obs.done:
        if isinstance(agent, LLMAgent):
            action_type, reasoning = agent.choose_action(obs)
            if verbose:
                print(f"\n  Step {obs.step_count + 1}: {action_type}")
                print(f"  Reasoning: {reasoning}")
        else:
            action_type = agent.choose_action(obs)
            if verbose:
                print(f"\n  Step {obs.step_count + 1}: {action_type}")

        action = LoanAction(action_type=action_type, application_id=obs.application_id)
        obs = env.step(action)

        if verbose:
            print(f"  → Reward: {obs.reward:+.3f}  |  Cumulative: {obs.cumulative_reward:.3f}")
            if obs.feedback:
                print(f"  → {obs.feedback}")

    # Grade the episode
    state = env.state
    episode_score = grade(
        action_log=state.action_log,
        ground_truth=state.ground_truth_decision,
        task_id=state.task_id,
    )

    result = {
        "application_id":    obs.application_id,
        "task_id":           obs.task_id,
        "final_decision":    obs.final_decision,
        "ground_truth":      state.ground_truth_decision,
        "correct":           obs.final_decision == state.ground_truth_decision,
        "cumulative_reward": state.cumulative_reward,
        "n_reveals":         len([e for e in state.action_log if e["action_type"].startswith("assess_")]),
        "n_steps":           state.step_count,
        "grade_score":       episode_score,
        "penalties":         state.penalty_total,
    }

    if verbose:
        correct_str = "✓ CORRECT" if result["correct"] else "✗ WRONG"
        print(f"\n  {'─'*50}")
        print(f"  Decision: {obs.final_decision.upper():10s}  Ground truth: {state.ground_truth_decision.upper()}")
        print(f"  Outcome:  {correct_str}")
        print(f"  Reward:   {state.cumulative_reward:.4f}   Grade score: {episode_score:.4f}")
        print(f"  Reveals:  {result['n_reveals']} / 6   Steps: {result['n_steps']}")

    return result


def run_episode_remote(
    base_url: str,
    task_id: str,
    agent,
    verbose: bool = True,
) -> dict:
    """
    Run one full episode against a remote FastAPI server.
    """
    import requests

    def _reset(tid: str) -> dict:
        r = requests.post(f"{base_url}/reset", json={"task_id": tid}, timeout=10)
        r.raise_for_status()
        return r.json()

    def _step(action_type: str, application_id: str) -> dict:
        r = requests.post(
            f"{base_url}/step",
            json={"action_type": action_type, "application_id": application_id},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()

    def _state() -> dict:
        r = requests.get(f"{base_url}/state", timeout=10)
        r.raise_for_status()
        return r.json()

    def _grade(action_log, ground_truth, task_id_str) -> dict:
        r = requests.post(
            f"{base_url}/grade",
            json={"action_log": action_log, "ground_truth": ground_truth, "task_id": task_id_str},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()

    obs_dict = _reset(task_id)
    obs = _dict_to_obs(obs_dict)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Episode (remote): {obs.application_id}  [{obs.task_id.upper()}]")
        print(f"  Business: {obs.business_name}  |  Loan: £{obs.loan_amount:,.0f}")
        print(f"{'='*60}")

    while not obs.done:
        if isinstance(agent, LLMAgent):
            action_type, reasoning = agent.choose_action(obs)
            if verbose:
                print(f"\n  Step {obs.step_count + 1}: {action_type}")
                print(f"  Reasoning: {reasoning}")
        else:
            action_type = agent.choose_action(obs)
            if verbose:
                print(f"\n  Step {obs.step_count + 1}: {action_type}")

        obs_dict = _step(action_type, obs.application_id)
        obs = _dict_to_obs(obs_dict)

        if verbose:
            print(f"  → Reward: {obs.reward:+.3f}  |  {obs.feedback[:80]}")

    state_dict = _state()
    grade_resp  = _grade(
        state_dict.get("action_log", []),
        state_dict.get("ground_truth_decision", ""),
        state_dict.get("task_id", "easy"),
    )

    result = {
        "application_id":    obs.application_id,
        "task_id":           obs.task_id,
        "final_decision":    obs.final_decision,
        "ground_truth":      state_dict.get("ground_truth_decision"),
        "correct":           grade_resp.get("correct", False),
        "cumulative_reward": state_dict.get("cumulative_reward", 0),
        "n_reveals":         grade_resp.get("n_reveals", 0),
        "n_steps":           state_dict.get("step_count", 0),
        "grade_score":       grade_resp.get("score", 0.0),
        "penalties":         state_dict.get("penalty_total", 0.0),
    }

    if verbose:
        correct_str = "✓ CORRECT" if result["correct"] else "✗ WRONG"
        print(f"\n  {'─'*50}")
        print(f"  Decision: {obs.final_decision.upper():10s}  Ground truth: {result['ground_truth'].upper()}")
        print(f"  Outcome:  {correct_str}")
        print(f"  Grade score: {result['grade_score']:.4f}")

    return result


def _dict_to_obs(d: dict) -> LoanObservation:
    """Reconstruct a LoanObservation from a server JSON response dict."""
    # Remove keys that aren't LoanObservation fields
    valid_fields = {f.name for f in dataclasses.fields(LoanObservation)}
    clean = {k: v for k, v in d.items() if k in valid_fields}
    return LoanObservation(**clean)


# ---------------------------------------------------------------------------
# Full evaluation across a tier or all tiers
# ---------------------------------------------------------------------------

def run_evaluation(
    agent,
    mode: str = "local",
    tier: Optional[str] = None,
    base_url: str = "http://localhost:7860",
    verbose: bool = False,
) -> dict:
    """
    Run every task in a tier (or all tiers) and print a summary table.

    Returns a dict with per-tier and overall statistics.
    """
    all_tasks = _load_tasks()

    if tier:
        tasks_to_run = [t for t in all_tasks if t["task_id"] == tier]
    else:
        tasks_to_run = all_tasks

    if not tasks_to_run:
        print(f"No tasks found for tier '{tier}'")
        return {}

    env = LoanEnvironment() if mode == "local" else None
    results = []

    print(f"\n{'='*60}")
    print(f"  SME Credit Risk Agent — Evaluation")
    print(f"  Mode: {mode.upper()}   Agent: {type(agent).__name__}")
    print(f"  Tasks: {len(tasks_to_run)}")
    print(f"{'='*60}\n")

    for task in tasks_to_run:
        tid = task["application_id"]
        try:
            if mode == "local":
                result = run_episode_local(env, tid, agent, verbose=verbose)
            else:
                result = run_episode_remote(base_url, tid, agent, verbose=verbose)
            results.append(result)
            status = "✓" if result["correct"] else "✗"
            print(
                f"  {status} {tid:12s} [{result['task_id']:6s}]  "
                f"decision={result['final_decision']:7s}  "
                f"gt={result['ground_truth']:7s}  "
                f"score={result['grade_score']:.3f}  "
                f"reveals={result['n_reveals']}"
            )
        except Exception as e:
            print(f"  ! {tid:12s} ERROR: {e}")

    if not results:
        return {}

    # Summary by tier
    print(f"\n{'─'*60}")
    print(f"  SUMMARY")
    print(f"{'─'*60}")

    by_tier: dict[str, list] = {}
    for r in results:
        by_tier.setdefault(r["task_id"], []).append(r)

    overall_scores = []
    for t in ["easy", "medium", "hard"]:
        if t not in by_tier:
            continue
        tier_results = by_tier[t]
        n = len(tier_results)
        n_correct  = sum(1 for r in tier_results if r["correct"])
        avg_score  = sum(r["grade_score"] for r in tier_results) / n
        avg_reveals = sum(r["n_reveals"] for r in tier_results) / n
        avg_reward  = sum(r["cumulative_reward"] for r in tier_results) / n
        overall_scores.extend(r["grade_score"] for r in tier_results)

        print(
            f"  {t.upper():6s}  accuracy={n_correct}/{n}  "
            f"avg_score={avg_score:.3f}  "
            f"avg_reveals={avg_reveals:.1f}  "
            f"avg_reward={avg_reward:.3f}"
        )

    overall_avg = sum(overall_scores) / len(overall_scores)
    n_total = len(results)
    n_correct_total = sum(1 for r in results if r["correct"])
    print(f"\n  OVERALL  accuracy={n_correct_total}/{n_total}  avg_score={overall_avg:.3f}")
    print(f"{'='*60}\n")

    return {
        "results":      results,
        "by_tier":      {t: by_tier[t] for t in by_tier},
        "overall_score": overall_avg,
        "accuracy":     n_correct_total / n_total,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--tier", default=None)
    args = parser.parse_args()

    if args.llm:
        # 🔥 FIXED API KEY LOADING
        api_key = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY")

        if not api_key:
            print("ERROR: Set HF_TOKEN or API_KEY in .env file")
            sys.exit(1)

        agent = LLMAgent(api_key=api_key, model=MODEL_NAME)
        print(f"Agent: LLM ({MODEL_NAME})")

    else:
        agent = HeuristicAgent()
        print("Agent: Heuristic")

    run_evaluation(agent, tier=args.tier)


if __name__ == "__main__":
    main()