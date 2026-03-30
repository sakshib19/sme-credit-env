"""
inference.py — SME Credit Risk RL Agent
=========================================
A rule-driven + LLM-hybrid agent that plays full episodes.

Modes:
  local   → talks directly to LoanEnvironment (no server needed)
  remote  → talks to the FastAPI server via HTTP

MANDATORY environment variables (hackathon requirement):
  API_BASE_URL   — LLM API endpoint, e.g. https://router.huggingface.co/v1
  MODEL_NAME     — model identifier,  e.g. mistralai/Mistral-7B-Instruct-v0.2
  HF_TOKEN       — Hugging Face / API key

Usage
-----
  python inference.py                       # heuristic, local, easy tier
  python inference.py --tier hard           # heuristic, hard tier
  python inference.py --eval                # eval all tasks in default tier
  python inference.py --eval --tier hard    # eval all hard tasks
  python inference.py --llm --tier medium   # LLM agent, medium tier
  python inference.py --mode remote --eval  # eval against running server
  python inference.py --task easy_01        # single task by ID
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Mandatory env vars — read at module load (hackathon requirement)
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "mistralai/Mistral-7B-Instruct-v0.2")
API_KEY      = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY") or ""

# ---------------------------------------------------------------------------
# OpenAI-compatible client (MANDATORY for hackathon)
# ---------------------------------------------------------------------------
from openai import OpenAI as _OpenAIClient

# ---------------------------------------------------------------------------
# Path bootstrap — make models / env importable from any cwd
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models import (
    LoanAction,
    LoanObservation,
    VALID_ACTIONS,
    ACTION_TO_FACTOR,
    REVEALABLE_FACTORS,
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

Decision actions (ENDS the episode):
  - decide_approve  → approve the loan
  - decide_reject   → reject the loan
  - decide_refer    → refer to senior underwriter (borderline cases)

## RISK FRAMEWORK
APPROVE when risk is LOW:
  ✓ credit_score ≥ 650
  ✓ dti ≤ 0.40
  ✓ loan-to-revenue ratio ≤ 0.50
  ✓ business_age_years ≥ 2
  ✓ cash_flow_volatility ≤ 0.40

REJECT when risk is HIGH:
  ✗ credit_score < 500 (hard floor — always reject)
  ✗ dti > 0.80
  ✗ very young business (<1 yr) with large loan and no collateral

REFER when signals are mixed / borderline.

## EFFICIENCY RULE
You earn a bonus for FEWER reveals. Decide as soon as you have enough
evidence — don't reveal all 6 factors if 2-3 are sufficient.

## RESPONSE FORMAT
Respond with ONE JSON object only — no prose, no markdown fences:
{"reasoning": "one sentence", "action_type": "assess_credit_score"}
""".strip()


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _obs_to_prompt(obs: LoanObservation) -> str:
    lines = [
        f"APPLICATION: {obs.application_id}  |  {obs.business_name}  ({obs.sector})",
        f"Loan requested: £{obs.loan_amount:,.0f}",
        f"Step: {obs.step_count}/{obs.max_steps}  |  Cumulative reward: {obs.cumulative_reward:.3f}",
        "",
        "REVEALED FACTORS:",
    ]
    factor_display = {
        "annual_revenue":       ("Annual Revenue",       lambda v: f"£{v:,.0f}"),
        "credit_score":         ("Credit Score",         lambda v: str(v)),
        "dti":                  ("DTI Ratio",            lambda v: f"{v:.1%}"),
        "collateral_value":     ("Collateral Value",     lambda v: f"£{v:,.0f}"),
        "business_age_years":   ("Business Age",         lambda v: f"{v:.1f} yrs"),
        "cash_flow_volatility": ("Cash Flow Volatility", lambda v: f"{v:.2f}"),
    }
    any_revealed = False
    for key, (label, fmt) in factor_display.items():
        val = getattr(obs, key, None)
        if val is not None:
            lines.append(f"  {label:28s}: {fmt(val)}")
            any_revealed = True
    if obs.loan_to_revenue is not None:
        lines.append(f"  {'Loan-to-Revenue':28s}: {obs.loan_to_revenue:.2f}x")
    if not any_revealed:
        lines.append("  (none revealed yet)")
    lines += [
        "",
        f"Factors still hidden: {obs.factors_remaining}",
        f"Last feedback: {obs.feedback}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM call — uses OpenAI-compatible client with mandatory env vars
# ---------------------------------------------------------------------------

def _call_llm(prompt: str) -> dict:
    """
    Call the LLM using the OpenAI-compatible client.
    Reads API_BASE_URL, MODEL_NAME, API_KEY from module-level env vars.
    Returns parsed JSON dict, or {} on any error (triggers heuristic fallback).
    """
    try:
        client = _OpenAIClient(base_url=API_BASE_URL, api_key=API_KEY or "no-key")
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.2,
            max_tokens=256,
        )
        text = response.choices[0].message.content or ""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception as exc:
        print(f"  [LLM error: {exc}] → heuristic fallback")
        return {}


# ---------------------------------------------------------------------------
# Heuristic agent — deterministic, no API key required
# ---------------------------------------------------------------------------

class HeuristicAgent:
    """
    Rule-based agent that mirrors the risk formula in generate_dataset.py.
    Near-perfect accuracy on easy tasks; good on medium; reasonable on hard.
    """

    REVEAL_PRIORITY = [
        "assess_credit_score",   # hard floor check
        "assess_dti",            # second-most predictive
        "assess_revenue",        # needed for loan-to-revenue ratio
        "assess_business_age",   # hard floor check
        "assess_collateral",     # discount factor
        "assess_cash_flow",      # least weight
    ]

    def choose_action(self, obs: LoanObservation) -> str:
        # Hard floors — decide immediately
        if obs.credit_score is not None and obs.credit_score < 500:
            return "decide_reject"
        if obs.dti is not None and obs.dti > 0.80:
            return "decide_reject"
        if (
            obs.business_age_years is not None and obs.business_age_years < 1.0
            and obs.collateral_value is not None
            and obs.collateral_value < obs.loan_amount * 0.5
        ):
            return "decide_reject"

        # Signal counting once ≥ 3 factors revealed
        n = len(obs.factors_assessed)
        if n >= 3:
            pos = neg = 0
            if obs.credit_score is not None:
                if obs.credit_score >= 650:  pos += 1
                elif obs.credit_score < 550: neg += 1
            if obs.dti is not None:
                if obs.dti <= 0.40:  pos += 1
                elif obs.dti > 0.60: neg += 1
            if obs.loan_to_revenue is not None:
                if obs.loan_to_revenue <= 0.50:  pos += 1
                elif obs.loan_to_revenue > 1.0:  neg += 1
            if obs.business_age_years is not None:
                if obs.business_age_years >= 3:  pos += 1
                elif obs.business_age_years < 1: neg += 1
            if obs.cash_flow_volatility is not None:
                if obs.cash_flow_volatility <= 0.40:  pos += 1
                elif obs.cash_flow_volatility > 0.60: neg += 1
            if obs.collateral_value is not None:
                cov = obs.collateral_value / max(obs.loan_amount, 1)
                if cov >= 0.8:   pos += 1
                elif cov < 0.2:  neg += 1

            total = pos + neg
            if total > 0:
                ratio = pos / total
                if ratio >= 0.75 and n >= 3: return "decide_approve"
                if ratio <= 0.30 and n >= 3: return "decide_reject"
                if n >= 5:                   return "decide_refer"

        # Reveal next hidden factor in priority order
        revealed = set(obs.factors_assessed)
        for act in self.REVEAL_PRIORITY:
            if ACTION_TO_FACTOR[act] not in revealed:
                return act

        return "decide_refer"


# ---------------------------------------------------------------------------
# LLM agent — wraps heuristic with LLM override
# ---------------------------------------------------------------------------

class LLMAgent:
    """Uses the LLM to pick actions; falls back to heuristic on errors."""

    def __init__(self):
        self._heuristic = HeuristicAgent()

    def choose_action(self, obs: LoanObservation) -> tuple[str, str]:
        """Returns (action_type, reasoning_string)."""
        result     = _call_llm(_obs_to_prompt(obs))
        action_type = result.get("action_type", "")
        reasoning   = result.get("reasoning", "")

        if action_type not in VALID_ACTIONS:
            action_type = self._heuristic.choose_action(obs)
            reasoning   = f"[heuristic fallback] {reasoning}"

        return action_type, reasoning


# ---------------------------------------------------------------------------
# Deserialise server JSON → LoanObservation
# ---------------------------------------------------------------------------

def _dict_to_obs(d: dict) -> LoanObservation:
    """
    Reconstruct a LoanObservation from a server JSON response dict.
    LoanObservation is a plain dataclass so use dataclasses.fields().
    """
    valid_fields = {f.name for f in dataclasses.fields(LoanObservation)}
    clean = {k: v for k, v in d.items() if k in valid_fields}
    return LoanObservation(**clean)


# ---------------------------------------------------------------------------
# Local episode runner
# ---------------------------------------------------------------------------

def run_episode_local(
    env: LoanEnvironment,
    task_id: str,
    agent,
    verbose: bool = True,
) -> dict:
    obs = env.reset(task_id)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Episode: {obs.application_id}  [{obs.task_id.upper()}]")
        print(f"  {obs.business_name}  ({obs.sector})  |  £{obs.loan_amount:,.0f}")
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

        obs = env.step(LoanAction(
            action_type=action_type,
            application_id=obs.application_id,
        ))

        if verbose:
            print(f"  → Reward: {obs.reward:+.3f}  |  Cumulative: {obs.cumulative_reward:.3f}")
            if obs.feedback:
                print(f"  → {obs.feedback}")

    st    = env.state
    score = grade(
        action_log=st.action_log,
        ground_truth=st.ground_truth_decision,
        task_id=st.task_id,
    )

    result = {
        "application_id":    obs.application_id,
        "task_id":           obs.task_id,
        "final_decision":    obs.final_decision,
        "ground_truth":      st.ground_truth_decision,
        "correct":           obs.final_decision == st.ground_truth_decision,
        "cumulative_reward": st.cumulative_reward,
        "n_reveals":         sum(1 for e in st.action_log
                                 if e["action_type"].startswith("assess_") and e["valid"]),
        "n_steps":           st.step_count,
        "grade_score":       score,
        "penalties":         st.penalty_total,
    }

    if verbose:
        mark = "✓ CORRECT" if result["correct"] else "✗ WRONG"
        print(f"\n  {'─'*50}")
        print(f"  Decision: {obs.final_decision.upper():10s}  GT: {st.ground_truth_decision.upper()}")
        print(f"  {mark}  |  Grade: {score:.4f}  |  Reveals: {result['n_reveals']}/6")

    return result


# ---------------------------------------------------------------------------
# Remote episode runner
# ---------------------------------------------------------------------------

def run_episode_remote(
    base_url: str,
    task_id: str,
    agent,
    verbose: bool = True,
) -> dict:
    import requests

    def _reset(tid):
        r = requests.post(f"{base_url}/reset", json={"task_id": tid}, timeout=15)
        r.raise_for_status(); return r.json()

    def _step(atype, app_id):
        r = requests.post(f"{base_url}/step",
                          json={"action_type": atype, "application_id": app_id},
                          timeout=15)
        r.raise_for_status(); return r.json()

    def _state():
        r = requests.get(f"{base_url}/state", timeout=10)
        r.raise_for_status(); return r.json()

    def _grade_remote(log, gt, tid_str):
        r = requests.post(f"{base_url}/grade",
                          json={"action_log": log, "ground_truth": gt, "task_id": tid_str},
                          timeout=10)
        r.raise_for_status(); return r.json()

    obs = _dict_to_obs(_reset(task_id))

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Episode (remote): {obs.application_id}  [{obs.task_id.upper()}]")
        print(f"  {obs.business_name}  |  £{obs.loan_amount:,.0f}")
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

        obs = _dict_to_obs(_step(action_type, obs.application_id))
        if verbose:
            print(f"  → Reward: {obs.reward:+.3f}  |  {obs.feedback[:80]}")

    st_dict    = _state()
    grade_resp = _grade_remote(
        st_dict.get("action_log", []),
        st_dict.get("ground_truth_decision", ""),
        st_dict.get("task_id", "easy"),
    )

    result = {
        "application_id":    obs.application_id,
        "task_id":           obs.task_id,
        "final_decision":    obs.final_decision,
        "ground_truth":      st_dict.get("ground_truth_decision"),
        "correct":           grade_resp.get("correct", False),
        "cumulative_reward": st_dict.get("cumulative_reward", 0),
        "n_reveals":         grade_resp.get("n_reveals", 0),
        "n_steps":           st_dict.get("step_count", 0),
        "grade_score":       grade_resp.get("score", 0.0),
        "penalties":         st_dict.get("penalty_total", 0.0),
    }

    if verbose:
        mark = "✓ CORRECT" if result["correct"] else "✗ WRONG"
        print(f"\n  {'─'*50}")
        print(f"  Decision: {obs.final_decision.upper()}  GT: {result['ground_truth'].upper()}")
        print(f"  {mark}  |  Grade: {result['grade_score']:.4f}")

    return result


# ---------------------------------------------------------------------------
# Full evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    agent,
    mode: str = "local",
    tier: Optional[str] = None,
    base_url: str = "http://localhost:7860",
    verbose: bool = False,
) -> dict:
    all_tasks    = _load_tasks()
    tasks_to_run = [t for t in all_tasks if t["task_id"] == tier] if tier else all_tasks

    if not tasks_to_run:
        print(f"No tasks found for tier '{tier}'")
        return {}

    env = LoanEnvironment() if mode == "local" else None

    print(f"\n{'='*60}")
    print(f"  SME Credit Risk — Evaluation")
    print(f"  Mode: {mode.upper()}  Agent: {type(agent).__name__}  Tasks: {len(tasks_to_run)}")
    print(f"{'='*60}\n")

    results = []
    for task in tasks_to_run:
        tid = task["application_id"]
        try:
            if mode == "local":
                r = run_episode_local(env, tid, agent, verbose=verbose)
            else:
                r = run_episode_remote(base_url, tid, agent, verbose=verbose)
            results.append(r)
            mark = "✓" if r["correct"] else "✗"
            print(
                f"  {mark} {tid:12s} [{r['task_id']:6s}]  "
                f"decision={r['final_decision']:7s}  gt={r['ground_truth']:7s}  "
                f"score={r['grade_score']:.3f}  reveals={r['n_reveals']}"
            )
        except Exception as exc:
            print(f"  ! {tid:12s} ERROR: {exc}")

    if not results:
        return {}

    print(f"\n{'─'*60}")
    by_tier: dict[str, list] = {}
    for r in results:
        by_tier.setdefault(r["task_id"], []).append(r)

    all_scores = []
    for t in ["easy", "medium", "hard"]:
        if t not in by_tier: continue
        tr     = by_tier[t]
        n      = len(tr)
        n_ok   = sum(1 for r in tr if r["correct"])
        avg_sc = sum(r["grade_score"] for r in tr) / n
        avg_rv = sum(r["n_reveals"]   for r in tr) / n
        all_scores.extend(r["grade_score"] for r in tr)
        print(f"  {t.upper():6s}  accuracy={n_ok}/{n}  avg_score={avg_sc:.3f}  avg_reveals={avg_rv:.1f}")

    overall = sum(all_scores) / len(all_scores)
    n_ok_total = sum(1 for r in results if r["correct"])
    print(f"\n  OVERALL  accuracy={n_ok_total}/{len(results)}  avg_score={overall:.3f}")
    print(f"{'='*60}\n")

    return {
        "results":       results,
        "by_tier":       by_tier,
        "overall_score": overall,
        "accuracy":      n_ok_total / len(results),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SME Credit Risk RL Agent")
    parser.add_argument("--mode",    default="local",  choices=["local", "remote"])
    parser.add_argument("--url",     default=os.environ.get("SME_ENV_URL", "http://localhost:7860"))
    parser.add_argument("--tier",    default=None, choices=["easy", "medium", "hard"])
    parser.add_argument("--task",    default=None, help="Single task by application_id")
    parser.add_argument("--eval",    action="store_true", help="Evaluate full tier (or all tiers)")
    parser.add_argument("--all",     action="store_true", help="Evaluate all tiers")
    parser.add_argument("--llm",     action="store_true", help="Use LLM agent")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.llm:
        if not API_KEY:
            print("WARNING: HF_TOKEN / API_KEY not set. Set with:")
            print("  export HF_TOKEN=hf_your_token_here")
            print("Continuing — some providers work without a key.")
        agent = LLMAgent()
        print(f"Agent: LLM  model={MODEL_NAME}  base={API_BASE_URL}")
    else:
        agent = HeuristicAgent()
        print("Agent: Heuristic (deterministic, no API key needed)")

    if args.eval or args.all:
        eval_tier = None if args.all else (args.tier or "easy")
        run_evaluation(agent, mode=args.mode, tier=eval_tier,
                       base_url=args.url, verbose=args.verbose)

    elif args.task:
        if args.mode == "local":
            run_episode_local(LoanEnvironment(), args.task, agent, verbose=True)
        else:
            run_episode_remote(args.url, args.task, agent, verbose=True)

    else:
        all_tasks  = _load_tasks()
        use_tier   = args.tier or "easy"
        tier_tasks = [t for t in all_tasks if t["task_id"] == use_tier]
        if not tier_tasks:
            print(f"No tasks for tier '{use_tier}'")
            sys.exit(1)
        tid = tier_tasks[0]["application_id"]
        if args.mode == "local":
            run_episode_local(LoanEnvironment(), tid, agent, verbose=True)
        else:
            run_episode_remote(args.url, tid, agent, verbose=True)


if __name__ == "__main__":
    main()