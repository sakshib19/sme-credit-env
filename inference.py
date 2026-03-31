"""
inference.py — SME Credit Risk RL Agent
=========================================
A rule-driven + LLM-hybrid agent that plays full episodes.

Modes:
  local   → talks directly to LoanEnvironment (no server needed)
  remote  → talks to the FastAPI server via HTTP

MANDATORY environment variables (hackathon requirement):
  API_BASE_URL   — LLM API endpoint, e.g. https://router.huggingface.co/v1
  MODEL_NAME     — model identifier,  e.g. Qwen/Qwen2.5-72B-Instruct
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
# Load .env file FIRST — before reading any env vars
# python-dotenv loads .env from the current working directory (or any parent).
# This means: cd into your project root, then run python inference.py.
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    # override=False keeps real env vars (export) higher priority than .env
    _env_path = Path(__file__).resolve().parent / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path, override=False)
        print(f"[info] Loaded .env from {_env_path}")
    else:
        load_dotenv(override=False)   # search parent dirs
except ImportError:
    pass   # python-dotenv not installed — rely on exported env vars

# ---------------------------------------------------------------------------
# Mandatory env vars (hackathon requirement)
# Read AFTER load_dotenv() so .env values are visible to os.environ.get()
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1").strip().strip('"').strip("'")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct").strip().strip('"').strip("'")
API_KEY      = (os.environ.get("HF_TOKEN") or os.environ.get("API_KEY") or "").strip().strip('"').strip("'")

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
Evaluate loan applications step-by-step. Reveal financial factors one at
a time, then make a final decision.

## AVAILABLE ACTIONS
Assess (reveal one hidden factor):
  - assess_credit_score   → credit_score (300–850, higher = better)
  - assess_dti            → dti (debt-to-income 0.0–1.0, lower = better)
  - assess_revenue        → annual_revenue in GBP (also reveals loan_to_revenue)
  - assess_business_age   → business_age_years (higher = better)
  - assess_collateral     → collateral_value in GBP (higher = better)
  - assess_cash_flow      → cash_flow_volatility (0=stable, 1=volatile, lower = better)

Decide (ends the episode — pick exactly one):
  - decide_approve  → approve the loan
  - decide_reject   → reject the loan
  - decide_refer    → refer to senior underwriter

## RISK SCORING (how the system actually scores applications)
Each factor contributes a weighted penalty (0.0 = good, 1.0 = bad):
  credit_score    weight 0.30  — (850 - score) / 550
  dti             weight 0.25  — 0 if ≤0.30, scales to 1.0 at dti=1.0
  loan_to_revenue weight 0.20  — 0 if ≤0.25, scales to 1.0 if >2.0
  business_age    weight 0.15  — 0 if ≥5yrs, 0.9 if <1yr
  cash_flow_vol   weight 0.10  — direct (0.4 = 0.4 penalty)

  risk_score = weighted_sum × (1 − 0.20 × collateral_coverage)
    APPROVE  if risk_score < 0.35
    REJECT   if risk_score > 0.65
    REFER    otherwise (borderline)

## HARD FLOORS (always reject regardless of other factors)
     credit_score STRICTLY BELOW 500 (i.e. credit_score ≤ 499)
     dti > 0.80

## KEY INSIGHT — ONE WEAK FACTOR ≠ REJECT
  A credit score of 520 is weak but is NOT below 500.
  If the other 5 factors are strong (low dti, low ltr, old business,
  low volatility, high collateral), the application can still APPROVE.
  NEVER reject on credit alone unless it is strictly below 500.
  NEVER reject on a single negative signal — use the weighted picture.

## DECISION GUIDE
  APPROVE:  Most factors positive, risk_score likely < 0.35
  REJECT:   Most factors negative, risk_score likely > 0.65, OR hard floor hit
  REFER:    Mixed signals, risk_score likely 0.35–0.65

## EFFICIENCY RULE
  Reveal enough to be confident — usually 3–4 factors.
  Each unrevealed factor earns a +0.05 efficiency bonus.
  Don't over-reveal if the picture is already clear.

## RESPONSE FORMAT - CRITICAL
Respond with ONLY a valid JSON object. No prose. No markdown. No code fences.
Example: {"reasoning": "Credit score 780 is well above threshold, approve.", "action_type": "decide_approve"}

The action_type MUST be exactly one of:
assess_revenue, assess_credit_score, assess_dti, assess_collateral,
assess_business_age, assess_cash_flow, decide_approve, decide_reject, decide_refer
""".strip()


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _obs_to_prompt(obs: LoanObservation) -> str:
    lines = [
        f"APPLICATION: {obs.application_id}  |  {obs.business_name}  ({obs.sector})",
        f"Loan requested: ₹{obs.loan_amount:,.0f}",
        f"Step: {obs.step_count}/{obs.max_steps}  |  Cumulative reward: {obs.cumulative_reward:.3f}",
        "",
        "REVEALED FACTORS:",
    ]
    factor_display = {
        "annual_revenue":       ("Annual Revenue",       lambda v: f"₹{v:,.0f}"),
        "credit_score":         ("Credit Score",         lambda v: str(v)),
        "dti":                  ("DTI Ratio",            lambda v: f"{v:.1%}"),
        "collateral_value":     ("Collateral Value",     lambda v: f"₹{v:,.0f}"),
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
        "",
        'Respond with JSON only: {"reasoning": "...", "action_type": "..."}',
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

    Risk formula (weighted):
      credit_score    0.30  penalty = (850 - score) / 550
      dti             0.25  0 if <=0.30, scales up to 1.0
      loan_to_revenue 0.20  0 if <=0.25, scales up
      business_age    0.15  0 if >=5yr, 0.9 if <1yr
      cash_flow_vol   0.10  direct value
      collateral discount: × (1 - 0.20 × coverage)

    APPROVE if risk < 0.35, REJECT if risk > 0.65, else REFER.
    Hard floors: credit < 500 OR dti > 0.80 → always reject.
    """

    REVEAL_PRIORITY = [
        "assess_credit_score",   # highest weight (0.30), hard floor check
        "assess_dti",            # second weight (0.25), hard floor check
        "assess_revenue",        # reveals loan_to_revenue (weight 0.20)
        "assess_business_age",   # weight 0.15
        "assess_collateral",     # discount multiplier
        "assess_cash_flow",      # lowest weight (0.10)
    ]

    # ── Penalty functions matching generate_dataset.py exactly ──────────
    @staticmethod
    def _credit_penalty(s: float) -> float:
        return max(0.0, min(1.0, (850 - s) / 550))

    @staticmethod
    def _dti_penalty(d: float) -> float:
        if d <= 0.30: return 0.0
        if d <= 0.45: return 0.2
        if d <= 0.60: return 0.5
        return min(1.0, 0.85 + (d - 0.60) * 0.75)

    @staticmethod
    def _ltr_penalty(l: float) -> float:
        if l <= 0.25: return 0.0
        if l <= 0.50: return 0.15
        if l <= 1.00: return 0.40
        if l <= 2.00: return 0.70
        return 1.0

    @staticmethod
    def _age_penalty(a: float) -> float:
        if a >= 5.0: return 0.0
        if a >= 3.0: return 0.15
        if a >= 2.0: return 0.35
        if a >= 1.0: return 0.60
        return 0.90

    def _estimate_risk(self, obs: LoanObservation) -> float | None:
        """
        Compute weighted risk score from revealed factors.
        Returns None if fewer than 3 factors revealed (not enough info).
        Missing factors are estimated at 0.5 (uncertain/neutral).
        """
        n = len(obs.factors_assessed)
        if n < 3:
            return None

        UNKNOWN = 0.5   # neutral estimate for unrevealed factors

        cp  = self._credit_penalty(obs.credit_score) if obs.credit_score is not None else UNKNOWN
        dp  = self._dti_penalty(obs.dti)             if obs.dti          is not None else UNKNOWN
        ltr = obs.loan_to_revenue if obs.loan_to_revenue is not None else (
              obs.loan_amount / max(obs.annual_revenue, 1) if obs.annual_revenue else UNKNOWN)
        lp  = self._ltr_penalty(ltr) if not isinstance(ltr, float) or ltr != UNKNOWN else UNKNOWN
        ap  = self._age_penalty(obs.business_age_years) if obs.business_age_years is not None else UNKNOWN
        vp  = min(1.0, obs.cash_flow_volatility * 1.2)  if obs.cash_flow_volatility is not None else UNKNOWN

        # Collateral coverage discount
        if obs.collateral_value is not None:
            cov      = obs.collateral_value / max(obs.loan_amount, 1)
            discount = 1.0 - 0.20 * min(cov, 2.0)   # cap coverage at 2.0
        else:
            discount = 1.0   # no collateral info → no discount

        raw = (cp * 0.30 + dp * 0.25 + lp * 0.20 + ap * 0.15 + vp * 0.10)
        return round(raw * discount, 4)

    def choose_action(self, obs: LoanObservation) -> str:
        # ── Hard floors — immediate reject ────────────────────────────
        if obs.credit_score is not None and obs.credit_score < 500:
            return "decide_reject"
        if obs.dti is not None and obs.dti > 0.80:
            return "decide_reject"

        # ── Estimate risk once we have enough data ────────────────────
        n = len(obs.factors_assessed)
        risk = self._estimate_risk(obs)

        if risk is not None:
            if risk < 0.35 and n >= 3:
                return "decide_approve"
            if risk > 0.65 and n >= 3:
                return "decide_reject"
            # Borderline: refer only after seeing enough factors
            if risk >= 0.35 and n >= 5:
                return "decide_refer"

        # ── Reveal next hidden factor in priority order ───────────────
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
        print(f"  {obs.business_name}  ({obs.sector})  |  ₹{obs.loan_amount:,.0f}")
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
        print(f"  {obs.business_name}  |  ₹{obs.loan_amount:,.0f}")
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
            print("ERROR: HF_TOKEN not found in environment or .env file.")
            print()
            print("  Option 1 — create a .env file in your project root:")
            print("    API_BASE_URL=https://router.huggingface.co/v1")
            print("    MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2")
            print("    HF_TOKEN=hf_your_token_here")
            print()
            print("  Option 2 — set env vars directly:")
            print("    Windows:  set HF_TOKEN=hf_your_token_here")
            print("    Mac/Linux: export HF_TOKEN=hf_your_token_here")
            print()
            print("  Continuing anyway — using heuristic fallback for all steps.")
        else:
            print(f"  HF_TOKEN loaded ✓ (first 8 chars: {API_KEY[:8]}...)")
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