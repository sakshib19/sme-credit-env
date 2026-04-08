"""
env/graders.py — Deterministic Episode Graders
===============================================
Three grading functions — one per difficulty tier — that evaluate a
completed episode's action_log and return a score in [0.0, 1.0].

Grading philosophy
------------------
Each grader weighs two axes:

  1. Correctness  (did the agent make the right call?)
       - Correct approve/reject  → full credit
       - Refer                   → partial credit  (reasonable caution)
       - Wrong approve/reject    → no credit (or penalty adjusted down)

  2. Efficiency   (how few reveals did the agent need?)
       - Ideal reveals: 0  (decided without any reveals)  → score 1.0
       - Max reveals: all 6 factors revealed               → score 0.0 on efficiency
       - Efficiency is penalised progressively on harder tiers

Score formula (per tier):
  score = correctness_weight × correctness_score
        + efficiency_weight  × efficiency_score

Difficulty-specific weights
---------------------------
  Easy   → correctness 0.80, efficiency 0.20
             (getting the answer right is what matters most)
  Medium → correctness 0.65, efficiency 0.35
             (efficiency starts to matter)
  Hard   → correctness 0.50, efficiency 0.50
             (both dimensions equally important)

All functions are fully deterministic — no randomness, no LLM calls.
"""

from __future__ import annotations

from typing import Optional

# Number of revealable factors (from models.py)
_N_FACTORS = 6

# Correctness scores
_CORRECT_SCORE  = 1.0
_REFER_SCORE    = 0.40   # Refer = plausible caution, not a clear win
_WRONG_SCORE    = 0.0

# How many assess actions happened before the decide?
def _count_reveals(action_log: list[dict]) -> int:
    return sum(
        1 for entry in action_log
        if entry.get("action_type", "").startswith("assess_")
        and entry.get("valid", True)
    )

def _final_decision_entry(action_log: list[dict]) -> Optional[dict]:
    """Return the last decide_* entry, or None if no decision was made."""
    for entry in reversed(action_log):
        if entry.get("action_type", "").startswith("decide_"):
            return entry
    return None

def _extract_decision(action_log: list[dict]) -> Optional[str]:
    entry = _final_decision_entry(action_log)
    if entry is None:
        return None
    mapping = {
        "decide_approve": "approve",
        "decide_reject":  "reject",
        "decide_refer":   "refer",
    }
    return mapping.get(entry["action_type"])

def _correctness_score(agent_decision: Optional[str], ground_truth: str) -> float:
    if agent_decision is None:
        return 0.0
    if agent_decision == ground_truth:
        return _CORRECT_SCORE
    if agent_decision == "refer":
        return _REFER_SCORE
    return _WRONG_SCORE

def _efficiency_score(n_reveals: int) -> float:
    """
    Linear decay from 1.0 (0 reveals) to 0.0 (all 6 reveals).
    Each extra reveal costs 1/6 of the efficiency score.
    """
    if n_reveals <= 0:
        return 1.0
    if n_reveals >= _N_FACTORS:
        return 0.0
    return round(1.0 - n_reveals / _N_FACTORS, 4)


# ---------------------------------------------------------------------------
# Public grading functions
# ---------------------------------------------------------------------------

def grade_easy(action_log: list[dict], ground_truth: str) -> float:
    """
    Grade an 'easy' episode.

    Easy cases have clear-cut financials — the agent should be able to
    decide correctly with minimal investigation.

    Weights: correctness=0.80, efficiency=0.20

    Parameters
    ----------
    action_log   : list of action-log dicts from LoanState.action_log
    ground_truth : the correct decision ("approve" | "reject" | "refer")

    Returns
    -------
    float in [0.0, 1.0]
    """
    w_correct, w_eff = 0.80, 0.20

    agent_decision = _extract_decision(action_log)
    n_reveals      = _count_reveals(action_log)

    c_score = _correctness_score(agent_decision, ground_truth)
    e_score = _efficiency_score(n_reveals)

    return round(w_correct * c_score + w_eff * e_score, 4)


def grade_medium(action_log: list[dict], ground_truth: str) -> float:
    """
    Grade a 'medium' episode.

    Medium cases have mixed signals — the agent needs to reveal a few
    key factors but should not exhaust all of them.

    Weights: correctness=0.65, efficiency=0.35

    Parameters
    ----------
    action_log   : list of action-log dicts from LoanState.action_log
    ground_truth : the correct decision ("approve" | "reject" | "refer")

    Returns
    -------
    float in [0.0, 1.0]
    """
    w_correct, w_eff = 0.65, 0.35

    agent_decision = _extract_decision(action_log)
    n_reveals      = _count_reveals(action_log)

    c_score = _correctness_score(agent_decision, ground_truth)
    e_score = _efficiency_score(n_reveals)

    return round(w_correct * c_score + w_eff * e_score, 4)


def grade_hard(action_log: list[dict], ground_truth: str) -> float:
    """
    Grade a 'hard' episode.

    Hard cases are borderline applications where the agent must read
    signals carefully and decide efficiently — both matter equally.

    Weights: correctness=0.50, efficiency=0.50

    Extra penalty logic for hard tier:
      - If the agent made a wrong decision AND revealed ALL factors,
        a −0.10 surcharge is applied (exhausted all information, still
        got it wrong → especially costly).

    Parameters
    ----------
    action_log   : list of action-log dicts from LoanState.action_log
    ground_truth : the correct decision ("approve" | "reject" | "refer")

    Returns
    -------
    float in [0.0, 1.0]
    """
    w_correct, w_eff = 0.50, 0.50

    agent_decision = _extract_decision(action_log)
    n_reveals      = _count_reveals(action_log)

    c_score = _correctness_score(agent_decision, ground_truth)
    e_score = _efficiency_score(n_reveals)

    base = w_correct * c_score + w_eff * e_score

    # Surcharge: wrong + exhausted all info
    if (
        agent_decision is not None
        and agent_decision != ground_truth
        and agent_decision != "refer"
        and n_reveals >= _N_FACTORS
    ):
        base = max(0.0, base - 0.10)

    return round(base, 4)


# ---------------------------------------------------------------------------
# Convenience dispatcher
# ---------------------------------------------------------------------------

def grade(action_log: list[dict], ground_truth: str, task_id: str) -> float:
    """
    Route to the correct grader based on task_id.

    Parameters
    ----------
    action_log   : list of action-log dicts
    ground_truth : correct decision string
    task_id      : "easy" | "medium" | "hard"

    Returns
    -------
    float in [0.0, 1.0]
    """
    if task_id == "easy":
        return grade_easy(action_log, ground_truth)
    if task_id == "medium":
        return grade_medium(action_log, ground_truth)
    if task_id == "hard":
        return grade_hard(action_log, ground_truth)
    raise ValueError(f"Unknown task_id '{task_id}'. Expected easy | medium | hard.")


# ---------------------------------------------------------------------------
# Smoke-test — run with: python env/graders.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  graders.py — Smoke Test")
    print("=" * 60)

    def _make_log(decision: str, n_reveals: int) -> list[dict]:
        """Helper to build a minimal action log."""
        log = []
        for i in range(n_reveals):
            log.append({
                "step": i + 1,
                "action_type": f"assess_credit_score",
                "valid": True,
                "reward": 0.10,
            })
        log.append({
            "step": n_reveals + 1,
            "action_type": f"decide_{decision}",
            "valid": True,
            "reward": 1.0,
        })
        return log

    scenarios = [
        # (label, grader, decision, n_reveals, ground_truth)
        ("easy  / correct / 0 reveals",  "easy",   "approve", 0, "approve"),
        ("easy  / correct / 3 reveals",  "easy",   "approve", 3, "approve"),
        ("easy  / wrong   / 1 reveal",   "easy",   "reject",  1, "approve"),
        ("easy  / refer   / 2 reveals",  "easy",   "refer",   2, "approve"),
        ("medium/ correct / 2 reveals",  "medium", "reject",  2, "reject"),
        ("medium/ wrong   / 5 reveals",  "medium", "approve", 5, "reject"),
        ("hard  / correct / 0 reveals",  "hard",   "reject",  0, "reject"),
        ("hard  / wrong   / 6 reveals",  "hard",   "approve", 6, "reject"),
        ("hard  / refer   / 3 reveals",  "hard",   "refer",   3, "reject"),
        ("no decision at all",           "easy",   None,      3, "approve"),
    ]

    print(f"\n{'Scenario':<42}  {'Score':>6}")
    print("-" * 52)
    for label, tier, decision, n_reveals, gt in scenarios:
        if decision is None:
            # simulate no decide action
            log = _make_log("approve", n_reveals)
            log = [e for e in log if not e["action_type"].startswith("decide_")]
        else:
            log = _make_log(decision, n_reveals)
        score = grade(log, gt, tier)
        print(f"  {label:<42}  {score:>6.4f}")

    print("\n  Graders functional.\n")
