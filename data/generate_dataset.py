"""
generate_dataset.py — SME Credit Risk Synthetic Dataset Generator
==================================================================
Run this script ONCE to generate all task data before building the environment.

    python data/generate_dataset.py

Output: data/tasks.json

What this produces
------------------
  Task 1 — Easy   : 10 applications, clear-cut approve/reject (single dominant factor)
  Task 2 — Medium : 20 applications, multi-factor risk scoring required
  Task 3 — Hard   : 20 applications, conflicting signals, edge-case adjudication

Every application has a pre-calculated ground truth:
  - decision         : "approve" | "reject" | "refer"
  - risk_score       : float 0.0 (low risk) → 1.0 (high risk)
  - factor_direction : per-factor signal ("positive"/"negative"/"neutral")
  - explanation      : human-readable rationale (for README / debugging)

No LLM involved. All ground truth is produced by the deterministic
risk formula defined in this file. Same script = same dataset every time
(fixed random seed).

Dependencies: pip install numpy faker
"""

import json
import random
import math
from pathlib import Path
from typing import TypedDict, Optional

# ---------------------------------------------------------------------------
# Reproducibility — fixed seed everywhere
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)

# Minimal faker-free name generation so there's zero extra dependency
BUSINESS_TYPES = [
    "Fabricators", "Solutions", "Logistics", "Consulting", "Manufacturing",
    "Services", "Enterprises", "Systems", "Trading", "Partners",
    "Industries", "Distributors", "Technologies", "Suppliers", "Group",
]
BUSINESS_PREFIXES = [
    "Redwood", "Oakdale", "Silverline", "Crestview", "Northgate",
    "Ironbridge", "Clearwater", "Summit", "Meridian", "Lakeside",
    "Pioneer", "Horizon", "Cornerstone", "Pinnacle", "Waverly",
    "Bridgeport", "Ashford", "Thornton", "Westfield", "Ellsworth",
    "Harborview", "Cedarwood", "Stonegate", "Maplewood", "Fairview",
    "Glenwood", "Brookside", "Riverside", "Highland", "Millbrook",
    "Cliffside", "Sunridge", "Greenfield", "Valleyview", "Eastgate",
    "Windmill", "Crossroads", "Copperfield", "Foxwood", "Ravenwood",
    "Bluewater", "Goldcrest", "Silverwood", "Ironwood", "Whitewater",
    "Blackstone", "Greymount", "Bluestone", "Redstone", "Whitfield",
]
BUSINESS_SECTORS = [
    "manufacturing", "retail", "logistics", "consulting", "food_services",
    "construction", "technology", "healthcare", "distribution", "hospitality",
]

_name_counter = 0

def _business_name() -> str:
    global _name_counter
    prefix = BUSINESS_PREFIXES[_name_counter % len(BUSINESS_PREFIXES)]
    btype  = BUSINESS_TYPES[_name_counter % len(BUSINESS_TYPES)]
    _name_counter += 1
    return f"{prefix} {btype} Ltd"


# ---------------------------------------------------------------------------
# Application field types
# ---------------------------------------------------------------------------

class LoanApplication(TypedDict):
    application_id:       str
    business_name:        str
    sector:               str
    # Financial factors (revealed one at a time by the agent)
    annual_revenue:       float      # GBP
    credit_score:         int        # 300–850
    dti:                  float      # debt-to-income ratio 0.0–1.0
    collateral_value:     float      # GBP  (0 = unsecured)
    business_age_years:   float      # years in operation
    cash_flow_volatility: float      # 0.0 = stable, 1.0 = highly volatile
    loan_amount:          float      # GBP requested
    # Ground truth (hidden from agent, used only by graders)
    risk_score:           float      # 0.0 = very low risk, 1.0 = very high risk
    decision:             str        # "approve" | "reject" | "refer"
    factor_directions:    dict       # per-factor: "positive"|"negative"|"neutral"
    explanation:          str        # human-readable rationale
    loan_to_revenue:      float      # derived: loan_amount / annual_revenue
    collateral_coverage:  float      # derived: collateral_value / loan_amount


# ---------------------------------------------------------------------------
# Ground truth risk formula
# ---------------------------------------------------------------------------
# Risk score is a weighted sum of normalised factor penalties.
# Lower score = lower risk = more likely to approve.
#
# Weights (must sum to 1.0):
#   credit_score          0.30  (most important)
#   dti                   0.25
#   loan_to_revenue       0.20
#   business_age          0.15
#   cash_flow_volatility  0.10
#
# Each factor contributes a penalty [0.0, 1.0] that is then weighted.
# Final risk_score = sum of weighted penalties, clamped to [0.0, 1.0].

def _credit_penalty(score: int) -> float:
    """High credit score → low penalty."""
    # 850 → 0.0,  300 → 1.0
    return max(0.0, min(1.0, (850 - score) / 550))

def _dti_penalty(dti: float) -> float:
    """High DTI → high penalty. Above 0.6 = danger zone."""
    if dti <= 0.30: return 0.0
    if dti <= 0.45: return 0.20
    if dti <= 0.60: return 0.50
    return 0.85 + (dti - 0.60) * 0.75

def _ltr_penalty(ltr: float) -> float:
    """High loan-to-revenue ratio → high penalty."""
    if ltr <= 0.25: return 0.0
    if ltr <= 0.50: return 0.15
    if ltr <= 1.00: return 0.40
    if ltr <= 2.00: return 0.70
    return 1.0

def _age_penalty(age: float) -> float:
    """Young business → higher risk."""
    if age >= 5:  return 0.0
    if age >= 3:  return 0.15
    if age >= 2:  return 0.35
    if age >= 1:  return 0.60
    return 0.90

def _volatility_penalty(vol: float) -> float:
    """High cash-flow volatility → higher risk."""
    return min(1.0, vol * 1.2)

def compute_risk_score(app: dict) -> float:
    """Deterministic risk score. Same inputs → same output always."""
    ltr = app["loan_amount"] / max(app["annual_revenue"], 1)

    penalty = (
        _credit_penalty(app["credit_score"])       * 0.30 +
        _dti_penalty(app["dti"])                   * 0.25 +
        _ltr_penalty(ltr)                          * 0.20 +
        _age_penalty(app["business_age_years"])    * 0.15 +
        _volatility_penalty(app["cash_flow_volatility"]) * 0.10
    )
    # Collateral coverage reduces risk: full coverage cuts risk by up to 20%
    cov = min(1.0, app["collateral_value"] / max(app["loan_amount"], 1))
    penalty = penalty * (1.0 - 0.20 * cov)

    return round(min(1.0, max(0.0, penalty)), 4)


# Hard rules that override the risk score
HARD_FLOOR_CREDIT  = 500   # below this → always reject
APPROVE_THRESHOLD  = 0.35  # risk_score below this → approve
REJECT_THRESHOLD   = 0.65  # risk_score above this → reject
                           # between → refer

def compute_decision(app: dict, risk_score: float) -> str:
    """Deterministic decision. Hard rules first, then risk thresholds."""
    # Hard rejection floor
    if app["credit_score"] < HARD_FLOOR_CREDIT:
        return "reject"
    # Very high debt load
    if app["dti"] > 0.80:
        return "reject"
    # Brand new business with no collateral and large loan
    if app["business_age_years"] < 1.0 and app["collateral_value"] < app["loan_amount"] * 0.5:
        return "reject"
    # Risk threshold routing
    if risk_score < APPROVE_THRESHOLD:
        return "approve"
    if risk_score > REJECT_THRESHOLD:
        return "reject"
    return "refer"


def compute_factor_directions(app: dict) -> dict:
    """
    Per-factor signal direction — used by graders to score agent assessments.
    "positive" = this factor supports approval
    "negative" = this factor supports rejection
    "neutral"  = borderline / no strong signal
    """
    ltr = app["loan_amount"] / max(app["annual_revenue"], 1)
    cov = app["collateral_value"] / max(app["loan_amount"], 1)

    def direction(penalty: float) -> str:
        if penalty < 0.25: return "positive"
        if penalty > 0.55: return "negative"
        return "neutral"

    return {
        "credit_score":         direction(_credit_penalty(app["credit_score"])),
        "dti":                  direction(_dti_penalty(app["dti"])),
        "loan_to_revenue":      direction(_ltr_penalty(ltr)),
        "business_age":         direction(_age_penalty(app["business_age_years"])),
        "cash_flow_volatility": direction(_volatility_penalty(app["cash_flow_volatility"])),
        "collateral":           "positive" if cov >= 0.8 else ("neutral" if cov >= 0.4 else "negative"),
    }


def build_explanation(app: dict, risk_score: float, decision: str) -> str:
    """Human-readable explanation for README / debugging."""
    ltr = app["loan_amount"] / max(app["annual_revenue"], 1)
    cov = app["collateral_value"] / max(app["loan_amount"], 1)
    parts = [
        f"Credit {app['credit_score']} ({'good' if app['credit_score'] >= 650 else 'poor' if app['credit_score'] < 550 else 'fair'})",
        f"DTI {app['dti']:.0%} ({'high' if app['dti'] > 0.5 else 'acceptable'})",
        f"LTR {ltr:.1f}x ({'elevated' if ltr > 1.0 else 'reasonable'})",
        f"Age {app['business_age_years']:.1f}yr ({'young' if app['business_age_years'] < 2 else 'established'})",
        f"Volatility {app['cash_flow_volatility']:.0%} ({'high' if app['cash_flow_volatility'] > 0.5 else 'low'})",
        f"Collateral coverage {cov:.0%}",
    ]
    return f"Risk={risk_score:.3f} → {decision.upper()}. " + ", ".join(parts) + "."


# ---------------------------------------------------------------------------
# Application factories
# ---------------------------------------------------------------------------

def _rand(lo, hi, dp=2):
    return round(random.uniform(lo, hi), dp)

def _make_application(
    app_id: str,
    credit_score: int,
    annual_revenue: float,
    dti: float,
    collateral_value: float,
    business_age_years: float,
    cash_flow_volatility: float,
    loan_amount: float,
    sector: Optional[str] = None,
) -> LoanApplication:
    """Assemble one application and compute all ground truth fields."""
    sector = sector or random.choice(BUSINESS_SECTORS)
    ltr    = loan_amount / max(annual_revenue, 1)
    cov    = collateral_value / max(loan_amount, 1)

    app: dict = {
        "application_id":       app_id,
        "business_name":        _business_name(),
        "sector":               sector,
        "annual_revenue":       annual_revenue,
        "credit_score":         credit_score,
        "dti":                  round(dti, 3),
        "collateral_value":     collateral_value,
        "business_age_years":   round(business_age_years, 1),
        "cash_flow_volatility": round(cash_flow_volatility, 3),
        "loan_amount":          loan_amount,
        "loan_to_revenue":      round(ltr, 3),
        "collateral_coverage":  round(cov, 3),
    }

    risk_score = compute_risk_score(app)
    decision   = compute_decision(app, risk_score)
    directions = compute_factor_directions(app)
    explanation = build_explanation(app, risk_score, decision)

    app.update({
        "risk_score":        risk_score,
        "decision":          decision,
        "factor_directions": directions,
        "explanation":       explanation,
    })
    return app  # type: ignore


# ---------------------------------------------------------------------------
# TASK 1 — EASY (10 applications)
# One dominant factor clearly drives the decision.
# Agent should be able to decide after 1–2 assessments.
# Mix: 4 approve, 4 reject, 2 borderline→refer
# ---------------------------------------------------------------------------

def generate_easy_task() -> list:
    apps = []

    # --- Clear approvals (low risk, high credit, healthy financials) ---
    apps.append(_make_application(
        "easy_01", credit_score=780, annual_revenue=620_000, dti=0.28,
        collateral_value=200_000, business_age_years=7.0,
        cash_flow_volatility=0.12, loan_amount=80_000, sector="manufacturing",
    ))
    apps.append(_make_application(
        "easy_02", credit_score=740, annual_revenue=480_000, dti=0.31,
        collateral_value=150_000, business_age_years=5.5,
        cash_flow_volatility=0.18, loan_amount=60_000, sector="retail",
    ))
    apps.append(_make_application(
        "easy_03", credit_score=800, annual_revenue=950_000, dti=0.22,
        collateral_value=400_000, business_age_years=9.0,
        cash_flow_volatility=0.08, loan_amount=120_000, sector="technology",
    ))
    apps.append(_make_application(
        "easy_04", credit_score=720, annual_revenue=350_000, dti=0.29,
        collateral_value=100_000, business_age_years=6.0,
        cash_flow_volatility=0.15, loan_amount=50_000, sector="consulting",
    ))

    # --- Clear rejections (single dominant red flag) ---
    # Very poor credit score (below hard floor)
    apps.append(_make_application(
        "easy_05", credit_score=420, annual_revenue=500_000, dti=0.30,
        collateral_value=200_000, business_age_years=4.0,
        cash_flow_volatility=0.20, loan_amount=75_000, sector="logistics",
    ))
    # Extreme DTI
    apps.append(_make_application(
        "easy_06", credit_score=620, annual_revenue=300_000, dti=0.85,
        collateral_value=0, business_age_years=3.0,
        cash_flow_volatility=0.30, loan_amount=60_000, sector="food_services",
    ))
    # Brand new business, no collateral, large loan
    apps.append(_make_application(
        "easy_07", credit_score=580, annual_revenue=180_000, dti=0.45,
        collateral_value=0, business_age_years=0.5,
        cash_flow_volatility=0.50, loan_amount=150_000, sector="retail",
    ))
    # Very high loan vs revenue
    apps.append(_make_application(
        "easy_08", credit_score=560, annual_revenue=120_000, dti=0.55,
        collateral_value=20_000, business_age_years=2.0,
        cash_flow_volatility=0.60, loan_amount=300_000, sector="construction",
    ))

    # --- Borderline → refer ---
    apps.append(_make_application(
        "easy_09", credit_score=610, annual_revenue=420_000, dti=0.42,
        collateral_value=80_000, business_age_years=3.5,
        cash_flow_volatility=0.38, loan_amount=90_000, sector="distribution",
    ))
    apps.append(_make_application(
        "easy_10", credit_score=635, annual_revenue=310_000, dti=0.48,
        collateral_value=50_000, business_age_years=2.5,
        cash_flow_volatility=0.42, loan_amount=70_000, sector="hospitality",
    ))

    return apps


# ---------------------------------------------------------------------------
# TASK 2 — MEDIUM (20 applications)
# Multiple factors must be weighed. No single dominant signal.
# Agent needs to assess at least 3 factors before deciding well.
# Mix: 7 approve, 7 reject, 6 refer
# ---------------------------------------------------------------------------

def generate_medium_task() -> list:
    apps = []

    # --- Approvals (good overall profile, some weak spots) ---
    apps.append(_make_application(
        "med_01", credit_score=710, annual_revenue=540_000, dti=0.35,
        collateral_value=180_000, business_age_years=4.5,
        cash_flow_volatility=0.22, loan_amount=100_000, sector="manufacturing",
    ))
    apps.append(_make_application(
        "med_02", credit_score=755, annual_revenue=800_000, dti=0.28,
        collateral_value=350_000, business_age_years=8.0,
        cash_flow_volatility=0.14, loan_amount=150_000, sector="technology",
    ))
    apps.append(_make_application(
        "med_03", credit_score=695, annual_revenue=460_000, dti=0.32,
        collateral_value=120_000, business_age_years=5.0,
        cash_flow_volatility=0.25, loan_amount=85_000, sector="consulting",
    ))
    apps.append(_make_application(
        "med_04", credit_score=730, annual_revenue=620_000, dti=0.30,
        collateral_value=200_000, business_age_years=6.5,
        cash_flow_volatility=0.19, loan_amount=110_000, sector="logistics",
    ))
    apps.append(_make_application(
        "med_05", credit_score=680, annual_revenue=390_000, dti=0.33,
        collateral_value=90_000, business_age_years=4.0,
        cash_flow_volatility=0.28, loan_amount=65_000, sector="retail",
    ))
    apps.append(_make_application(
        "med_06", credit_score=760, annual_revenue=750_000, dti=0.26,
        collateral_value=300_000, business_age_years=7.5,
        cash_flow_volatility=0.11, loan_amount=130_000, sector="healthcare",
    ))
    apps.append(_make_application(
        "med_07", credit_score=705, annual_revenue=500_000, dti=0.34,
        collateral_value=160_000, business_age_years=5.5,
        cash_flow_volatility=0.23, loan_amount=95_000, sector="distribution",
    ))

    # --- Rejections (multiple compounding negatives) ---
    apps.append(_make_application(
        "med_08", credit_score=510, annual_revenue=250_000, dti=0.58,
        collateral_value=30_000, business_age_years=1.5,
        cash_flow_volatility=0.65, loan_amount=180_000, sector="food_services",
    ))
    apps.append(_make_application(
        "med_09", credit_score=480, annual_revenue=320_000, dti=0.52,
        collateral_value=50_000, business_age_years=2.0,
        cash_flow_volatility=0.55, loan_amount=200_000, sector="construction",
    ))
    apps.append(_make_application(
        "med_10", credit_score=540, annual_revenue=180_000, dti=0.70,
        collateral_value=20_000, business_age_years=1.0,
        cash_flow_volatility=0.72, loan_amount=250_000, sector="retail",
    ))
    apps.append(_make_application(
        "med_11", credit_score=460, annual_revenue=140_000, dti=0.62,
        collateral_value=0, business_age_years=0.8,
        cash_flow_volatility=0.80, loan_amount=120_000, sector="hospitality",
    ))
    apps.append(_make_application(
        "med_12", credit_score=530, annual_revenue=220_000, dti=0.68,
        collateral_value=15_000, business_age_years=1.2,
        cash_flow_volatility=0.68, loan_amount=190_000, sector="logistics",
    ))
    apps.append(_make_application(
        "med_13", credit_score=490, annual_revenue=170_000, dti=0.75,
        collateral_value=0, business_age_years=0.5,
        cash_flow_volatility=0.85, loan_amount=140_000, sector="food_services",
    ))
    apps.append(_make_application(
        "med_14", credit_score=555, annual_revenue=280_000, dti=0.65,
        collateral_value=40_000, business_age_years=1.8,
        cash_flow_volatility=0.58, loan_amount=210_000, sector="construction",
    ))

    # --- Refer (balanced signals, genuine uncertainty) ---
    apps.append(_make_application(
        "med_15", credit_score=630, annual_revenue=380_000, dti=0.46,
        collateral_value=70_000, business_age_years=3.0,
        cash_flow_volatility=0.40, loan_amount=95_000, sector="manufacturing",
    ))
    apps.append(_make_application(
        "med_16", credit_score=645, annual_revenue=420_000, dti=0.44,
        collateral_value=85_000, business_age_years=3.5,
        cash_flow_volatility=0.36, loan_amount=100_000, sector="technology",
    ))
    apps.append(_make_application(
        "med_17", credit_score=615, annual_revenue=340_000, dti=0.50,
        collateral_value=60_000, business_age_years=2.8,
        cash_flow_volatility=0.45, loan_amount=88_000, sector="retail",
    ))
    apps.append(_make_application(
        "med_18", credit_score=655, annual_revenue=460_000, dti=0.43,
        collateral_value=95_000, business_age_years=4.0,
        cash_flow_volatility=0.38, loan_amount=105_000, sector="consulting",
    ))
    apps.append(_make_application(
        "med_19", credit_score=620, annual_revenue=360_000, dti=0.48,
        collateral_value=65_000, business_age_years=3.2,
        cash_flow_volatility=0.42, loan_amount=92_000, sector="distribution",
    ))
    apps.append(_make_application(
        "med_20", credit_score=640, annual_revenue=400_000, dti=0.47,
        collateral_value=75_000, business_age_years=3.8,
        cash_flow_volatility=0.39, loan_amount=98_000, sector="healthcare",
    ))

    return apps


# ---------------------------------------------------------------------------
# TASK 3 — HARD (20 applications)
# Deliberately conflicting signals. No single factor gives the answer.
# Designed so frontier LLMs genuinely struggle without assessing ALL factors.
# Mix: 6 approve, 7 reject, 7 refer
#
# Edge case patterns used:
#   A. High revenue + terrible credit (revenue saves it? or credit kills it?)
#   B. Young business + excellent collateral (secured despite age)
#   C. Good credit + extreme DTI (credit vs debt load)
#   D. Old stable business + declining revenue + high loan (track record vs trajectory)
#   E. Volatile cash flow but excellent everything else (volatility vs strength)
#   F. Perfect credit + massive loan-to-revenue (creditworthy but over-leveraged)
# ---------------------------------------------------------------------------

def generate_hard_task() -> list:
    """
    20 applications with deliberately conflicting signals.
    Target distribution: 6 approve / 7 reject / 7 refer.

    Each application is designed around a specific edge-case pattern
    that a naive agent (looking at only one factor) will get wrong.

    Pattern A — High revenue rescues poor credit
    Pattern B — Young business saved by strong collateral
    Pattern C — Good credit but extreme DTI / over-leveraged
    Pattern D — Established business under financial stress
    Pattern E — Volatile cash flow despite otherwise strong profile
    Pattern F — Perfect credit but dangerously over-leveraged
    Pattern G — All factors at the boundary (genuine uncertainty)
    Pattern X — Multiple compounding negatives → clear reject
    """
    apps = []

    # ---------------------------------------------------------------
    # PATTERN A: High revenue rescues poor credit
    # ---------------------------------------------------------------
    # A1: Revenue 2.1M + strong collateral offsets credit=520 → APPROVE
    apps.append(_make_application(
        "hard_01", credit_score=520, annual_revenue=2_100_000, dti=0.25,
        collateral_value=800_000, business_age_years=6.0,
        cash_flow_volatility=0.18, loan_amount=150_000, sector="manufacturing",
    ))
    # A2: Revenue high but credit=490 (below floor) + extreme DTI → REJECT (hard floor)
    apps.append(_make_application(
        "hard_02", credit_score=490, annual_revenue=900_000, dti=0.72,
        collateral_value=50_000, business_age_years=1.5,
        cash_flow_volatility=0.55, loan_amount=400_000, sector="construction",
    ))
    # A3: Revenue OK, credit borderline=535, large loan, balanced stress → REFER
    apps.append(_make_application(
        "hard_03", credit_score=535, annual_revenue=340_000, dti=0.64,
        collateral_value=80_000, business_age_years=2.5,
        cash_flow_volatility=0.60, loan_amount=290_000, sector="logistics",
    ))

    # ---------------------------------------------------------------
    # PATTERN B: Young business, strong collateral
    # ---------------------------------------------------------------
    # B1: Age=1.2yr but collateral=550k covers loan=200k fully → APPROVE (secured)
    apps.append(_make_application(
        "hard_04", credit_score=690, annual_revenue=280_000, dti=0.30,
        collateral_value=550_000, business_age_years=1.2,
        cash_flow_volatility=0.28, loan_amount=200_000, sector="technology",
    ))
    # B2: Young + no collateral + high volatility → REJECT
    apps.append(_make_application(
        "hard_05", credit_score=570, annual_revenue=200_000, dti=0.75,
        collateral_value=0, business_age_years=0.8,
        cash_flow_volatility=0.80, loan_amount=300_000, sector="food_services",
    ))
    # B3: Young, partial collateral, borderline credit → REFER
    apps.append(_make_application(
        "hard_06", credit_score=625, annual_revenue=320_000, dti=0.65,
        collateral_value=70_000, business_age_years=2.2,
        cash_flow_volatility=0.61, loan_amount=300_000, sector="retail",
    ))

    # ---------------------------------------------------------------
    # PATTERN C: Good credit vs. extreme debt load
    # ---------------------------------------------------------------
    # C1: Credit=710 is good, but DTI=0.68 + young + volatile → REFER
    apps.append(_make_application(
        "hard_07", credit_score=710, annual_revenue=400_000, dti=0.68,
        collateral_value=60_000, business_age_years=3.0,
        cash_flow_volatility=0.42, loan_amount=160_000, sector="consulting",
    ))
    # C2: Good credit=720 wiped out by DTI=0.82 + no collateral → REJECT
    apps.append(_make_application(
        "hard_08", credit_score=720, annual_revenue=340_000, dti=0.82,
        collateral_value=0, business_age_years=4.0,
        cash_flow_volatility=0.35, loan_amount=200_000, sector="hospitality",
    ))
    # C3: Excellent credit=795, DTI=0.58 looks high but revenue+collateral absorb it → APPROVE
    apps.append(_make_application(
        "hard_09", credit_score=795, annual_revenue=920_000, dti=0.58,
        collateral_value=380_000, business_age_years=8.0,
        cash_flow_volatility=0.14, loan_amount=180_000, sector="healthcare",
    ))

    # ---------------------------------------------------------------
    # PATTERN D: Established business under stress
    # ---------------------------------------------------------------
    # D1: 10yr history but DTI=0.68, high volatility, large loan → REFER
    apps.append(_make_application(
        "hard_10", credit_score=565, annual_revenue=480_000, dti=0.68,
        collateral_value=120_000, business_age_years=10.0,
        cash_flow_volatility=0.58, loan_amount=320_000, sector="manufacturing",
    ))
    # D2: Very established=12yr but extreme DTI=0.74 + high vol → REJECT
    apps.append(_make_application(
        "hard_11", credit_score=545, annual_revenue=280_000, dti=0.72,
        collateral_value=20_000, business_age_years=1.2,
        cash_flow_volatility=0.78, loan_amount=400_000, sector="construction",
    ))
    # D3: 11yr history, moderate stress absorbed by revenue+collateral → APPROVE
    apps.append(_make_application(
        "hard_12", credit_score=700, annual_revenue=700_000, dti=0.36,
        collateral_value=320_000, business_age_years=11.0,
        cash_flow_volatility=0.30, loan_amount=130_000, sector="technology",
    ))

    # ---------------------------------------------------------------
    # PATTERN E: Volatile cash flow vs otherwise strong profile
    # ---------------------------------------------------------------
    # E1: Credit=755, good revenue, but vol=0.82 + big loan → REFER
    apps.append(_make_application(
        "hard_13", credit_score=755, annual_revenue=500_000, dti=0.55,
        collateral_value=100_000, business_age_years=3.0,
        cash_flow_volatility=0.82, loan_amount=300_000, sector="technology",
    ))
    # E2: Vol=0.88 destroys otherwise decent profile → REJECT
    apps.append(_make_application(
        "hard_14", credit_score=530, annual_revenue=160_000, dti=0.78,
        collateral_value=0, business_age_years=0.5,
        cash_flow_volatility=0.85, loan_amount=350_000, sector="food_services",
    ))
    # E3: Vol=0.50 but excellent credit+collateral absorb it → APPROVE
    apps.append(_make_application(
        "hard_15", credit_score=780, annual_revenue=740_000, dti=0.24,
        collateral_value=500_000, business_age_years=7.0,
        cash_flow_volatility=0.50, loan_amount=100_000, sector="manufacturing",
    ))

    # ---------------------------------------------------------------
    # PATTERN F: Excellent credit, dangerously over-leveraged
    # ---------------------------------------------------------------
    # F1: Credit=820 but loan=500k vs revenue=150k, no collateral → REFER
    apps.append(_make_application(
        "hard_16", credit_score=820, annual_revenue=150_000, dti=0.55,
        collateral_value=0, business_age_years=2.0,
        cash_flow_volatility=0.20, loan_amount=500_000, sector="consulting",
    ))
    # F2: Credit=780 but loan=550k vs revenue=180k, tiny collateral → REFER
    apps.append(_make_application(
        "hard_17", credit_score=780, annual_revenue=180_000, dti=0.60,
        collateral_value=10_000, business_age_years=1.5,
        cash_flow_volatility=0.25, loan_amount=550_000, sector="retail",
    ))
    # F3: Credit=800, revenue=1.2M makes large loan reasonable → APPROVE
    apps.append(_make_application(
        "hard_18", credit_score=800, annual_revenue=1_200_000, dti=0.27,
        collateral_value=600_000, business_age_years=8.0,
        cash_flow_volatility=0.16, loan_amount=280_000, sector="logistics",
    ))

    # ---------------------------------------------------------------
    # PATTERN G: All factors at the boundary — genuine uncertainty
    # ---------------------------------------------------------------
    # G1: Every metric sits at the refer boundary → REFER
    apps.append(_make_application(
        "hard_19", credit_score=648, annual_revenue=300_000, dti=0.60,
        collateral_value=60_000, business_age_years=2.0,
        cash_flow_volatility=0.58, loan_amount=260_000, sector="distribution",
    ))
    # G2: Construction sector + boundary metrics + volatility → REFER
    apps.append(_make_application(
        "hard_20", credit_score=655, annual_revenue=350_000, dti=0.62,
        collateral_value=70_000, business_age_years=2.5,
        cash_flow_volatility=0.60, loan_amount=280_000, sector="construction",
    ))

    return apps


# ---------------------------------------------------------------------------
# Validation — sanity-check all generated tasks
# ---------------------------------------------------------------------------

def validate_tasks(tasks: dict) -> None:
    """Print a summary and assert key invariants."""
    print("\n" + "=" * 60)
    print("  DATASET VALIDATION REPORT")
    print("=" * 60)

    for task_id, apps in tasks.items():
        decisions = [a["decision"] for a in apps]
        approve_n = decisions.count("approve")
        reject_n  = decisions.count("reject")
        refer_n   = decisions.count("refer")
        risks     = [a["risk_score"] for a in apps]

        print(f"\n  Task: {task_id.upper()}  ({len(apps)} applications)")
        print(f"    Approve: {approve_n}  |  Reject: {reject_n}  |  Refer: {refer_n}")
        print(f"    Risk scores: min={min(risks):.3f}  max={max(risks):.3f}  "
              f"avg={sum(risks)/len(risks):.3f}")

        # Assert all fields present
        required = [
            "application_id", "business_name", "sector", "annual_revenue",
            "credit_score", "dti", "collateral_value", "business_age_years",
            "cash_flow_volatility", "loan_amount", "loan_to_revenue",
            "collateral_coverage", "risk_score", "decision",
            "factor_directions", "explanation",
        ]
        for app in apps:
            for field in required:
                assert field in app, f"Missing field '{field}' in {app.get('application_id')}"
            assert 0.0 <= app["risk_score"] <= 1.0, \
                f"risk_score out of range in {app['application_id']}: {app['risk_score']}"
            assert app["decision"] in ("approve", "reject", "refer"), \
                f"Invalid decision in {app['application_id']}: {app['decision']}"

        # Grader sanity: verify determinism
        for app in apps:
            r2 = compute_risk_score(app)
            assert r2 == app["risk_score"], \
                f"Non-deterministic risk score in {app['application_id']}"

        # Check task 1 has no "refer" decisions (easy should be clear-cut)
        if task_id == "easy":
            borderline = [a for a in apps if a["decision"] == "refer"]
            if borderline:
                print(f"    NOTE: {len(borderline)} refer(s) in easy task "
                      f"({[b['application_id'] for b in borderline]})")

        print(f"    All {len(apps)} applications validated OK")

    # Verify all app IDs are unique across tasks
    all_ids = [a["application_id"] for apps in tasks.values() for a in apps]
    assert len(all_ids) == len(set(all_ids)), "Duplicate application IDs detected!"

    print(f"\n  Total applications: {len(all_ids)}")
    print("  All validations passed.\n")


# ---------------------------------------------------------------------------
# Main — generate and save
# ---------------------------------------------------------------------------

def main():
    print("Generating SME Credit Risk synthetic dataset...")
    print(f"Random seed: {SEED} (fully reproducible)")

    easy_apps   = generate_easy_task()
    medium_apps = generate_medium_task()
    hard_apps   = generate_hard_task()

    tasks = {
        "easy":   easy_apps,
        "medium": medium_apps,
        "hard":   hard_apps,
    }

    validate_tasks(tasks)

    # Build the final JSON structure
    output = {
        "metadata": {
            "seed":              SEED,
            "version":           "1.0.0",
            "description":       "SME Credit Risk Assessment — synthetic dataset",
            "ground_truth_formula": {
                "weights": {
                    "credit_score":          0.30,
                    "dti":                   0.25,
                    "loan_to_revenue":       0.20,
                    "business_age":          0.15,
                    "cash_flow_volatility":  0.10,
                },
                "collateral_discount":   0.20,
                "approve_threshold":     APPROVE_THRESHOLD,
                "reject_threshold":      REJECT_THRESHOLD,
                "hard_floor_credit":     HARD_FLOOR_CREDIT,
            },
            "task_counts": {k: len(v) for k, v in tasks.items()},
            "decision_counts": {
                task_id: {
                    "approve": sum(1 for a in apps if a["decision"] == "approve"),
                    "reject":  sum(1 for a in apps if a["decision"] == "reject"),
                    "refer":   sum(1 for a in apps if a["decision"] == "refer"),
                }
                for task_id, apps in tasks.items()
            },
        },
        "tasks": tasks,
    }

    out_path = Path(__file__).parent / "tasks.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Dataset written to: {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024:.1f} KB")

    # Print a sample application so the developer can inspect the output
    print("\n  Sample application (easy_01):")
    sample = easy_apps[0]
    for k, v in sample.items():
        print(f"    {k:28s}: {v}")

    print("\nDone. Run your environment with: uvicorn server.app:app --port 7860")


if __name__ == "__main__":
    main()
