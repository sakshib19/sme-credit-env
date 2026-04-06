---
title: SME Credit Risk RL Environment
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - finance
  - credit-risk
---

# SME Credit Risk RL Environment

A multi-step reinforcement learning environment for **small business loan underwriting**, built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

An agent assesses loan applications **step by step** — revealing financial factors one at a time, then making a final approve / reject / refer decision. This mirrors how real loan officers work: gather evidence incrementally, decide when confident enough.

---

## The Task

```
reset("hard_01")                      → hidden application loaded, all factors concealed
step("assess_credit_score")           → credit_score = 520  [NEGATIVE]  +0.10
step("assess_dti")                    → dti = 0.25          [POSITIVE]  +0.10
step("assess_revenue")                → revenue = £2.1M     [POSITIVE]  +0.10
step("decide_approve")                → CORRECT ✓           +1.35  (hard bonus)

[END] success=true steps=4 score=0.750 rewards=0.10,0.10,0.10,1.35
```

The agent earns rewards for revealing informative factors, and a large terminal reward for the correct decision. Efficiency bonus: **+0.05 per unrevealed factor** at decision time — sharper agents score higher.

---

## Action Space

### Assess actions — reveal one hidden factor
| Action | Reveals | Signal weight |
|---|---|---|
| `assess_credit_score` | credit_score (300–850) | 0.30 |
| `assess_dti` | debt-to-income ratio (0–1) | 0.25 |
| `assess_revenue` | annual_revenue GBP + loan_to_revenue | 0.20 |
| `assess_business_age` | business_age_years | 0.15 |
| `assess_collateral` | collateral_value GBP | discount |
| `assess_cash_flow` | cash_flow_volatility (0=stable) | 0.10 |

### Decision actions — end the episode
| Action | Meaning |
|---|---|
| `decide_approve` | Approve the loan |
| `decide_reject` | Reject the loan |
| `decide_refer` | Refer to senior underwriter |

---

## Observation Space

| Field | Type | Always visible? |
|---|---|---|
| `application_id` | str | ✓ |
| `task_id` | str (easy/medium/hard) | ✓ |
| `business_name`, `sector` | str | ✓ |
| `loan_amount` | float (GBP) | ✓ |
| `loan_to_revenue` | float | After assess_revenue |
| `credit_score` | int 300–850 | After assess_credit_score |
| `dti` | float 0–1 | After assess_dti |
| `annual_revenue` | float | After assess_revenue |
| `collateral_value` | float | After assess_collateral |
| `business_age_years` | float | After assess_business_age |
| `cash_flow_volatility` | float 0–1 | After assess_cash_flow |
| `factors_assessed` | list[str] | ✓ |
| `factors_remaining` | list[str] | ✓ |
| `step_count`, `max_steps` | int | ✓ |
| `cumulative_reward` | float | ✓ |
| `feedback` | str | ✓ |
| `done`, `reward` | bool, float | ✓ |

---

## Reward Structure

| Event | Reward |
|---|---|
| Reveal informative factor (strong signal) | **+0.10** |
| Reveal neutral factor | **+0.05** |
| Reveal duplicate factor | **−0.05** |
| Wrong application_id or invalid action | **−0.10** |
| Correct decision (easy / medium) | **+1.00** |
| Correct decision (hard) | **+1.20** |
| Refer (partial credit) | **+0.30** |
| Wrong decision | **−0.50** |
| Efficiency bonus (per unrevealed factor) | **+0.05 each** |
| Timeout — no decision in max_steps | **−0.20** |

Maximum possible: **+1.50** (correct hard decision, zero reveals).

---

## Risk Formula (ground truth)

Applications are scored by a weighted risk formula:

```
credit_penalty = (850 - credit_score) / 550          # weight 0.30
dti_penalty    = piecewise(dti)                       # weight 0.25
ltr_penalty    = piecewise(loan / revenue)            # weight 0.20
age_penalty    = piecewise(business_age_years)        # weight 0.15
vol_penalty    = cash_flow_volatility * 1.2           # weight 0.10

raw_risk  = weighted_sum_of_penalties
collateral_discount = 1 - 0.20 * min(collateral / loan, 2.0)
risk_score = raw_risk * collateral_discount

APPROVE  if risk_score < 0.35
REJECT   if risk_score > 0.65   (or hard floors: credit<500, dti>0.80)
REFER    otherwise
```

---

## Task Tiers

| Tier | Count | Description |
|---|---|---|
| **easy** | 10 | Clear single or dual-factor decisions |
| **medium** | 20 | Multi-factor analysis required |
| **hard** | 20 | Conflicting signals — borderline applications |

50 synthetic SME applications, generated deterministically (seed=42).

---

## Grading

Three deterministic graders (`env/graders.py`) score completed episodes on **[0.0, 1.0]**:

| Tier | Correctness weight | Efficiency weight |
|---|---|---|
| easy | 80% | 20% |
| medium | 65% | 35% |
| hard | 50% | 50% |

`efficiency = 1 − (n_reveals / 6)`. Hard tier penalises wrong decisions after revealing all 6 factors.

---

## Baseline Scores

Heuristic agent (deterministic, no API key needed):

| Tier | Accuracy | Avg Score | Avg Reveals |
|---|---|---|---|
| Easy | 9/10 (90%) | 0.845 | 3.2 |
| Medium | 20/20 (100%) | 0.807 | 3.3 |
| Hard | 19/20 (95%) | 0.667 | 3.7 |
| **Overall** | **48/50 (96%)** | **0.759** | **3.3** |

---

## Project Structure

```
sme-credit-env/
├── inference.py           ← baseline agent + [START][STEP][END] logs
├── models.py              ← LoanAction, LoanObservation, LoanState dataclasses
├── tasks.json             ← 50 pre-generated applications (root copy)
├── openenv.yaml           ← OpenEnv metadata manifest
├── Dockerfile
├── requirements.txt
├── README.md
├── data/
│   ├── generate_dataset.py   ← reproducible dataset generator (seed=42)
│   └── tasks.json            ← primary data location
├── env/
│   ├── environment.py     ← LoanEnvironment: reset() / step() / state
│   └── graders.py         ← grade_easy() / grade_medium() / grade_hard()
└── server/
    └── app.py             ← FastAPI server (OpenEnv HTTP endpoints)
```

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Run heuristic baseline (no API key needed)

```bash
python inference.py --eval --all
```

Output:
```
[START] task=easy_01 env=sme-credit-env model=...
[STEP] step=1 action=assess_credit_score reward=0.10 done=false error=null
[STEP] step=4 action=decide_approve reward=1.15 done=true error=null
[END] success=true steps=4 score=0.900 rewards=0.10,0.10,0.10,1.15
```

### 3. Run LLM agent

Set environment variables in `.env`:

```dotenv
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
HF_TOKEN=hf_your_token_here
GROQ_API_KEY=gsk_your_groq_key   # optional: free fallback if HF credits run out
```

```bash
python inference.py --llm --eval --all
```

### 4. Start the server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 5. API endpoints

```bash
# Health check
curl http://localhost:7860/health

# Start episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "hard_01"}'

# Reveal a factor
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "assess_credit_score", "application_id": "hard_01"}'

# Make decision
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "decide_approve", "application_id": "hard_01"}'

# Full internal state (includes ground truth)
curl http://localhost:7860/state

# Grade a completed episode
curl -X POST http://localhost:7860/grade \
  -H "Content-Type: application/json" \
  -d '{"action_log": [...], "ground_truth": "approve", "task_id": "hard"}'

# List all 50 tasks
curl http://localhost:7860/tasks
```

### 6. Docker

```bash
docker build -t sme-credit-env:latest .
docker run -d -p 7860:7860 \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct \
  -e HF_TOKEN=hf_your_token \
  sme-credit-env:latest
```

### 7. Run against server

```bash
python inference.py --mode remote --url http://localhost:7860 --eval --all
```

---

## Python API

```python
from env.environment import LoanEnvironment
from env.graders import grade
from models import LoanAction

env = LoanEnvironment()
obs = env.reset("hard_04")

while not obs.done:
    action = LoanAction(
        action_type="assess_credit_score",
        application_id=obs.application_id,
    )
    obs = env.step(action)

state = env.state
score = grade(state.action_log, state.ground_truth_decision, state.task_id)
print(f"Score: {score:.3f}")
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | Yes (LLM mode) | LLM API endpoint |
| `MODEL_NAME` | Yes (LLM mode) | Model identifier |
| `HF_TOKEN` | Yes (LLM mode) | Hugging Face / API key |
| `GROQ_API_KEY` | No | Auto-fallback when HF credits exhausted |
| `PORT` | No (default 7860) | Server port |
| `WORKERS` | No (default 2) | Uvicorn worker processes |

---

## Design Notes

**Why SME lending?** Small business credit decisions are genuinely hard — they involve noisy, correlated signals, hard floors, and borderline cases where a wrong call costs real money. It's a natural fit for multi-step RL: the agent must decide *which* information to gather and *when* it has enough to decide, balancing thoroughness against efficiency.

**Why multi-step?** A single-shot classifier would memorise the formula. By hiding factors and charging efficiency bonuses, we create an information-gathering RL problem where the agent must learn *sequential decision-making under uncertainty*, not just pattern matching.

**Why three tiers?** Easy cases test whether the agent catches hard floors and clear signals. Medium cases require combining 3+ factors. Hard cases have intentionally conflicting signals (e.g., excellent collateral but poor credit) that land in the 0.35–0.65 borderline zone — these are where frontier models genuinely struggle.

