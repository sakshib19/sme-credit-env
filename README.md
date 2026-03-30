# SME Credit Risk RL Environment

A multi-step reinforcement learning environment for SME loan decisions, built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

An agent evaluates a small business loan application **step by step** — revealing financial factors one at a time, then making a final decision. This is a genuine RL problem, not a classification task.

---

## The Core Idea

```
reset("easy_01")                    → hidden application loaded
step("assess_credit_score")         → credit_score = 780  [POSITIVE] +0.10
step("assess_dti")                  → dti = 0.28          [POSITIVE] +0.10
step("decide_approve")              → CORRECT ✓            +1.25
```

At each step the agent chooses to **reveal a factor** or **make a final decision**. Fewer reveals = higher efficiency bonus. Wrong decision = penalty. The episode ends when a decision is made.

---

## Action Space

### Assess actions (reveal one hidden factor)
| Action | Reveals |
|---|---|
| `assess_revenue` | `annual_revenue` (GBP) |
| `assess_credit_score` | `credit_score` (300–850) |
| `assess_dti` | `dti` (debt-to-income ratio 0–1) |
| `assess_collateral` | `collateral_value` (GBP) |
| `assess_business_age` | `business_age_years` |
| `assess_cash_flow` | `cash_flow_volatility` (0=stable, 1=volatile) |

### Decision actions (end the episode)
| Action | Meaning |
|---|---|
| `decide_approve` | Approve the loan |
| `decide_reject` | Reject the loan |
| `decide_refer` | Refer to senior underwriter |

---

## Reward Structure

| Event | Reward |
|---|---|
| Reveal informative factor (strong signal) | **+0.10** |
| Reveal neutral factor | **+0.05** |
| Reveal duplicate factor | **−0.05** |
| Invalid action / wrong application_id | **−0.10** |
| Correct decision (easy / medium) | **+1.00** |
| Correct decision (hard) | **+1.20** |
| Refer decision (partial credit) | **+0.30** |
| Wrong decision | **−0.50** |
| Efficiency bonus (per unrevealed factor at decision time) | **+0.05 each** |
| Timeout — no decision in 8 steps | **−0.20** |

Maximum possible score per episode: **+1.50** (correct hard decision, zero reveals).

---

## Dataset

50 synthetic SME applications generated deterministically (seed=42) by `generate_dataset.py`.

| Tier | Count | Description |
|---|---|---|
| **Easy** | 10 | One dominant factor drives the decision |
| **Medium** | 20 | Multi-factor — need 3+ reveals to decide well |
| **Hard** | 20 | Conflicting signals — designed so agents struggle without all factors |

Ground truth computed by a weighted risk formula (no LLM involved):

```
risk_score = credit_penalty×0.30 + dti_penalty×0.25 + ltr_penalty×0.20
           + age_penalty×0.15 + volatility_penalty×0.10
           × (1 − 0.20 × collateral_coverage)

approve  if risk_score < 0.35
reject   if risk_score > 0.65  (or hard floors: credit < 500, dti > 0.80)
refer    otherwise
```

---

## Grading

Three graders (`env/graders.py`) evaluate completed episodes on a **[0, 1]** scale:

| Tier | Correctness weight | Efficiency weight |
|---|---|---|
| easy | 80% | 20% |
| medium | 65% | 35% |
| hard | 50% | 50% |

Efficiency score = `1 − (reveals / 6)`. A wrong decision after revealing all 6 factors on hard gets an extra −0.10 surcharge.

---

## Project Structure

```
sme-credit-env/
├── models.py              ← Type-safe dataclasses (LoanAction, LoanObservation, LoanState)
├── generate_dataset.py    ← Synthetic dataset generator
├── tasks.json             ← 50 pre-generated applications
├── inference.py           ← LLM + heuristic agent, CLI evaluation runner
├── requirements.txt
├── Dockerfile
├── openenv.yaml
│
├── env/
│   ├── __init__.py
│   ├── environment.py     ← Core RL loop (LoanEnvironment)
│   └── graders.py         ← Deterministic scoring functions
│
└── server/
    ├── __init__.py
    └── app.py             ← FastAPI server
```

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Run the heuristic agent locally

```bash
python inference.py                   # one easy episode
python inference.py --tier hard       # one hard episode
python inference.py --eval            # all 10 easy tasks
python inference.py --eval --all      # all 50 tasks
```

### 3. Run the LLM agent

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python inference.py --llm --tier medium
python inference.py --llm --eval --all
```

### 4. Start the server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### 5. Call the server

```bash
# List all tasks
curl http://localhost:7860/tasks

# Start an episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "hard_01"}'

# Reveal a factor
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "assess_credit_score", "application_id": "hard_01"}'

# Make a decision
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "decide_approve", "application_id": "hard_01"}'

# Get full state (includes ground truth)
curl http://localhost:7860/state

# Grade an episode
curl -X POST http://localhost:7860/grade \
  -H "Content-Type: application/json" \
  -d '{"action_log": [...], "ground_truth": "approve", "task_id": "hard"}'
```

### 6. Docker

```bash
docker build -t sme-credit-env:latest .
docker run -d -p 7860:7860 sme-credit-env:latest
```

### 7. Deploy to HF Spaces

```bash
openenv push --repo-id YOUR_USERNAME/sme-credit-env
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
    # your policy here
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

## Run the Agent Against the Server

```bash
# Start server in one terminal
uvicorn server.app:app --port 7860

# Run agent against it in another
python inference.py --mode remote --eval --all
```

---

## Sample Episode Trace

```
============================================================
  Episode: hard_04  [HARD]
  Business: Ironbridge Technologies Ltd  |  Sector: technology
  Loan: £200,000
============================================================

  Step 1: assess_credit_score
  → Reward: +0.100  | Revealed credit_score = 690 [POSITIVE]

  Step 2: assess_collateral
  → Reward: +0.100  | Revealed collateral_value = £550,000 [POSITIVE]

  Step 3: assess_dti
  → Reward: +0.100  | Revealed dti = 0.300 [POSITIVE]

  Step 4: decide_approve
  → Reward: +1.350  | Decision: APPROVE. CORRECT ✓

  ──────────────────────────────────────────────────────
  Decision: APPROVE     Ground truth: APPROVE
  Outcome:  ✓ CORRECT
  Reward:   1.4000   Grade score: 0.8333
  Reveals:  3 / 6   Steps: 4
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  AGENT (inference.py)                               │
│  HeuristicAgent | LLMAgent (Claude)                 │
└─────────────────────┬───────────────────────────────┘
                      │  reset / step / grade
          ┌───────────┴──────────┐
          │  local               │  remote HTTP
          ▼                      ▼
┌──────────────────┐   ┌──────────────────────────────┐
│  LoanEnvironment │   │  FastAPI Server (server/app)  │
│  (env/)          │   │  → wraps LoanEnvironment      │
└──────────────────┘   └──────────────────────────────┘
          │
          ▼
┌──────────────────┐
│  tasks.json      │  50 synthetic SME applications
│  (ground truth)  │  easy / medium / hard
└──────────────────┘
```
