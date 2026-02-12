# 🚀 AEZ Evolution - Demo Guide

## Quick Start

```bash
pip install numpy fastapi uvicorn pydantic
python3 demo.py
```

---

## Three Ways to Experience It

### 1. Narrated Demo (Recommended)
```bash
python3 demo.py
```
Watch 60 agents evolve over 100 rounds with narrative storytelling.

### 2. Interactive Visualization
```bash
# Terminal 1: Start API
cd orchestrator && python -m uvicorn server:app --port 8000

# Browser: Open dashboard/trust-cascade-demo.html
```

### 3. Quick Test
```bash
python3 test_all.py
```
Runs 145 tests — all should pass.

---

## What You'll See

**The Trust Cascade:**

| Phase | Rounds | What Happens |
|-------|--------|--------------|
| Chaos | 1-20 | Defectors dominate |
| Clustering | 21-40 | Cooperators find each other |
| Learning | 41-60 | Agents adapt strategies |
| Stabilization | 61-80 | Networks form |
| Equilibrium | 81-100 | Cooperation wins |

---

## Demo Tips for Judges

**Opening:** *"AI agents learning to cooperate. No humans. All verifiable."*

**Key moments to highlight:**
- Round 20: Defectors winning
- Round 50: Clusters forming
- Round 100: Cooperation emerged

**Closer:** *"We didn't program cooperation—we created conditions for it to emerge."*

---

## Troubleshooting

**API won't start:**
```bash
pip install fastapi uvicorn
```

**Demo errors:**
```bash
pip install numpy pydantic
```

---

## Files

| File | Purpose |
|------|---------|
| `demo.py` | Main narrated demo |
| `test_all.py` | Test suite (145 tests) |
| `dashboard/` | Web visualizations |
| `engine/` | Core simulation |

---

*"The network learned to trust."*
