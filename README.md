# CoSpec

Coordinated adaptive speculative decoding for distributed LLM serving
(CMU 11-868 LLM Systems project, Alvin Wang and Arjun Shah).

In DP-attention / expert-parallel serving, ranks advance in lockstep, so the
slowest rank sets every step's time. CoSpec chooses speculation budgets for
all ranks jointly and co-designs request routing with them. One V100 supplies
real step-time profiles and acceptance traces; a trace-driven simulator
replays them across many virtual lockstep ranks.

The proposal is in [paper/proposal/](paper/proposal/).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-sim.txt   # simulator + tests
.venv/bin/pip install -r requirements-gpu.txt   # GPU scripts (on the V100 machine)
.venv/bin/python -m pytest
```
