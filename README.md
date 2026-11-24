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

## GPU runs on Google Colab

[colab/cospec_colab.ipynb](colab/cospec_colab.ipynb) runs every GPU step (SD check, profile and
cost-model fit, traces, validation) and writes outputs to Google Drive. Run
`colab/make_code_zip.sh`, upload `colab/cospec_code.zip` to `MyDrive/cospec_run/`, open the
notebook in Colab with a GPU runtime and run all cells. Every step checkpoints to Drive and prints
progress with an ETA; after a disconnect, run all cells again to resume. Unzip the `cospec_outputs.zip` it downloads
in the repo root to get `data/` and `results/` for local sweeps.
