# MOXEC Full Pipeline

Runs the complete MOXEC experiment (search + baselines + evaluation) across all 12
datasets and saves every result, figure, and table to `./outputs/`.

## Setup

```bash
pip install -r requirements.txt
```

## Run everything

```bash
python moxec_full_pipeline.py
```

That's it — no arguments needed. This can take **25–40+ hours** depending on hardware,
so run it in the background (e.g. `nohup python moxec_full_pipeline.py &`) or in a
persistent terminal session (`tmux`/`screen`). It resumes automatically if
interrupted — just re-run the same command.
