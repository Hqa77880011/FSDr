# FSDr

Implementation of **Forgetting-Aware Support-Gated Doubly Robust Evaluation**. A past-only attention outcome model includes concept-dependent forgetting and a shrinkage-controlled learner-history adjustment. Sequential doubly robust evaluation combines fitted values with calibrated behavior probabilities; a support gate controls release of each comparison.

## Installation

Requires Python `>=3.10`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, scikit-learn 1.6.1, matplotlib 3.10.0. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data and preprocessing

Generate chronological finite learner logs; private ability/forgetting is used only by the generator:

```bash
python finite_data.py --output data/learners.csv --learners 60 --seed 42
fsdr prepare --csv data/learners.csv --output data/prepared --history 20 --horizon 20 --time-unit-seconds 3600 --seed 42
```

External ASSISTments/EdNet CSV uses `learner,exercise,concept,correct,timestamp` with binary correctness and timestamps in seconds. Fix each exercise's concept mapping. Preparation assigns complete learners to train/propensity-calibration/evaluation roles.

## Training and inference

```bash
fsdr fit --data data/prepared --output runs/fit --folds 5 --epochs 30 --width 48 --seed 42
```

## Evaluation

```bash
fsdr evaluate --data data/prepared --models runs/fit --output runs/evaluate --folds 5 --bootstrap 1000 --seed 42
```

## Comparisons and ablations

```bash
fsdr fit --data data/prepared --seed 42 --mode no_time --output runs/no_time
fsdr fit --data data/prepared --seed 42 --mode global --output runs/global
fsdr fit --data data/prepared --seed 42 --mode time --output runs/time
fsdr evaluate --data data/prepared --seed 42 --models runs/no_time --output runs/no_time_eval
fsdr evaluate --data data/prepared --seed 42 --models runs/global --output runs/global_eval
fsdr evaluate --data data/prepared --seed 42 --models runs/time --output runs/time_eval
```

## Outputs

Fitting stores one checkpoint per policy/fold. Evaluation writes `report.json` with release status, DR/IPS/SNIPS/DM diagnostics and intervals, plus `support.png`. Gate failures keep diagnostic estimates while released values stay null. The complete pipeline accepts optional external `--csv`:

```bash
python reproduce.py --plan
python reproduce.py --output outputs/finite-setting
```
