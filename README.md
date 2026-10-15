# FSDr

Implementation of **Forgetting-Aware Support-Gated Doubly Robust Evaluation**. A past-only attention outcome model includes concept-dependent forgetting and a shrinkage-controlled learner-history adjustment. Sequential doubly robust evaluation combines fitted values with calibrated behavior probabilities; a support gate controls release of each comparison.

## Install and prepare logs

Use Python 3.10+ and activate a virtual environment:

```bash
python -m venv .venv
python -m pip install -e .
fsdr prepare --csv data/interactions.csv --output data/prepared
fsdr fit --data data/prepared --output runs/fit
fsdr evaluate --data data/prepared --models runs/fit --output runs/evaluate
```

Export ordered ASSISTments or EdNet-KT1 interactions using the following CSV columns:

```text
learner,exercise,concept,correct,timestamp
learner01,exercise01,concept01,1,1700000000
learner01,exercise02,concept02,0,1700003600
```

Use actual learner/exercise identifiers, binary correctness and numeric timestamps in seconds. Each exercise needs a fixed concept mapping; resolve multi-concept tagging consistently before export. Preparation defines the exercise/concept vocabulary from identifiers, sorts each learner chronologically and assigns complete learners to 70% training, 10% propensity calibration and 20% evaluation. No response outcomes are used to construct the vocabulary.

The default horizon is 20 and history length is 20. Nonoverlapping complete episodes are retained; short final fragments are excluded. Every state contains only earlier interactions, including history preceding an episode. `--time-unit-seconds` defaults to 3600, making elapsed time an hourly quantity. The terminal state cannot contribute reward beyond the finite horizon.

## Outcome fitting and policy definitions

Five learner folds are used by default. Each nuisance model fits four training folds, excludes all evaluation learners and temperature-calibrates behavior logits using the separate calibration learners. Evaluation learners are assigned to one frozen fold model; no scored learner appears in its fitting identities. The outcome head is trained by fitted-Q Bellman regression for a fixed target policy, with a separate immediate-correctness objective for AUC diagnostics.

The executable target policies are uniform exercise choice, training-popularity choice, concept knowledge-gap choice, and a strongly concentrated gap choice (`aggressive`). The gap score uses Beta-smoothed prior incorrectness by concept. All operate from pre-decision history. The current action bank is treated as feasible for every state; construct a prerequisite-feasible action bank before export if the application restricts exercise availability. The aggressive policy is a support stressor, not labeled as a separately trained offline-RL policy.

The personalized decay combines global concept rates and an amortized learner-history offset, shrunk by history count. This supplies a personalized rate for previously unseen evaluation learners without fitting a free parameter on their future outcomes. The `global`, `time` and `no_time` modes change only the decay mechanism:

```bash
fsdr fit --mode no_time --output runs/no_time
fsdr fit --mode global --output runs/global
fsdr fit --mode time --output runs/time
fsdr evaluate --models runs/no_time --output runs/no_time_eval
fsdr evaluate --models runs/global --output runs/global_eval
fsdr evaluate --models runs/time --output runs/time_eval
```

`--policies` selects a common policy list at fitting and evaluation. `--width`, `--epochs`, `--batch-size`, `--lr`, `--folds`, `--seed` and `--device` control nuisance fitting. These are explicit implementation choices for the manuscript's unspecified attention layers and fitting schedule.

## Support and intervals

Sequential ratios accumulate over the horizon. DR clipping sensitivity uses 10, 20 and 50; the primary estimate uses 20. ESS uses the untruncated ratios with numerically stable scaling so clipping cannot conceal poor overlap. The release rule requires:

- minimum fitted behavior probability at actions with target probability above `--eta 0.01` of at least `--min-propensity 0.02`;
- minimum horizon ESS divided by trajectory count of at least `--min-ess 0.1`;
- maximum DR change across clipping levels no greater than `--max-clipping-change 0.02`.

Values are divided by the maximum discounted return `sum(0.95**t)` so the clipping-change threshold has the declared normalized scale. IPS and SNIPS use the same primary clipped cumulative ratios. DM is the direct plug-in value of the fitted-Q model; it is named `DM_FQE` rather than reporting an identical plug-in calculation as a second independent estimator.

Confidence intervals resample complete learners, keeping all their episodes together. `--bootstrap` defaults to 1000. Shared resamples produce paired policy-difference intervals. A comparison is released only if both policies pass the gate. Unsupported values remain explicitly under diagnostics; the released estimate and difference are `null`.

## Oracle-backed validation metrics

When policy truth is available from an independently specified simulator, pass a JSON mapping policy names to true normalized values:

```bash
fsdr evaluate --truth data/oracle_values.json --output runs/oracle_evaluation
fsdr summarize --reports run1/report.json run2/report.json run3/report.json --output summary.json
```

The truth file must contain measured or simulated oracle estimates, not observational estimates relabeled as truth. Evaluation calculates absolute bias, squared error, interval inclusion and policy-rank Spearman correlation. `summarize` computes mean absolute bias, RMSE and nominal-interval coverage across independent oracle-backed reports. It rejects reports that lack oracle truth. Observational logs alone do not supply those counterfactual validation quantities.

## Outputs

Fitting writes one checkpoint per target policy and learner fold. `report.json` contains gate status, released DR values/intervals, IPS/SNIPS/DM diagnostics, clipping sensitivity, minimum propensity, minimum ESS ratio, immediate correctness AUC, long-gap AUC, and paired value-difference intervals. `--long-gap` defaults to 24 hours; no prior practice is treated as an unbounded gap. A single-class AUC stratum is `null`. `support.png` visualizes the ESS diagnostic and release threshold.

The behavioral model and Q model are compact attention networks, not an imported AKT implementation. State construction, concept-conditioned decay, finite-horizon Bellman targets, learner separation and the selective release rule are explicit in the source. Gate thresholds remain fixed before target-policy values are examined.
