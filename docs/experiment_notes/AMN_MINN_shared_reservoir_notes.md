# Shared AMN/MINN Reservoir Notes

This file is the source of truth for the iML1515 FluxTransformer reservoir used
across both AMN-style growth prediction and MINN-style flux/pFBA experiments.
Keep task-specific implementation detail in `AMN_experiment_notes.md` and
`MINN_training_notes.md`; record shared data, checkpoint compatibility, trials,
and cross-task conclusions here.

## Update Contract

Update this note in the same change whenever any of the following occurs:

- the shared generator, its defaults, regimes, input order, bounds, or objective
  changes;
- either shared trial notebook changes its checkpoint, data, preprocessing,
  architecture, validation protocol, metrics, or interpretation;
- a shared training trial, checkpoint, model directory, or dataset is added,
  replaced, promoted, or rejected;
- `flux_transformer.py` or a training script changes in a way that affects the
  shared checkpoint or its consumers; or
- a new AMN/MINN shared-reservoir file is introduced.

For every material trial, add or update a row in the trial registry and record
the exact model, data, changed variable, evaluation protocol, main results, and
decision. Do not copy metrics from incomplete runs or infer missing checkpoint
metadata from a filename without labeling the inference.

The repository post-edit hook flags related changed files when this note is not
also changed. The hook detects documentation drift; the experimenter or Codex
must still write an evidence-based update.

## Main Files

- `generate_ecoli_iML1515_AMN_MINN_data.py`: shared simulated-data generator.
- `generate_ecoli_iML1515_AB_union_data.py`: balanced literal A union B
  simulated-data generator.
- `iML1515_sampling_study_notes.md`: broader A/B/A union B/C/D/E comparison
  across both downstream tasks.
- `ecoli_iML1515_AMN_MINN_model_testing_trial.ipynb`: AMN growth branch.
- `ecoli_iML1515_MINN_AMN_model_testing_trial.ipynb`: MINN Table 4-style branch.
- `models/AMN_MINN_500k_d256_h8_l3_ff1024/`: current shared model directory.
- `data/iML1515_AMN_MINN_test_data_50000_samples.csv`: current shared test data.
- `AMN_experiment_notes.md`: AMN-only design and interpretation.
- `MINN_training_notes.md`: MINN-only training, mapping, and pFBA detail.

## Shared Data Design

The shared generator is intended to train one iML1515 FluxTransformer that can
serve both no-glucose Faure-like AMN media and glucose/oxygen-driven MINN
conditions. It is not a strict reproduction of Faure et al.

- Model: `models/iML1515.xml`.
- Objective: `BIOMASS_Ec_iML1515_core_75p37M`.
- Default solver: pFBA with `fraction_of_optimum=0.999`.
- Inputs: the 38 Faure-style exchange identities plus `EX_glc__D_e` and
  `EX_etoh_e`.
- First five context inputs: `EX_glc__D_e`, `EX_o2_e`, `EX_co2_e`,
  `EX_etoh_e`, and `EX_ac_e`.
- Regimes: `minn`, `faure`, and `mixed`, with default weights 0.50, 0.40, and
  0.10.
- All regimes use a fixed non-carbon base nutrient rate of 50.
- MINN and mixed rows independently sample integer caps: glucose 1--15 and
  oxygen 1--20 as uptake caps; CO2 0--15, ethanol 0--1, and acetate 0--3 as
  secretion caps.
- Faure rows omit glucose, vary oxygen from 1--10, retain fixed glycerol and
  amino-acid caps of 2.2, and sample one to four Faure carbon sources.
- Acetate is context-dependent: it may be a carbon-source uptake cap in Faure
  rows and a secretion cap in glucose/MINN rows.

As of 2026-08-13, the generator's working defaults are 50,000 accepted samples,
seed 9, and output prefix `iML1515_AMN_MINN_test_data`. These are test-data
defaults. For a training run, set the sample count, seed, and training-specific
prefix explicitly and record the exact command here.

## Literal A Union B Generator

`generate_ecoli_iML1515_AB_union_data.py` is the separate literal-union
baseline for the broader iML1515 sampling study. It does not contain the mixed
bridge regime from `generate_ecoli_iML1515_AMN_MINN_data.py`.

- Default dataset size: 1,000,000 accepted samples.
- Accepted regime quotas: exactly 500,000 A and 500,000 Tazza-style B rows,
  deterministically shuffled before generation.
- Default seed: 42.
- Objective: `BIOMASS_Ec_iML1515_core_75p37M`.
- Solver: pFBA with `fraction_of_optimum=0.999`.
- Input vocabulary: 41 exchanges, the exact union of the current A and
  Tazza-style B input identities.
- A oxygen: continuous 1--10; B oxygen: integer 1--20. The ranges remain
  regime-specific to preserve the component distributions.
- A rows retain base rate 10 and glycerol/amino-acid rates of 2.2. B rows retain
  base rate 50 and the widened integer Tazza-style glucose/O2/CO2/ethanol/
  acetate bounds.
- Inputs outside the selected regime are zero. The regime label is used only
  during generation and is not written as a FluxTransformer input.

No full A union B dataset or checkpoint has been generated yet.

## Current Shared Checkpoint

- Model name: `AMN_MINN_500k_d256_h8_l3_ff1024`.
- Checkpoint:
  `models/AMN_MINN_500k_d256_h8_l3_ff1024/AMN_MINN_500k_d256_h8_l3_ff1024_checkpoint.pth`.
- Architecture inferred by both notebooks: `d_model=256`, 8 heads, 3 layers,
  and `d_ff=1024`.
- Saved checkpoint epoch: 4.
- Checkpoint kind: temporary-best checkpoint without `config` or `data_info`;
  the notebooks infer architecture from the state dictionary and recover input
  metadata from the loaded CSV.
- The name indicates a 500k training run, but the exact training file and
  generator arguments are not stored in the checkpoint. Treat them as unknown
  until confirmed from an external training log or other primary artifact.

Both shared notebooks load the 50,000-sample seed-9 test dataset and assert
vocabulary size and input-token index compatibility with the checkpoint.

Generator change (2026-09): `generate_ecoli_iML1515_AMN_MINN_data.py` now
includes the fixed cobalamin exchange `EX_cbl1_e` in the base medium by
default (opt-out with `--exclude-cbl1`). Cobalamin is required by the
wild-type iML1515 objective if that variant is used and is harmless for the
core objective; it is a fixed basal nutrient, not a variable carbon source.
This makes the shared/AMN-MINN vocabulary 41 inputs by default instead of 40.
The earlier 40-input C model (`AMN_MINN_1M_*`) was generated without
cobalamin; any new shared-C generation should use the default (41 inputs), and
a data-volume comparison against the 1M C checkpoint should keep the
vocabulary consistent (regenerate C) or explicitly treat the 40/41 difference
as a controlled change.

## AMN Branch: Current Snapshot

The AMN shared-reservoir notebook predicts experimental `GR_AVG` for 110 samples
using repeated stratified 10-fold cross-validation with split seeds 10, 11, and
12. The frozen shared FluxTransformer is paired with a small trainable prior
network.

The previously saved pooled OOF metrics across split seeds were:

| Method | R2 | MAE | RMSE |
| --- | ---: | ---: | ---: |
| Shared FluxTransformer reservoir, superseded fixed-base setup | 0.8838 +/- 0.0084 | 0.022685 +/- 0.000603 | 0.028746 +/- 0.000979 |
| TabPFN | 0.7964 +/- 0.0037 | 0.030961 +/- 0.000498 | 0.038047 +/- 0.000349 |

The reservoir row was generated while the notebook converted fixed experimental
presence flags to base bounds of `10`, rather than the shared generator's `50`.
The notebook code was corrected on 2026-08-14 and its setup output was cleared.
The experimental reservoir section through its prior-net comparison must be
rerun before retaining a replacement shared-reservoir result.
The saved TabPFN comparison is independent of this fixed-medium conversion.

## MINN Branch: Current Snapshot

The MINN shared-reservoir notebook uses the restored `6b1e3bf` workflow: a
one-hidden-layer width-512 ReLU front MLP, raw Huber loss, full-vocabulary frozen
reservoir forward, `minn_fitted` targets, and `co2_etoh_ac_cap` downstream pFBA.
The Table 4-style comparison contains 29 samples and 47 mapped flux metrics.

Saved notebook outputs include:

| pFBA context | R2 | MAE | RMSE | NE |
| --- | ---: | ---: | ---: | ---: |
| Baseline measured glucose/O2 only | 0.892825 +/- 0.132254 | 0.495038 +/- 0.365933 | 0.832498 +/- 0.633807 | 0.309400 +/- 0.373326 |
| Measured glucose/O2 plus predicted CO2/ethanol/acetate caps | 0.870880 +/- 0.135621 | 0.520665 +/- 0.347119 | 0.859572 +/- 0.566047 | 0.319484 +/- 0.318528 |

The maintained MINN note records the predicted-context result as `R2=0.895539`
and `MAE=0.486514`, slightly better than baseline pFBA on those two metrics. Its
result cell is not currently saved in the notebook, so rerun and save that branch
before treating the result as independently reproducible from the notebook.

## Cross-Task Interpretation

The current checkpoint is operationally compatible with both notebook input
vocabularies. Its AMN experimental advantage must be re-established after the
fixed-base correction. The MINN result is narrower: predicted context provides
a small improvement over baseline pFBA in the maintained result, whereas the
measured-context cap variant is worse. This supports continued shared-reservoir
experiments, but does not establish that every shared-context formulation helps.

Do not present the shared generator as exact Faure media, and do not treat high
regression scores as proof of flux feasibility. Preserve full-vocabulary
reservoir forward passes in MINN because output subsetting changes the attention
token set.

## Trial Registry

| Date | Trial | Model/data | Material change | Main result | Decision |
| --- | --- | --- | --- | --- | --- |
| 2026-07/08 | Shared reference before AMN input correction | `AMN_MINN_500k_d256_h8_l3_ff1024`; shared 50k test set | One checkpoint evaluated on AMN growth and MINN Table 4-style tasks | AMN R2 0.8838 used fixed base 10; maintained MINN predicted-context R2 0.895539 | Retain MINN evidence; supersede AMN metric |
| 2026-08-14 | AMN fixed-base alignment correction | Same shared checkpoint; `ecoli_iML1515_AMN_MINN_model_testing_trial.ipynb` | Changed fixed present base inputs from 10 to the shared-generator value 50; retained 2.2 glycerol/amino caps and zero absent glucose/ethanol | Corrected AMN result pending rerun | Retain code correction; do not quote a new metric yet |
| Historical | Two-layer GELU MINN front network | Same shared reservoir family | Per-flux normalized loss and deeper front MLP | Worse than restored legacy pipeline | Rejected; retained in Git history |

Add future trials chronologically. A new model trial is incomplete until its
generator command or dataset provenance, checkpoint path, architecture, seeds,
protocol, metrics, and keep/reject decision are recorded.

## Update Checklist

- Confirm generator defaults and record the exact data-generation command.
- Record training and test file paths, sample counts, seeds, and token order.
- Record checkpoint path, architecture, epoch, loss, and training provenance.
- Update both task branches affected by the change; do not assume success on one
  task transfers to the other.
- Copy only metrics visible in saved outputs or another identified primary
  artifact, and label unavailable or stale outputs.
- Record failed trials and the reason for rejection.
- Update `AMN_experiment_notes.md` or `MINN_training_notes.md` when their
  task-specific workflows or conclusions also change.

## Combined literal-union evaluation implementation (2026-09-08)

`ecoli_iML1515_AB_union_model_testing.ipynb` and `iml1515_ab_evaluation.py`
implement a single evaluation notebook for the literal A union B reservoir.
It shares only the frozen checkpoint: AMN and each MINN context mode train
independent MLPs. No TabPFN tests are included. The A branch preserves base
10/absent cobalamin and the B branch base 50/present cobalamin. The C trial
notebooks and their generator contracts are unchanged.

Outer-test targets are excluded from tuning and epoch selection in the new
workflow. Both MINN context modes remain in the final pFBA comparison, using
observed glucose/O2 bounds and predicted secretion caps. The sampling-study
note records full scope, metric definitions and pending production provenance.
The user reports a new union model, but its checkpoint/log path is not yet
verified locally; the notebook requires explicit configuration. No new
production result or keep/reject decision follows from implementation checks.

## Union checkpoint paths configured (2026-09-10)

The combined notebook now selects
`models/AB_1M_d256_h8_l4_ff1024/AB_1M_d256_h8_l4_ff1024_checkpoint.pth`
and its sibling `AB_1M_d256_h8_l4_ff1024_training.log`. These files are now
available locally, superseding the earlier pending-path status. The log records
41 inputs, 2712 outputs, and an 800k/200k split of
`data/iML1515_AB_union_training_data_1000000_samples.csv`.
Exact command provenance and simulated-test paths remain unset in the
notebook; this path configuration does not constitute an evaluation run.

## Notebook usability and checkpoint plots (2026-09-10)

Command provenance is optional and never blocks evaluation. The training log
is optional. Full A/B simulated fidelity is off by default and skipped optional
sections no longer mark otherwise completed requested evaluations incomplete.
The notebook selects the available union test CSV for a biomass-only diagnostic;
if that optional file is absent it skips the plot and proceeds with experiments.
Checkpoint training/validation loss curves and the biomass 2x2 diagnostic reuse
the earlier AMN notebook's plotting style (sizes, colors, fonts, residual limits,
and histograms), with an A union B label. No other simulated flux plots are added.
The biomass diagnostic checks exact token order and uses full-output inference.

The notebook's final cell now displays and exports `experimental_final_summary.csv`:
biomass and pooled regression R2, MAE and RMSE for AMN, both direct MINN context
modes and available pFBA variants. AMN scores are averaged across repeat scores;
its sole measured target is biomass, so pooled and biomass scores coincide.
MINN biomass is scored across conditions; pooled scores flatten conditions and
fluxes. Direct and pFBA target counts and successful sample coverage are explicit.

## Union MINN numerical stability (2026-09-12)

A reported MINN trial stopped at gradient clipping because its gradient norm
was nonfinite with CUDA AMP enabled. The trace establishes gradient failure;
mixed-precision overflow is a suspected cause, not a reproduced diagnosis.
The union notebook now defaults to FP32. If AMP is explicitly enabled, a
nonfinite fit restarts from its original seed in FP32 before any result is
accepted. Invalid gradients never update weights. Numerical HPO failures are
recorded as failed trials and do not terminate remaining trials; if all fail,
the run stops explicitly. Fit histories record actual AMP use.

## AMN plot and metric alignment with model C (2026-09-12)

The union AMN growth-error plot now uses the fourth OOF plot in the model C
AMN trial notebook: 6x6 figure, blue #3F7BD9 points, orange #CC6E00 experimental
and prediction SD bars, red dashed identity line, 0--0.5 axes, matching fonts,
gray spines, grid, and R2 annotation. Values are computed from union results.
The plot and final summary now score per-medium mean OOF predictions, matching
model C, instead of averaging repeat scores. Metric uncertainty remains the
population SD of per-repeat scores; prediction error bars use sample SD across
repeats. Repeat scores remain available separately. This supersedes earlier
mean-of-repeat-score descriptions for the final AMN summary. Training and inner
epoch selection are unchanged; this does not make the legacy validation
protocol identical to the union protocol or copy its historical score.

## Current MINN protocol: model C-style tuning (2026-09-12)

At the user's request, this supersedes the earlier per-outer-fold HPO and
inner-epoch refit plan for MINN. Each glucose/O2 context mode now runs ONE
full-dataset five-fold HPO study (50 trials), then reuses its best parameters
for all 29 LOO fits. The sampler uses multivariate/group TPE and median pruning,
matching model C. Each LOO fit early-stops on its held-out condition and restores
the best front-MLP weights, matching the current C notebook. HPO records are
saved once per mode; fold records explicitly identify full-dataset HPO and
outer-test early stopping. Early stopping may be revised AFTER the first run.
This protocol can yield optimistic scores; it is retained for the requested
runtime and C-protocol comparison. AMN validation remains unchanged.

Downstream pFBA now uses fraction_of_optimum=1.0, matching C. Simulated-data
objective checks retain the generator's 0.999 contract. FP32 and numerical
failure handling remain enabled after the reported AMP crash. The comparison
table retains C-compatible mean-per-condition Pearson_r2, MAE and RMSE;
regression R2 and pooled scores remain separately labelled additional metrics.

Notebook compatibility fix (2026-09-13): the AMN plot and final summary cells
reload an older imported evaluation module if `summarize_amn_oof` is absent.
Existing trained results remain in memory; rerunning these cells does not
require a kernel restart or retraining. Metric formulas are unchanged.
