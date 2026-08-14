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
