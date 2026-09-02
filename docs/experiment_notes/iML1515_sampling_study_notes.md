# iML1515 Sampling Study Notes

This file is the source of truth for the planned comparison of iML1515
FluxTransformer pretraining distributions across both downstream experimental
tasks:

1. AMN-style growth prediction from the Faure medium-composition data.
2. MINN-style missing-flux prediction from the Tazza/Ishii data.

The detailed sampling rationale remains in
`docs/working_notes/iML1515_sampling_plan.md`. Keep confirmed generator,
dataset, checkpoint, evaluation, result, and keep/reject decisions here.

## Research Question

How does the simulated nutrient and exchange-bound distribution used for
FluxTransformer pretraining affect transfer to AMN growth prediction and MINN
missing-flux prediction?

Every model in the series must be evaluated on both experimental tasks. A model
label describes its pretraining distribution, not the downstream task on which
it is allowed to be evaluated.

## Model and Generator Registry

| Model | Pretraining distribution | Generator | Generator status |
|---|---|---|---|
| **A** | AMN-specific | `generate_ecoli_iML1515_AMN_data.py` | Implemented |
| **B** | Tazza-style MINN-specific | `generate_ecoli_iML1515_MINN_data_tazza.py` | Implemented |
| **A ∪ B** | Balanced literal mixture of A and B | `generate_ecoli_iML1515_AB_union_data.py` | Implemented |
| **C** | Task-relevant AMN/MINN distribution with a bridge regime | `generate_ecoli_iML1515_AMN_MINN_data.py` | Implemented |
| **D** | Broad distribution with explicit A- and B-like coverage | Planned: `generate_ecoli_iML1515_broad_task_aware_data.py` | Not implemented |
| **E** | Broad task-agnostic distribution | Planned: `generate_ecoli_iML1515_broad_agnostic_data.py` | Not implemented |

`generate_ecoli_iML1515_MINN_data.py` belongs to the earlier MINN workflow. It
does not define B in this study because B uses the separate Tazza-style sampler
that independently samples glucose, oxygen, CO2, ethanol, and acetate bounds.

## Current Distribution Contracts

### A — AMN-specific

- Ten eligible AMN carbon sources; one to four active per sample.
- Continuous selected-carbon bounds from 0.05--2.2.
- Glycerol and alanine, proline, threonine, and glycine fixed at 2.2.
- Oxygen sampled continuously from 1--10.
- Current fixed base rate: 10.
- Current generator supports FBA and pFBA; use pFBA for the new comparison.

The working plan also considers a later experimental-medium-faithful A variant
without glycerol. That variant is not the current implemented A contract and
must receive a separate dataset/checkpoint identity if adopted.

### B — Tazza-style MINN-specific

- Glucose uptake bound: integer 1--15.
- Oxygen uptake bound: integer 1--20.
- CO2 secretion bound: integer 0--15.
- Ethanol secretion bound: integer 0--1.
- Acetate secretion bound: integer 0--3.
- Other base inputs fixed at 50.
- pFBA with `fraction_of_optimum=0.999`.

### A ∪ B — literal union

- Exactly balanced A and B accepted-sample quotas.
- Default size: 1,000,000 rows, comprising 500,000 A and 500,000 B rows.
- One 41-exchange input vocabulary containing the exact A/B identity union.
- A oxygen remains continuous 1--10; B oxygen remains integer 1--20.
- pFBA with `fraction_of_optimum=0.999` for both regimes.
- No mixed or intermediate rows.
- Roihu generation job: `scripts/roihu/samplejob_AB.sh`.
- Roihu resources: one CPU in `small`, 72-hour wall time, and 16 GiB memory.

The Roihu job explicitly requests one million samples and pFBA at 0.999. The
72-hour request is the current maximum for the Roihu `small` partition. The
16 GiB memory request replaces the earlier 8 GiB sampling allocation that was
insufficient during a long generation run. No A union B production job has yet
been recorded as submitted or completed.

### C — task-relevant bridge distribution

- Uses the current shared AMN/MINN generator.
- Default regime weights are 0.50 MINN, 0.40 Faure, and 0.10 mixed.
- The mixed regime introduces conditions outside the literal A ∪ B mixture.
- Uses 40 inputs because cobalamin is excluded by default.
- Default solver is pFBA with `fraction_of_optimum=0.999`.

### D and E — broad distributions

D and E are planned but not implemented. They must use the same broad exchange
vocabulary and general bound ranges. D explicitly allocates probability mass to
A- and B-like regimes, while E samples only from the task-agnostic broad rule.
The D-versus-E comparison is intended to isolate the value of explicit
task-relevant sampling density.

## Controlled Comparison Requirements

For models used in the main comparison, keep these fixed unless a named
ablation changes one of them:

- `models/iML1515.xml` reconstruction;
- `BIOMASS_Ec_iML1515_core_75p37M` objective;
- pFBA with `fraction_of_optimum=0.999`;
- full iML1515 reaction-output vocabulary and reaction order;
- FluxTransformer architecture apart from unavoidable input-token differences;
- optimizer, loss, preprocessing, split, early-stopping, and epoch settings;
- downstream AMN and MINN feature preparation and validation splits; and
- held-out evaluation rows used to compare models.

Use seed 42 for final training generation and a different seed, currently 9,
for held-out simulated data. Record every exact command, dataset path, accepted
sample count, input order, checkpoint, and training log.

The current task-specific A and B checkpoints were trained at a smaller scale
than the planned one-million-row broad models. Treat them as historical anchors
unless matched-size retraining or an explicit data-volume control is included.
Do not attribute a difference solely to sampling breadth when training-set size
also differs.

## Evaluation Plan

### Simulated-data fidelity

Evaluate every trained model on independent held-out simulated datasets for:

1. A-style conditions;
2. B-style conditions;
3. C-style intermediate conditions; and
4. broad D/E conditions once those samplers exist.

Report pooled and per-flux R2, MAE, and RMSE. Also inspect biomass, difficult
reactions, activity frequency, mass-balance residuals, bound violations, and
objective consistency where relevant.

### AMN growth prediction

Use the maintained AMN pipeline and the 110 Faure experimental media. Freeze
each pretrained FluxTransformer and train the same front model under identical
cross-validation splits and tuning rules. Report pooled out-of-fold R2, MAE,
and RMSE with repeat variability. Keep the TabPFN and other retained baselines
fixed across model comparisons.

### MINN missing-flux prediction

Use the maintained MINN experimental pipeline with identical source-to-token
mapping, sign handling, context mode, pFBA constraints, validation splits, and
target file for every pretrained model. The primary cross-task comparison is
the current Table 4-style FluxTransformer-to-pFBA workflow. The separate Table
2 direct-flux benchmark may be reported as a secondary analysis, but every
model must receive the same raw glucose and oxygen inputs.

Report R2, MAE, RMSE, normalized error, feasibility counts, and per-sample
constraint-binding diagnostics. Keep predicted context values distinct from
hard pFBA bounds.

## Required Comparisons

The final result matrix should contain every available model on both tasks:

| Pretrained model | AMN growth | MINN missing flux | A simulated | B simulated | Intermediate | Broad |
|---|---|---|---|---|---|---|
| A | Pending | Pending | Pending | Pending | Pending | Pending |
| B | Pending | Pending | Pending | Pending | Pending | Pending |
| A ∪ B | Pending | Pending | Pending | Pending | Pending | Pending |
| C | Pending | Pending | Pending | Pending | Pending | Pending |
| D | Pending | Pending | Pending | Pending | Pending | Pending |
| E | Pending | Pending | Pending | Pending | Pending | Pending |

Do not fill this table from unmatched historical runs. Add results only after
the dataset, checkpoint, downstream inputs, and evaluation protocol are
confirmed.

## Trial Registry

Record each trained sampling-study model here.

| Model | Training data | Checkpoint | Changed variable | Validation protocol | Main results | Decision |
|---|---|---|---|---|---|---|
| A | Pending | Pending | Specialized A distribution | Pending | Pending | Pending |
| B | Pending | Pending | Specialized Tazza-B distribution | Pending | Pending | Pending |
| A ∪ B | Pending | Pending | Balanced literal mixture | Pending | Pending | Pending |
| C | Pending | Pending | Added task-relevant bridge coverage | Pending | Pending | Pending |
| D | Pending | Pending | Broad sampling with explicit task coverage | Pending | Pending | Pending |
| E | Pending | Pending | Broad task-agnostic sampling | Pending | Pending | Pending |

## Coordination

Update this note whenever an A/B/A ∪ B/C/D/E generator, dataset, input order,
checkpoint, training configuration, downstream evaluation, result, or
interpretation changes. Continue to update `AMN_experiment_notes.md` and
`MINN_training_notes.md` for task-specific implementation details, and
`AMN_MINN_shared_reservoir_notes.md` for the current shared C workflow.
