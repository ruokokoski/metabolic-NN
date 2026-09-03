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

Model A keeps glycerol fixed at 2.2, matching the current implemented A
generator and Faure's simulated reservoir. A glycerol-absent A variant is not
planned.

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
- Uses 41 inputs because cobalamin (`EX_cbl1_e`) is now included by default
  (opt-out with `--exclude-cbl1`). Cobalamin is a fixed basal nutrient, not a
  variable carbon source. The earlier 40-input C checkpoint and runs used
  `--exclude-cbl1` semantics; regenerate C data before comparing C against
  other 1M-row models.
- Default solver is pFBA with `fraction_of_optimum=0.999`.

### D and E — broad distributions

D and E are planned but not implemented. They must use the same shared
selectable organic-source pool and the same general broad distribution P_G, so
the only difference is that D explicitly allocates probability mass to exact
A- and B-style regimes, while E samples only from the task-agnostic broad rule.
The D-versus-E comparison is intended to isolate the value of explicit
task-aware density allocation versus task-agnostic sampling.

The concrete D/E nutrient pool is recorded in
`docs/working_notes/iML1515_sampling_plan.md` under "D/E nutrient pool --
concrete design". In short:

- Shared selectable organic-source pool: the same 31-source pool for both D
  and E, including all ten AMN carbon sources, glucose, glycerol (variable),
  the four AMN amino acids, common side-stream sugars/sugar alcohols, organic
  acids and fermentation products.
- D also injects exact A- and B-style regimes; E uses only the general broad
  distribution P_G. The D-versus-E comparison isolates task-aware density
  allocation from task-agnostic sampling.
- General G/E uptake: log-uniform over 0.05--5.0 (tentative ceiling),
  broader than the A-specific 0.05--2.2 so the broad models cover higher-flux
  side-stream conditions.
- Oxygen is separately variable (not counted toward the active-source count).
- In the shared P_G, the four AMN amino acids are ordinary members of the
  selectable pool for both D and E; D's explicit P_A component supplies dense
  all-four-present AMN coverage.
- The required "at least both AMN and MINN" condition is met by keeping the
  exact A and B regime contracts as the A/B components of D.

| Model | Variable carbon sources | Fixed exchanges | Carbon uptake range | Active count |
|---|---|---|---|---|
| **A** | 10 AMN carbons (ribose, maltose, melibiose, trehalose, fructose, galactose, acetate, D-lactate, succinate, pyruvate) | AMN base (22) + fixed glycerol + four fixed amino acids; O2 variable 1--10 | 0.05--2.2 | 1--4 |
| **B** | Glucose + variable CO2/ethanol/acetate secretion | MINN base (23, incl. cobalamin), base 50 | Glucose 1--15; O2 1--20; secretion caps CO2 0--15, ethanol 0--1, acetate 0--3 | 1 |
| **A ∪ B** | Union: 10 AMN carbons + glucose (per regime) | A base or B base per regime | A 0.05--2.2; B integer bounds | 1--4 or 1 |
| **C** | 10 AMN carbons + glucose + amino acids + CO2/ethanol/acetate | Shared base (41 inputs, fixed cobalamin included) | A-like and B-like ranges | mixed |
| **D** | Shared 31-source general pool (glycerol variable) + A/B task regimes | 23 fixed base (incl. cobalamin); O2 variable 1--10 | G log-uniform 0.05--5.0; A/B task ranges | G 1--8, E[K]=3 |
| **E** | Same shared 31-source general pool, no task regimes | Same 23 fixed base (incl. cobalamin); O2 variable | log-uniform 0.05--5.0 | 1--8, E[K]=3 |

The 181 growth-capable organic-exchange pool was computed on
`models/iML1515.xml` for the D-like base medium, close to the 185 carbon
sources reported in `ecoli_iML1515_exploration.ipynb`. The shared 31-source
D/E pool is the frozen reviewed subset chosen from this larger list.

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

Trained checkpoints already exist for A, B, and C, each trained on one
million rows (800k train / 200k test):
- A: `AMN_1M_d256_h8_l4_ff1024` on
  `data/iML1515_AMN_training_data_1000000_samples.csv`;
- B: `MINN_1M_d256_h8_l4_ff1024` on
  `data/iML1515_MINN_training_data_1000000_samples.csv`;
- C: `AMN_MINN_1M_d256_h8_l4_ff1024` on
  `data/iML1515_AMN_MINN_training_data_1000000_samples.csv`.

These 1M-row models match the planned scale of the broad D/E models, so a data
volume control is not needed for A/B/C versus D/E comparisons on that axis.
Treat earlier 500k-row variants (e.g. `AMN_500k_*`, `iML1515_MINN_500k_*`) as
historical anchors, not as the current A/B/C checkpoints.

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
| A | `data/iML1515_AMN_training_data_1000000_samples.csv` | `models/AMN_1M_d256_h8_l4_ff1024/AMN_1M_d256_h8_l4_ff1024.pth` | Specialized A distribution (1M rows) | 800k/200k; Huber; d256 h8 l4; best test loss 0.006774 at epoch 12 | Pending | Keep |
| B | `data/iML1515_MINN_training_data_1000000_samples.csv` | `models/MINN_1M_d256_h8_l4_ff1024/MINN_1M_d256_h8_l4_ff1024.pth` | Specialized Tazza-B distribution (1M rows) | 800k/200k; Huber; d256 h8 l4; best test loss 0.000042 at epoch 10 | Pending | Keep |
| A ∪ B | Pending | Pending | Balanced literal mixture | Pending | Pending | Pending |
| C | `data/iML1515_AMN_MINN_training_data_1000000_samples.csv` | `models/AMN_MINN_1M_d256_h8_l4_ff1024/AMN_MINN_1M_d256_h8_l4_ff1024.pth` | Added task-relevant bridge coverage (1M rows) | 800k/200k; Huber; d256 h8 l4; best test loss 0.000202 at epoch 13 | Pending | Keep |
| D | Pending | Pending | Broad sampling with explicit task coverage | Pending | Pending | Pending |
| E | Pending | Pending | Broad task-agnostic sampling | Pending | Pending | Pending |

## Coordination

Update this note whenever an A/B/A ∪ B/C/D/E generator, dataset, input order,
checkpoint, training configuration, downstream evaluation, result, or
interpretation changes. Continue to update `AMN_experiment_notes.md` and
`MINN_training_notes.md` for task-specific implementation details, and
`AMN_MINN_shared_reservoir_notes.md` for the current shared C workflow.
