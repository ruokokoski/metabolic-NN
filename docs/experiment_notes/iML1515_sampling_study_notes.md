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
| **D** | Broad distribution with explicit A- and B-like coverage | `generate_ecoli_iML1515_D_data.py` | Implemented |
| **E** | Broad task-agnostic distribution | `generate_ecoli_iML1515_E_data.py` | Implemented |

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
- Current generator defaults to plain FBA and supports optional pFBA; use
  pFBA with `fraction_of_optimum=0.999` for the new comparison and record the
  actual generation settings for existing data/checkpoints.

Model A keeps glycerol fixed at 2.2, matching the current implemented A
generator and Faure's simulated reservoir. A glycerol-absent A variant is not
planned.

Author comparison verified 2026-10-06: **A independently draws one-to-four
sources from the ten-source list**, whereas the author iML1515 simulated
reservoir uses only the 110 experimentally tested presence patterns
(10/20/40/40 patterns with one/two/three/four variable sources), with 100 random
cap draws per pattern. These are 11,000 simulated pFBA rows, not additional
measured conditions. Author variable-source caps follow a cardinality-scaled
99-level grid, all fixed caps are 2.2, and the pFBA call leaves the optimum
fraction at COBRA's default 1.0. Consequently, A is an adaptation in subset
selection, bounds, and solver settings; matching glycerol alone does not
reproduce that reservoir. The same unrestricted-subset distinction applies to
the A component of A ∪ B, the `faure` regime of C, and A-style regimes of D.
The author's P. putida experiments use plain FBA, so "all AMN experiments
used pFBA" is not supported. See
[AMN experiment notes](AMN_experiment_notes.md#verified-author-sampling-and-solver-scope-2026-10-06)
and the [repository inspection](../reference/amn_repository_notes.md).

### B — Tazza-style MINN-specific

- Glucose uptake bound: integer 1--15.
- Oxygen uptake bound: integer 1--20.
- CO2 secretion bound: integer 0--15.
- Ethanol secretion bound: integer 0--1.
- Acetate secretion bound: integer 0--3.
- Other base inputs fixed at 50.
- pFBA with `fraction_of_optimum=0.999`.

Author comparison verified 2026-10-06: the
[MINN source audit](../reference/minn_repository_notes.md) finds a
glucose-minimal 587-reaction iAF1260 reservoir model with 16 basal imports,
no AMN glycerol/amino-acid supplements, and CO2/ethanol/acetate as exports.
The paper describes 2,000 randomized five-channel exchange simulations, but
the generator and pretraining data are not released. B's integer sampling,
base rate 50, iML1515 GEM, and pFBA at 0.999 are adaptations, not established
original sampling settings. The widened ranges cover the observed raw exchange
values but exclude fitted O2/CO2 maxima of 26.2417/19.6169; the released author
reservoir loader uses fitted uptake inputs and five fitted exchange targets.
Keep this support distinction explicit when interpreting B and MINN regimes
within A ∪ B, C, and D.

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

D and E are implemented as standalone generator scripts for direct Roihu
deployment. They use the same shared selectable organic-source pool and the
same general broad distribution P_G, so
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

Implementation defaults and schema:

- D entry point: `generate_ecoli_iML1515_D_data.py`.
- E entry point: `generate_ecoli_iML1515_E_data.py`.
- Default output prefixes: `iML1515_D_training_data` and
  `iML1515_E_training_data`.
- Each generator contains its complete sampling implementation. Keep their P_G
  constants, input order, and shared regime behavior synchronized when editing
  either file.
- Both default to 1,000,000 accepted samples, seed 42, and pFBA with
  `fraction_of_optimum=0.999`.
- Both use the same ordered 55-exchange input vocabulary: the five MINN context
  exchanges first, followed by the remaining fixed-base and selectable-organic
  exchanges without duplicates.
- P_G uses a fixed-base rate of 50, continuous oxygen 1--10, and the documented
  truncated-geometric active-source distribution with `E[K]=3`.
- D uses `--task-fraction 0.2` as a configurable pilot default and schedules
  exact accepted-row quotas. At 1M rows this gives 100k A, 100k B, and 800k G.
  The final task fraction still requires the planned pilot comparison.
- E schedules all accepted rows from P_G.
- Roihu generation jobs: `scripts/roihu/samplejob_D.sh` and
  `scripts/roihu/samplejob_E.sh`. Both use the `small` partition, one CPU,
  72-hour wall time, and 16 GiB memory, matching the A union B job setup.
- The initial thin entry points imported `iml1515_broad_sampling.py` and failed
  on Roihu when that helper was absent from `CODEDIR`. The generators are now
  standalone so each sampling job requires only its corresponding generator.
- No D or E production dataset has yet been recorded as generated.

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
fixed across model comparisons. The standalone AMN notebook now explicitly
selects TabPFN-3.5 for future baseline runs. Historical scores are not 3.5
results; rerun before using the updated baseline in model comparisons.

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

**Metric correction, 2026-09-23:** all MINN comparisons of sample spaces A,
B, A-union-B, C, D and E must use regression R2 (`sklearn.metrics.r2_score`,
1 - SSE/SST), preserving negative scores. For Table 4-style baseline pFBA
versus measured/predicted-context reservoir plus CO2/ethanol/acetate caps,
compute across the 47 mapped fluxes per held-out condition, then report mean
and population SD across the 29 conditions. Pooled regression R2 is additional
and must be labelled separately. Squared Pearson correlation is not Table 4
regression R2 and must not be labelled R2 in these comparisons. Undefined
scores and failed conditions require explicit coverage reporting.

This metric contract follows Goncalves Section 2.5, Equation 4 and Tazza's
Section 2.4 statement that it uses the same metrics. Their released
`omics2flux` and `MINN` code instead computes squared Pearson correlation
with `linregress(...).r**2`; the papers do not state that definition. See
`MINN_training_notes.md` for the source pages, fitted-file verification, and
the 0.658478 regression versus 0.892825 squared-Pearson baseline comparison.

The 2026-10-06 [author audit](../reference/minn_repository_notes.md) further
reproduces Tazza's copied **Table 2** pFBA baseline from the byte-identical
Goncalves artifact: Pearson r2 0.822702 +/- 0.156467 versus regression R2
0.778061 +/- 0.228708, on 45 original signed fluxes. Table 4 is a separate
47-flux comparison; its final author pFBA code and predictions are missing.
The released reservoir export names two columns for five latent controls,
while the paper describes three extra constraints. Thus Pearson is the
verified released metric and the supported inference for its Table 4 lineage;
the exact two published Table 4 rows remain unverified. The local regression-R2
contract is preserved, and local scores must not be labelled exact author
reproductions merely because their averages are close.

This supersedes the historical Pearson-based Table 4 conventions recorded
later in this note. The standalone `ecoli_iML1515_MINN_model_testing.ipynb`
is corrected first; the shared evaluator and AB/C/D/E/shared-trial legacy
summaries remain pending, not corrected by this documentation change.
See `MINN_training_notes.md` for the verified regression baseline and exact
implementation scope. The user will rerun the standalone reservoir variants.

**Standalone B evaluation update, 2026-09-27:**
`ecoli_iML1515_MINN_model_testing.ipynb` now prepares a five-row pFBA
comparison. The original MINN-fitted file supplies targets and measured
glucose/oxygen inputs for pFBA and measured/predicted-context FT+pFBA with
CO2, ethanol, and acetate caps. A separate iML1515-fitted file supplies
targets and inputs for pFBA and a newly trained measured-context FT+pFBA
run with the same three caps. The final table reports per-condition mean
and population SD for regression R2 and squared Pearson r under separate
labels, plus solver coverage. Ethanol/acetate-only evaluations are removed.
The new front-MLP training and full comparison have not been executed; no
new reservoir scores are available.

The baseline fitted-file gap is dominated by `WT_0.1h-1`: original MINN-fit
R2 -5.608750 versus iML1515-fit R2 0.970738. It accounts for 87.8% of the
0.258449 mean R2 difference; excluding it leaves means 0.882308 and
0.915005. The original fitted file matches the upstream MINN copy exactly,
while the iML1515 row reproduces from the local fitter. Tazza describes a
Euclidean FBA fit, whereas the local iML1515 fitter uses weighted L1 deviation.
The original fit gives this low-growth condition glucose uptake 5.862 versus
1.34 raw and 2.028 iML1515-fitted; pFBA with uptake caps then maximizes
biomass to 0.320904 against a 0.1000 target. This is a target/constraint
compatibility sensitivity, not evidence of a CSV copy error or a corrected
Tazza score. See `MINN_training_notes.md` for the per-sample and counterfactual
solver checks.

### Full-output checkpoint metrics in combined notebooks (2026-09-28)

`ecoli_iML1515_AB_union_model_testing.ipynb` and the C, D, and E model-testing
notebooks now compute overall regression R2, MAE, and RMSE on their own
independent 50,000-row simulated test CSVs, immediately after the biomass
diagnostic plots. Each metric pools all rows and all 2,712 output fluxes, as
in the standalone AMN notebook. The new cells stream batches, verify exact
checkpoint input/output order and training/test separation, use full-output
inference, and restore the original device after any CPU fallback. These
metrics describe simulated surrogate fidelity; they are separate from the
experimental AMN/MINN results and from optional `RUN_SIMULATED` A/B
domain-specific fidelity analysis, whose test paths remain unset.

The four real checkpoints and corresponding CSV schemas were checked. A
two-row smoke test for each model matched scikit-learn's flattened R2, MAE,
and RMSE. The complete 50,000-row metric cells have not been executed, so no
new full-test scores are claimed.

### E full-output distribution-shift tests (2026-10-01)

Immediately after E's in-distribution overall-metrics cell,
`ecoli_iML1515_E_model_testing.ipynb` now evaluates the same pooled regression
R2, MAE and RMSE on the complete independent A and B test CSVs selected by
the standalone AMN and MINN notebooks. A uses
`data/iML1515_AMN_test_data_50000_samples.csv` (38 nutrient inputs); B uses
`data/iML1515_MINN_tazza_test_data_50000_samples.csv` (27 inputs). Both have
the E checkpoint's 2,712 flux targets in the same order. The source generator
input order is checked, and each nutrient bound is mapped by exchange name
into E's 55-input vocabulary. E inputs absent from the source file are zero,
including cobalamin for A; B supplies its own cobalamin value. Full-output
inference and all 50,000 rows per file preserve the in-distribution metric
aggregation; the standalone notebooks' later 80/20 diagnostic split is not
applied. Results and the missing-input lists are exported separately.
This probes performance on the exact saved source datasets. The completed
50,000-row run in
`pics/E_1M_d256_h8_l4_ff1024/E_evaluation/20261001T052716734294Z/overall_distribution_shift_metrics.csv`
reports pooled regression R2 of `0.554681` for A and `0.665759` for B
(MAE `0.135508`/`0.087076`, RMSE `0.697911`/`0.898517`). E's own test
set reports R2 `0.998954`, MAE `0.003638`, RMSE `0.025395` in the saved
notebook. An independent two-row check per dataset matched the notebook's
input tensor against exchange-name mapping and its R2, MAE and RMSE against
scikit-learn on the same full-output predictions. The saved A/B CSVs contain
all 2,712 outputs in E's checkpoint order.

E's larger input vocabulary does not make its general training distribution
a superset of A and B media. A's 17 absent E inputs are closed by the A
generator; zero-filling them preserves the source medium, including absent
cobalamin. A fixes basal bounds at `10` (E uses `50`), uses CO2 as an uptake
bound of `10` (E uses a secretion cap of `50`), and has fixed glycerol and
four amino-acid uptake bounds of `2.2`. The first saved A target row matches
ordinary FBA, whereas E's generator uses pFBA at `0.999` of optimum. B
already supplies cobalamin at `50`; its 28 absent E organic inputs have no
uptake under the iML1515 SBML defaults, so zero-filling is appropriate.
Across all 50,000 B rows, `50.1%` of oxygen bounds exceed E's `10` maximum
and `66.8%` of glucose bounds exceed E's `5` maximum. B samples CO2
secretion caps from `0` to `15` (E fixes `50`), while positive ethanol and
acetate values denote secretion caps in B but uptake bounds in E. These
constraint and solver differences confound attribution of the score gap to
nutrient coverage alone; the checks found no metric or input-order bug.

### Planned distribution alignment for new models

- Expand the next E general-distribution generator and training set so glucose
  uptake covers B's `1`-to-`15` range and oxygen uptake covers B's `1`-to-`20`
  range (current E maxima are `5` and `10`). Record the new sampling law and
  generate a fresh independent test set before training and evaluating a new
  E checkpoint. Extending these ranges alone will not reproduce B's CO2,
  ethanol and acetate secretion-cap semantics.
- Generate future AMN-style simulated data with pFBA at the shared `0.999`
  fraction of optimum and harmonize fixed basal uptake bounds to `50`, as in
  the B and E general regimes, instead of the current AMN defaults of FBA and
  `10`. Specify CO2's bound direction separately when doing this: current A
  permits CO2 uptake, whereas E's general regime caps CO2 secretion. Keep
  today's A data and checkpoint labeled with their original policy; regenerate
  data and retrain before reporting results under the aligned policy.

The standalone AMN notebook now also tests its 38-input checkpoint against
the complete B and E test CSVs with full-output pooled R2, MAE and RMSE.
Input names and target order are checked against the source generators and
checkpoint. All 27 B and 55 E source inputs map to reaction tokens in AMN's
full 2,712-token vocabulary and are injected in full-output inference. Three
B inputs (glucose, ethanol and cobalamin) and 17 E inputs fall outside AMN's
38 declared training-input tokens. The notebook records these additional
inputs, their activity, and trained inputs zero-filled when absent from the
source. Scores will test extrapolation to unseen context positions as well as
the source distributions' bound and solver shifts; results await the full run.

The standalone AMN notebook also has a Faure Figure S9-style downstream test:
retain validation-fold front-network `Vin`, feed those positive uptake caps to
ordinary iML1515 FBA, and compare the resulting growth predictions with the
frozen FluxTransformer's predictions on the same CV splits. The front model
must be rerun to produce fold-level `Vin`; no full FBA result is reported yet.
The existing three-repeat outer-fold epoch selection remains the comparison
protocol and limits strict held-out interpretation.

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

## Combined A union B evaluation notebook (2026-09-08)

`ecoli_iML1515_AB_union_model_testing.ipynb` now implements one notebook for
both experimental tasks, backed by `iml1515_ab_evaluation.py`. It shares the
frozen union reservoir and trains independent AMN, MINN measured-context, and
MINN predicted-context MLPs. No TabPFN tests are included. A-style neural inputs
use base 10 and absent cobalamin; B-style inputs use base 50 with cobalamin.

AMN uses repeated stratified ten-fold CV with training-only inner epoch
selection and outer-training refits. MINN uses LOO with five-fold HPO inside
each outer training set; the median winning-trial inner epoch controls the
refit. Neither task uses outer-test targets for early stopping. Historical
A/B/C scores require matching reruns under this protocol. Both MINN modes
retain observed glucose/O2 pFBA caps and OOF secretion caps. The pFBA fraction
is 0.999; the maintained Table 4 SBML background medium is retained.

The notebook exports A/B simulated fidelity, OOF metrics, fold provenance,
context values, solver coverage and cap-binding diagnostics. Regression R2
and squared Pearson correlation are separate. Optional target-file/cap-set
sensitivities are supported; C/broad fidelity remains deferred.

The local union checkpoint path and production provenance remain unconfirmed.
The user reports creating the model; configure its actual checkpoint and log
in the notebook. Implementation validation uses a small untrained reservoir
and representative real-data/pFBA checks, not production performance results.
The union result/decision registry remains pending.

Verification: ten focused tests passed, including miniature-reservoir AMN/CV
and MINN/HPO execution, real-data mappings, frozen gradients, checkpoint
rejection, simulated A/B bound/objective reconstruction, and representative
iML1515 pFBA solves. Notebook JSON and code-cell syntax were validated.
Clean-kernel setup reached the expected missing-artifact preflight stop; the
production notebook has not been executed through training.

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

Final summary update (2026-09-14): the last union-notebook cell and its CSV
now contain only pooled R2, MAE and RMSE, with separate biomass metric columns
removed. The pooled target sets and aggregation are unchanged.

## Combined model C notebook (2026-09-14)

`ecoli_iML1515_C_model_testing.ipynb` uses shared `iml1515_evaluation.py`;
`iml1515_ab_evaluation.py` remains a compatibility import. Explicit generator
contracts support AB, C (40/41 inputs), D and E. The configured
`AMN_MINN_1M_d256_h8_l4_ff1024` checkpoint has 40 inputs (no cobalamin).
`data/iML1515_AMN_MINN_test_data_50000_samples.csv` matches its ordered schema
and supplies the default independent biomass diagnostic.

C AMN uses basal inputs including CO2 at 50, glycerol/amino acids at 2.2;
MINN common basal inputs are 50. Cobalamin is 50 only when included in the
checkpoint inputs. AB/D AMN retain basal 10; E uses 50. E experimental-media
mapping is an evaluation choice, not evidence of a matched training regime.
D/E checkpoint execution remains unverified.

Training protocols, plot styles and pooled-only final metrics remain unchanged.
MINN tunes once per context mode; LOO held-out-condition early stopping may
change after the first run. No TabPFN tests. Copied outputs are cleared.
Optional A/B cross-domain physics uses A/B source bounds; C intermediate and
D/E broad physics remain deferred. No new scores or keep/reject decision yet.

## Combined C AMN protocol restoration (2026-09-16)

The combined C notebook now explicitly uses `legacy_outer_early_stopping`
to reproduce `ecoli_iML1515_AMN_MINN_model_testing_trial.ipynb` with
`AMN_MINN_1M_d256_h8_l4_ff1024` and `AMN_data/iML1515_EXP.csv` (110 media).
It restores checkpoint-order variable features, no gradient clipping,
training/validation batches 1/2, and best-checkpoint selection on the scored
outer fold, without inner splitting or refitting. Seeds 10/11/12, training
seed 10, width 512, zero dropout, AdamW lr/weight decay 0.001, Huber delta
0.03, 100 maximum epochs and patience 15 match the original. Both score
per-medium mean repeated predictions. MINN and the default inner-refit AMN
protocol for other callers are unchanged. Fold exports identify the actual
selection protocol and validation rows.

Original saved pooled R2: 0.8723; superseded combined result: 0.812463.
Affected combined outputs were cleared. No replacement score is claimed
until the full AMN run completes. Held-out targets select epochs in the
restored protocol, so its OOF score is not an untouched-test estimate.
Decision: retain the original protocol for the requested reproduction.

## Combined C glycolysis t-SNE plots (2026-09-16)

The combined C notebook now includes reaction-colored and nutrient-context
rainbow glycolysis plots, using the original AMN_MINN notebook's plotting
function and 14-reaction list. Both take the first 4,000 independent simulated
test CSV contexts, perplexity 40, seed 10, batch 128, and final-layer embeddings
from the original glycolysis-plus-injected-input subset forward. They retain
reaction-center annotations, the reaction palette and reversed rainbow map.
This is a qualitative subset visualization, not full-vocabulary evaluation.
Both PNGs save under `pic_dir` with model-specific names; `tsne_settings.json`
records context rows, source hash and extraction settings. The context selection
is explicit and does not depend on the legacy notebook's `X_test` split.

## Combined C MINN original-protocol restoration (2026-09-16)

The combined C notebook now opts into `legacy_protocol` for MINN to reproduce
`ecoli_iML1515_MINN_AMN_model_testing_trial.ipynb`. It retains the same 1M C
checkpoint, original MINN-fitted targets (29 conditions, 141 features, 42 neural
targets), and 47-flux pFBA evaluation. Basal cobalamin is injected at 50 through
its output token even though the historical checkpoint has only 40 inputs.
This is an explicit reproduction exception, not a change to the training schema.

The legacy trainer seeds once before measured-mode HPO and preserves RNG state
through both studies and LOO fits. It uses CUDA AMP, batch 2 on CUDA/5 on CPU,
and the original duplicate front forward, validation DataLoader passes, AdamW,
raw Huber loss, clipping, warmup/cosine schedule and best-checkpoint restoration.
Global HPO and held-out-condition early stopping remain as in the original.
Default evaluator callers retain their existing FP32/fold-seeded behavior.

pFBA makes a fresh SBML-model copy per condition as in the original. Its original
final comparison is mean/population-SD per condition of squared Pearson r,
MAE, RMSE and NE; pooled regression scores remain separately labelled.
Affected MINN outputs are cleared; no reproduced full-run score is claimed.

Verification: short CPU and CUDA fits for both context modes match the original
notebook function in weights, predictions, context values and final RNG states.
The evaluator test suite has 24 passing tests and one existing unrelated failure
requiring the AB-union notebook to have no saved outputs. Combined notebook
schema/syntax and the final diff pass validation. Full HPO/LOO was not rerun.

The restored pFBA baseline was rerun successfully for all 29 conditions.
Its original-style mean/SD reproduces the original saved baseline to six
decimals: R2 0.892825/0.132254, MAE 0.495038/0.365933,
RMSE 0.832498/0.633807, and NE 0.309400/0.373326.

## AB-union alignment with corrected C evaluation (2026-09-17)

`ecoli_iML1515_AB_union_model_testing.ipynb` now follows the corrected C
notebook cell sequence and shared evaluation implementation. It retains
`AB_1M_d256_h8_l4_ff1024`, the independent
`iML1515_AB_union_test_data_50000_samples.csv`, explicit `AB_union` schema,
and its own artifact directory. AMN retains generator-aligned basal 10,
glycerol/amino acids 2.2 and absent cobalamin/glucose/ethanol zero; MINN
retains basal 50 including its declared cobalamin input.

AMN now uses checkpoint-order features, outer-fold best-checkpoint selection,
no inner refit, no clipping, and training/validation batches 1/2, matching C.
MINN uses the original-protocol trainer, CUDA AMP (CPU FP32), batches 2/5,
one seed before measured-mode HPO, continued RNG state through predicted-mode
HPO/LOO, and fresh pFBA model copies per condition. At this stage the final
table included the original per-condition Pearson-r-squared/MAE/RMSE/NE mean and population SD;
pooled regression diagnostics remain separately labelled as in C.

Both glycolysis post-layer t-SNE plots are present: reaction colors and
nutrient-context rainbow, using 4,000 AB-union test contexts, perplexity 40,
seed 10, original styles and recorded extraction settings. Old AB-union outputs
are cleared because their training protocols differ. No C scores are copied.
Full AB-union HPO/LOO results require rerunning; held-out epoch selection
and global HPO retain the same interpretation limits as the reference C run.

Verification: compared all 40 cells against C: 31 have identical source;
the other nine contain reviewed model identity/path or media/provenance text
changes. All executable cells match after normalizing only model-specific
strings, enforced by a regression test. All 27 evaluator tests pass. Notebook
schema and syntax pass. A clean-scope smoke execution loads the real AB
checkpoint and CSV, verifies both media loaders, runs all three front-model
forward paths, and renders both t-SNE PNGs with 16 contexts. Full HPO/LOO and
the default 4,000-context t-SNE runs were not executed. C is unchanged.

AB-union pFBA comparison update (2026-09-28): the notebook's last cell now
reports five rows: original MINN fit pFBA, measured and predicted three-cap
FT+pFBA; and iML1515 fit pFBA and measured three-cap FT+pFBA. The iML1515
branch loads `fluxomics_iML1515_minn_like_fit.csv` as both uptake input and
47-flux target, checks experiment and mapping order against the original file,
and retrains its measured-context front MLP with the same HPO/LOO settings.
The original file remains the primary target. For each method, regression R2,
Pearson r2, MAE, RMSE and NE are computed across the 47 fluxes per successful
condition, then summarized as mean and population SD (`ddof=0`). Negative
regression R2 is retained; undefined R2/correlation and solver coverage are
shown. The old Pearson-as-R2 per-condition table is removed; the separate
pooled regression diagnostics remain. Both 29-condition pFBA baselines were
recomputed and match the standalone MINN notebook's metric values. The full
AB-union HPO/LOO run and new FT+pFBA scores remain unexecuted.

The same dual-fit pFBA branch and five-row final table were added to the C,
D and E evaluation notebooks on 2026-09-28. Each keeps its own checkpoint,
model-family input contract and artifact directory; all use the original MINN
fit for the three primary methods and retrain measured context on the
iML1515 fit for the two sensitivity rows. Both fitted files align on 29
experiments and 47 mapped pFBA fluxes for C's legacy 40-input checkpoint and
the 55-input D/E checkpoints. Notebook schema and code syntax were checked;
the expensive C/D/E HPO/LOO evaluations were left for future runs.

## Model D notebook identity correction (2026-09-21)

`ecoli_iML1515_D_model_testing.ipynb` now selects family D,
`models/D_1M_d256_h8_l4_ff1024/D_1M_d256_h8_l4_ff1024_checkpoint.pth`,
its matching training log, and `data/iML1515_D_test_data_50000_samples.csv`.
Artifact paths and manifest code hashes identify D. Copied C-specific text,
40-input claims and saved outputs are removed. D has 55 inputs; experimental
AMN uses its A-regime basal 10 with absent cobalamin, while MINN uses B-regime
basal 50 including cobalamin. Unused broad carbon channels are zero in these
task evaluations. Training, metric formulas and t-SNE procedures are unchanged.

Verified: real D checkpoint loads with 55 inputs and 2712 outputs; test CSV
input/output order matches exactly. Full-vocabulary test inference and AMN,
measured-MINN and predicted-MINN forwards are finite. Notebook JSON/schema,
code syntax and stale-C-reference checks pass. Full evaluation was not rerun.

## Model E notebook identity correction (2026-09-22)

`ecoli_iML1515_E_model_testing.ipynb` selects family E,
`models/E_1M_d256_h8_l4_ff1024/E_1M_d256_h8_l4_ff1024_checkpoint.pth`,
its matching training log, and `data/iML1515_E_test_data_50000_samples.csv`.
The biomass diagnostic and both t-SNE plots use that final test path. Artifact
paths, manifest code hashes and descriptions now identify E; copied D outputs
are cleared. E has 55 inputs. Its experimental AMN mapping uses basal inputs
including CO2 and cobalamin at 50, fixed glycerol/amino acids at 2.2, and other
unused channels at zero. MINN uses basal inputs including cobalamin at 50.
These describe experimental task mappings, not D's A/B training regimes.
Training, metrics and t-SNE procedures remain unchanged from the copied D notebook.

Verified: notebook schema and code syntax; all 40 cells compared against D,
with executable differences limited to model identity/path strings. The real
E checkpoint loads with 55 inputs and 2712 outputs. AMN, measured-MINN and
predicted-MINN prediction/context forwards are finite. The final E test CSV
is still being generated, so its schema, simulated inference and t-SNE plots
remain unverified. Full experimental evaluation was not rerun.


## AMN controlled diagnostic sweep (2026-09-24)

`generate_AMN_sweep.py` adds a standalone 100 x 100 fructose/oxygen grid
using the current A/AMN medium and exact 38-input/2,712-output schema.
It defaults to FBA as the authoritative AMN generator does; the study's
pFBA comparison policy is unchanged. This is diagnostic data, not a change
to pretraining. Separate metadata preserves grid coordinates without
changing model tokens. A 3 x 3 smoke run is optimal throughout; the full
grid and embedding analysis remain pending. See `AMN_experiment_notes.md`
for the complete sweep and failure-handling contract.

AMN visualization update (2026-09-25): glycolysis and PPP now also provide
a separate joint embedding of the 10,000-condition sweep with 10,000 seeded
random independent AMN test conditions. The original sweep-only figures remain
unchanged. This is qualitative context coverage, not a new predictive score;
see `AMN_experiment_notes.md` for sampling and plotting provenance.

Combined AMN visualization settings now use a full-range 25 x 40 sweep
subset (1,000 conditions) plus the same 10,000 random test contexts, with
perplexity 80 and smaller points. This supersedes the earlier equal-sized
combined fit; sweep-only data and predictive evaluation remain unchanged.

The standalone AMN notebook also adds a fructose-only diagnostic slice:
100 sweep conditions at oxygen uptake 10 plus the same 10,000 random test
conditions, separately for glycolysis and PPP. These are additional
qualitative plots; no pretraining or predictive metrics change.

## Model E fructose/oxygen sweep visualization (2026-10-01)

`ecoli_iML1515_E_model_testing.ipynb` now uses the AMN sweep's validated
100 x 100 fructose/oxygen grid for qualitative post-layer token t-SNE. Its
checkpoint needs 55 inputs, so the 38 saved AMN sweep inputs are mapped by
name into E's vocabulary. The E AMN evaluation medium supplies fixed basal
inputs at 50, including cobalamin; glycerol and the four amino acids stay
at 2.2, and other E-only inputs are zero. Thus the variable grid and row
order match AMN, while the full fixed medium matches E's evaluation contract.
The saved AMN sweep flux targets are not used to score E.

An editable reaction-set dictionary and selector default to glycolysis and
PPP. For each set, one seeded openTSNE fit over all 10,000 contexts feeds
the side-by-side fructose/oxygen panels followed by the same four-color
joint plot as AMN. There are no sweep-plus-random figures in E. The plot
settings, E fixed rates, selected reactions and source paths are recorded
in the run directory. The full grid input validation and checkpoint mapping
passed; a 16-context real-checkpoint smoke run completed both pathways and
both plot paths. Full 10,000-context t-SNE fits were left for the notebook run.

## Standalone MINN A/E distribution-shift evaluation (2026-10-02)

The MINN notebook now adds pooled overall regression R2, MAE and RMSE for the
complete independent A and E test CSVs immediately after its in-distribution
overall metrics. It validates the source headers and shared 2,712-output
checkpoint order, then injects every source nutrient at the matching full
reaction-vocabulary token. A has 38 source inputs (14 outside MINN's 27
declared training inputs); E has 55 (28 outside). A omits MINN-trained
glucose, ethanol and cobalamin, so those context slots are zero. The saved
protocol records injected and additional inputs and their activity. These
scores test cross-distribution full-context inference, including unseen
input-token positions and distinct A/E FBA or pFBA target policies; they
should not be interpreted as a controlled nutrient-range-only shift. The
full 50,000-row runs are pending notebook execution.

## MINN checkpoint as a Faure-style growth reservoir (2026-10-03)

The MINN notebook now adds a 110-point AMN front-MLP experiment immediately
after t-SNE. It preserves the standalone AMN notebook's repeated stratified
10-fold training and medium bounds while replacing only the frozen reservoir
with `MINN_1M_d256_h8_l4_ff1024`. Full-vocabulary injection carries all 38
AMN source exchanges into the MINN checkpoint, including 14 outside its
declared 27-input training schema. Glucose, ethanol and cobalamin remain zero.
Per-medium OOF predictions are averaged across split seeds 10/11/12 before
pooled regression R2, MAE and RMSE; one matching scatter shows measured and
prediction uncertainty bars. The scored fold still selects the best epoch,
matching AMN's legacy protocol, so this is a cross-task control rather than
a strict held-out estimate. The full front-MLP CV run is pending.

The standalone MINN notebook's separate iML1515-fitted measured-context
sensitivity can now be rerun after a kernel restart without first recreating
the original fitted-file measured/predicted FT+pFBA results. It restores the
pre-run notebook state when those original results are absent. The final
comparison table still depends on both original results; fitted-file HPO/LOO
settings and metrics are unchanged.
