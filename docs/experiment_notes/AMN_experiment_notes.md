# AMN Experiment Notes

This file collects the current working notes for the iML1515 AMN-style
experiments. It is intentionally lighter than the MINN guide for now; expand it
as the notebook stabilizes.

## Purpose

The AMN experiments are inspired by Faure et al. 2023, where AMN means
Artificial Metabolic Network. In the paper, an AMN combines a neural layer with
a mechanistic metabolic layer so that growth-rate predictions are learned while
still being constrained by metabolic network structure.

In this repository, the current AMN-style experiment uses a pretrained
FluxTransformer as a frozen metabolic surrogate. A small trainable prior dense
network learns how to translate experimental medium composition into
FluxTransformer input values, and the frozen transformer predicts biomass flux.
The biological target is experimental growth rate.

## Main Files

- `ecoli_iML1515_A_model_testing.ipynb`: main AMN-style evaluation notebook.
- `ecoli_iML1515_AMN_model_testing.ipynb`: separate AMN_11k evaluation,
  retaining A's tests and plots with the experimental-pattern reservoir.
- `generate_ecoli_iML1515_A_data.py`: recommended simulated iML1515 data
  generator for future Faure-style AMN FluxTransformer training data.
- `generate_ecoli_iML1515_AMN_data.py`: separate experimental-pattern sampler
  with loguniform carbon/oxygen caps and pFBA by default.
- `generate_ecoli_iML1515_C_data.py`: shared AMN/MINN simulated-data
  generator for training a FluxTransformer reservoir that sees both Faure-like
  no-glucose media and MINN Table 4-style glucose/oxygen context.
- `flux_transformer.py`: canonical FluxTransformer model definition.
- `docs/Faure etal 2023.pdf`: main paper for AMN context.
- `docs/Faure_supplementary.pdf`: supplementary AMN architecture and benchmark
  details.
- [AMN repository inspection](../reference/amn_repository_notes.md): verified
  author-code, dataset, and publication provenance for the comparisons below.

Generator naming (2026-10-06): `generate_ecoli_iML1515_A_data.py` now names
the AMN-specific Model A sampler. This file rename preserves sampling, solver
defaults, token order, and the `iML1515_AMN_training_data` output prefix.

Notebook naming (2026-10-06): the standalone AMN evaluation notebook is now
`ecoli_iML1515_A_model_testing.ipynb`. Only its filename changed; checkpoint,
data paths, code cells, saved outputs, and experimental workflow are preserved.

## Faure Paper Context

Faure et al. frame AMNs as neural-mechanistic hybrids for improving
constraint-based metabolic predictions. Their E. coli growth-rate experiment
uses iML1515, M9-like media, combinations of carbon sources, and repeated
experimental growth measurements. The reported workflow uses stratified
cross-validation and compares AMN variants against purely mechanistic and
purely neural alternatives.

For this repository, the exact architecture is not copied directly. Instead,
the frozen FluxTransformer plays the role of a learned mechanistic reservoir.
This means the experiment asks a slightly different question: can a transformer
trained on simulated FBA media conditions provide a useful metabolic prior for
experimental growth-rate prediction?

Also note that the local FluxTransformer uses its checkpoint's native iML1515
reaction vocabulary. Do not assume the reduced or duplicated-reaction setup from
the Faure AMN figures unless the generator and checkpoint were built that way.

### Verified author sampling and solver scope (2026-10-06)

The authors' iML1515 simulated reservoir uses **only the 110 variable-source
presence patterns in the experimental data**, with 100 randomized uptake-cap
draws per pattern, giving 11,000 pFBA samples. The 110 patterns comprise 10
single-source, 20 two-source, 40 three-source, and 40 four-source conditions
from the ten variable carbon sources. The simulation does not independently
draw arbitrary one-to-four-source subsets from that list. It uses experimental
composition to choose the active sources; simulated targets are pFBA fluxes,
not experimental growth labels.

For a condition with `k` variable sources, every selected source stays active
and its cap is independently drawn as `(k+1) * j * 2.2 / 99`, with integer
`j=1,...,99`. All 28 fixed inputs, including oxygen, glycerol, and the four
amino acids, have cap 2.2. This gives variable-source maxima of 4.4, 6.6, 8.8,
and 11 for `k=1,2,3,4`. Evidence: author `Build_Dataset.ipynb` cell 13,
`Library/Build_Dataset.py` lines 509--544, and the saved
`Dataset_model/iML1515_UB.npz`; detailed provenance is in the
[repository inspection](../reference/amn_repository_notes.md).

**pFBA is not used in every AMN experiment.** The inspected author release
(`10db2a62e8ea17b303ce21a9a1261b7328f861e9`) uses:

| Author workflow | Solver or target source | Evidence (one-based notebook cells) |
|---|---|---|
| E. coli core simulated datasets | pFBA | `Build_Dataset.ipynb` cell 11; saved core NPZ method fields |
| iML1515 simulated reservoir | pFBA | `Build_Dataset.ipynb` cell 13; `iML1515_UB.npz` method field |
| E. coli mechanistic baseline / reservoir-cap comparison | pFBA | Actual `run_cobra(method='pFBA')` call in `Build_Dataset.ipynb` cell 20 |
| E. coli Biolog knockout mechanistic comparator | pFBA | `Build_Dataset_KO.ipynb` cell 14 |
| P. putida/iJN1463 simulated data and mechanistic comparator | Plain FBA | `Build_Dataset.ipynb` cells 14 and 21; `IJN1463_10_UB.npz` method field |
| Experimental growth targets / direct AMN-Wt, AMN-LP, AMN-QP | Measured labels / learned mechanistic layers | `method='EXP'` loads measurements; direct AMN training does not call COBRA pFBA |

The author pFBA branch calls `cobra.flux_analysis.pfba(model)` without
overriding `fraction_of_optimum` (COBRA default 1.0). It preserves the maximum
growth objective and then minimizes total flux. The helper's default argument
`method='FBA'` does not identify the solver actually requested by each notebook.
Likewise, a notebook's `method='EXP'` setting identifies data loading, not the
solver in its later comparator call. Publication wording such as
"FBA-generated" alone does not distinguish plain FBA from pFBA.

The [author MINN inspection](../reference/minn_repository_notes.md), completed
2026-10-06, establishes the comparison's different nutrient support: glucose
minimal medium with no AMN glycerol/amino-acid supplements in the 587-reaction
iAF1260 reservoir model. Its 16 basal imports have XML caps of 999999, and
CO2/ethanol/acetate are secretion channels rather than additional supplied
carbon sources. The MINN paper describes 2,000 randomized five-channel
simulations, but no original generator/dataset establishes their exact bounds
or FBA variant. This contrasts with AMN's verified 110-pattern, 11,000-row
pFBA prior; do not extend the AMN pFBA finding to undocumented MINN pretraining
or treat the two authors' media as equivalent.

## Simulated AMN Data

`generate_ecoli_iML1515_A_data.py` defines Model A's existing sampling contract
with media settings chosen to resemble the Faure experimental setup. The
separate experimental-pattern sampler below does not replace Model A.

Key points:

- The model is `models/iML1515.xml`.
- The generator writes input columns in the same 38-exchange order as
  `AMN_data/iML1515_EXP.csv`, after removing Faure's `"_i"` suffix.
- The generator explicitly sets the objective to
  `BIOMASS_Ec_iML1515_core_75p37M`.
- Variable carbon sources are selected from the Faure-style carbon set:
  ribose, maltose, melibiose, trehalose, fructose, galactose, acetate,
  D-lactate, succinate, and pyruvate.
- D-glucose is intentionally not an AMN input and is closed during generation.
- Glycerol is fixed as an additional carbon source.
- Base medium exchanges include phosphate, CO2, protons, water, ammonia,
  oxygen, ions, sulfate, sodium, chloride, and trace elements.
- Alanine, proline, threonine, and glycine are treated as fixed amino-acid
  exchanges.
- The generator uses uptake rates around the Faure experimental scale for
  carbon-containing supplements: selected variable carbon sources, glycerol, and
  amino-acid exchanges default to `2.2`, while non-carbon base nutrients default
  to `10.0`.
- Each sample independently selects one to four of the ten variable carbon
  sources, without restricting the subset to the authors' 110 experimental
  patterns. Selected-source caps are continuous draws from `0.05` to `2.2`,
  rather than the authors' cardinality-scaled discrete grid. The fixed `2.2`
  glycerol/amino-acid caps match the author simulation, but non-carbon base
  caps use `10.0` rather than `2.2`, and oxygen varies from `1.0` to `10.0`.
  Use `--fixed-oxygen` only for an explicit ablation; this alone does not
  reproduce the author distribution.
- The current solver default is **plain FBA** (`--flux-solver-mode fba`).
  Optional `--flux-solver-mode pfba` defaults to
  `--pfba-fraction-of-optimum 0.999`, distinct from the author pFBA call's
  default 1.0. Record the actual generation arguments when interpreting a
  checkpoint; a Faure-style medium does not establish pFBA provenance.
- Each sample starts from a closed uptake medium while preserving the model's
  default exchange upper bounds for secretion. This avoids carrying stale solver
  bounds between samples while keeping unselected nutrients closed.
- The generator follows the robust MINN-style process: it loops until the
  accepted sample target is reached, writes through a timestamped temporary CSV,
  reports attempts and feasible rate, has a max-attempt guard, and periodically
  reloads the model/solver.
- The default output prefix is `iML1515_AMN_training_data`, so a default
  50,000-sample run saves `./data/iML1515_AMN_training_data_50000_samples.csv`
  unless that file already exists. Use `--overwrite-existing` or a different
  `--output-prefix` deliberately.
- Outputs are all iML1515 reaction fluxes with `"_flux"` suffixes.

Use this generator as the source of truth for input column order, exchange
names, and rate conventions when checking the notebook.

### Experimental-pattern AMN sampler (2026-10-06)

`generate_ecoli_iML1515_AMN_data.py` is a new sampler; Model A and its existing
consumers remain unchanged. It uses the same model, objective, 38-input order,
full reaction-flux outputs, closed-medium reset, accepted-sample loop,
temporary-file publishing, and solver reload handling as Model A.

- Conditions come from `AMN_data/EXP110.csv` by default (`--conditions-csv`).
  Only the ten binary carbon-source indicators are used; measured growth rates
  do not become simulated targets. The loader checks 110 unique patterns with
  source-count frequencies 10/20/40/40 for one/two/three/four sources.
- Conditions are visited in shuffled cycles of all 110 patterns. Failed solves
  retry the same pattern with new uptake caps. The default `--n-samples 11000`
  therefore gives exactly 100 accepted rows per condition. Any positive sample
  count is supported; condition counts differ by at most one, including zeros
  when fewer than 110 rows are requested.
- Each selected variable source is drawn independently loguniformly over
  **0.05--10** (`--carbon-rate-min`, `--carbon-rate-max`). The lower bound is
  inherited from Model A because the request specified only the new maximum.
- Oxygen is loguniform over **1--25**; other 22 basal uptake caps stay at **10**.
  Glycerol and alanine/proline/threonine/glycine stay at **2.2**. Loguniform
  draws retain Model A's rounding to two decimal places. Positive cap values
  map to negative exchange lower bounds, while secretion upper bounds remain
  the model defaults. `--fixed-oxygen` remains an explicit optional ablation.
- **pFBA is the default**, with `fraction_of_optimum=0.999` inherited from A;
  `--flux-solver-mode fba` and `--pfba-fraction-of-optimum` remain selectable.
  Default training seed is 42; use `--seed 9` for test data. The output is
  `data/iML1515_AMN_training_data_11000_samples.csv`; use a distinct
  `--output-prefix` or data directory for A/AMN runs of equal sample count.

This matches the author's experimental-pattern restriction and default
100-draw allocation, while deliberately changing basal caps, oxygen sampling,
variable-carbon sampling, and the optimum fraction. It is not an exact
reproduction of the author's cardinality-scaled discrete cap grid or fixed
2.2 basal medium. No training data or checkpoints were replaced.

Example:

```bash
python generate_ecoli_iML1515_AMN_data.py --n-samples 11000 --seed 42
```

Roihu batch entry point: `scripts/roihu/samplejob_AMN.sh` runs 11,000 samples
with seed 42 and pFBA fraction 0.999, using the sampler's default uptake caps.
It follows the existing CPU environment and project paths, explicitly reads
`$CODEDIR/AMN_data/EXP110.csv`, and writes into `$WORKDIR/data`. Submit with
`sbatch scripts/roihu/samplejob_AMN.sh` after deploying the sampler, model, and
experimental condition CSV under `$CODEDIR`.

### AMN_11k evaluation notebook (2026-10-06)

`ecoli_iML1515_AMN_model_testing.ipynb` copies all 59 cells from the A
evaluation notebook, retaining the same simulated diagnostics, B/E
distribution-shift tests, pathway t-SNE plots, deterministic fructose/oxygen
sweep and combined random-context plots, experimental-growth CV/repeated CV,
FBA-with-predicted-Vin comparison, and TabPFN baselines. CV seeds, splits,
metrics, prior-network settings, plotting functions, and sweep bounds are
unchanged. The original A notebook is preserved.

- Expected checkpoint:
  `models/AMN_11k_d256_h8_l4_ff1024/AMN_11k_d256_h8_l4_ff1024_checkpoint.pth`.
  This follows A's model naming; the AMN_11k checkpoint is not present locally
  yet, so architecture, weights, and training provenance cannot be verified
  until it is copied into this location. The notebook exposes `model_name`
  and loads architecture dimensions from checkpoint configuration.
- Main independent test CSV:
  `data/iML1515_AMN_11k_test_data_11000_samples.csv`, generated with the new
  AMN sampler with **test seed 9**; training uses **seed 42**. A complete
  11,000-row run gives exactly **100 rows per pattern**. Presence patterns
  are shared intentionally; independent seeds produce separate uptake-cap draws.
  All sampler defaults remain unchanged (pFBA 0.999, carbon loguniform
  0.05--10, oxygen loguniform 1--25, basal 10, fixed organic caps 2.2).
- Provenance belongs in these experiment notes; no adjacent JSON file is
  required or used by the notebook. The user requested removal of the JSON
  provenance sidecar on 2026-10-07.
- `N_EVAL_SAMPLES` is 11,000; the 80/20 diagnostic split remains unchanged.
  The 10,000 random contexts for combined t-SNE come from this test CSV.
  Input/token mapping, full output order, and separation from the checkpoint's
  training-data path are asserted before evaluation.
- The existing deterministic 10,000-row sweep and B/E test-data paths remain
  identical to A. They are comparison conditions, not regenerated to match
  AMN_11k's wider loguniform caps. These files are currently absent locally.
- Figure/report directories are model-specific: the existing
  `pics/{date}/{model_name}` and `insights/thesis/{model_name}`. Copied A
  outputs and execution counts are cleared to avoid claiming new-model results.
  The FBA comparison imports its unchanged medium-reset helper from the new
  AMN sampler. Full notebook execution awaits the checkpoint, shared test
  artifacts, and the notebook dependencies (including TabPFN).

No new experimental-growth scores or neural model results are claimed.

Seed correction (2026-10-07): training **must use 42**, and test generation
**must use 9**. The sampler default and Roihu training job now use 42. The
previously generated test CSV used seeds 10--13 and is **not the requested
seed-9 test set**; its earlier numerical validation does not establish the
correct seed provenance. Replace it before evaluating the AMN_11k model.
Any training CSV/checkpoint produced with seed 9 likewise needs regeneration
and retraining with seed 42. Existing checkpoints have not been verified or
retrained. The incorrect test sidecar was removed. The user chose to run generation
themselves; neither existing CSV was regenerated during this correction.

Reproduction commands from the repository root (explicit overwrite is needed
when the same filenames already exist):

```bash
.venv/bin/python generate_ecoli_iML1515_AMN_data.py --n-samples 11000 --seed 42 --output-prefix iML1515_AMN_training_data --overwrite-existing
.venv/bin/python generate_ecoli_iML1515_AMN_data.py --n-samples 11000 --seed 9 --output-prefix iML1515_AMN_11k_test_data --overwrite-existing
```

## Shared AMN/MINN Reservoir Data

Cross-task generator, checkpoint, trial, and result state is maintained in
`AMN_MINN_shared_reservoir_notes.md`. Keep this section focused on AMN-specific
implications and update both notes when a shared change affects AMN behavior.

`generate_ecoli_iML1515_C_data.py` is a separate generator for training a
single iML1515 FluxTransformer reservoir that should be usable in both the
AMN-style experimental growth notebook and the MINN Table 4-style reservoir
workflow. Both local generators are adaptations. Reproducing the authors'
simulation requires their 110-pattern restriction, cap distribution, fixed
bounds, and solver settings described above.

Key points:

- The input set contains the 38 Faure-style exchange identities plus two extra
  MINN-context exchanges: `EX_glc__D_e` and `EX_etoh_e`.
- The first five inputs are the MINN reservoir context exchanges:
  `EX_glc__D_e`, `EX_o2_e`, `EX_co2_e`, `EX_etoh_e`, and `EX_ac_e`.
- The default solver mode is pFBA with `fraction_of_optimum=0.999`, matching the
  MINN simulated-data convention more closely than plain FBA.
- The generator samples explicit regimes:
  - `minn`: all five MINN context caps are independently sampled as integers;
    glucose/oxygen are uptake caps, CO2/ethanol/acetate are secretion upper
    caps, and glycerol/amino supplements are absent.
  - `faure`: glucose is absent, oxygen is flexible, glycerol/amino acids use
    their Faure `2.2` caps, and 1-4 Faure carbon sources are independently
    selected without the authors' 110-pattern restriction. Its variable-carbon
    caps, oxygen, fixed base bounds, and pFBA optimum fraction also differ
    from the author simulation.
  - `mixed`: the same five MINN caps are sampled with optional non-acetate
    Faure carbon sources to bridge the two distributions.
- Shared non-carbon base nutrients use `50` in all three regimes. Downstream
  AMN evaluation of a checkpoint trained from this shared generator must also
  use fixed base inputs of `50`. This is a fixed cross-regime normalization, not
  an additional randomized AMN variable.
- Default regime weights are `minn=0.50`, `faure=0.40`, and `mixed=0.10`.
- Current training defaults are `1,000,000` accepted samples, seed `42`, and
  output prefix `iML1515_AMN_MINN_training_data`. These remain normal CLI
  options; use seed `9` and a test-specific prefix for a separate test set.
- Acetate is necessarily context-dependent in this shared file: in no-glucose
  Faure rows it can still represent a Faure carbon-source uptake cap, while in
  glucose/MINN rows it is used as a sampled secretion upper cap. Avoid using
  this generator for claims that require a single unambiguous Faure acetate-input
  interpretation.
- The AMN notebook now zero-fills optional absent `EX_glc__D_e` and `EX_etoh_e`
  experimental columns when a shared AMN/MINN checkpoint is loaded. Unknown
  missing inputs still raise an error.
- In `ecoli_iML1515_AMN_MINN_model_testing_trial.ipynb`, fixed present base
  inputs now use `50`, matching the shared generator. The earlier saved
  reservoir metrics used `10` and are superseded; rerun the experimental
  reservoir section through its prior-net comparison before reporting a
  corrected shared-branch result. Glycerol and amino-acid caps remain `2.2`,
  and absent glucose and ethanol remain zero.

Important legacy note: checkpoints made with an older generator may come from a
fixed-attempt loop that reset only exchange lower bounds. The current
`generate_ecoli_iML1515_A_data.py` uses an accepted-sample loop and a fully
reset medium.

## Current Notebook Workflow

The notebook currently has four main parts.

1. Load and inspect a pretrained FluxTransformer.

   The active checkpoint is currently:
   `./models/iML1515_500k_d256_h8_l3_ff1024/iML1515_500k_d256_h8_l3_ff1024_checkpoint.pth`.

   This checkpoint predates the current stable generation behavior in
   `generate_ecoli_iML1515_A_data.py`. Do not treat its metrics as results from
   newly regenerated data unless the model is retrained and the notebook rerun.

   The simulated test data loaded for FluxTransformer diagnostics is currently:
   `./data/iML1515_test_data_50000_samples.csv`.

   Use a separately generated test file for reported simulated-flux diagnostics.
   Do not use `data_info["dataset"]` for those metrics, since that points to the
   file used to train the checkpoint and can leak training rows into evaluation.

2. Evaluate simulated-data flux predictions.

   The notebook plots FluxTransformer diagnostics for selected fluxes, including
   the iML1515 biomass reaction:
   `BIOMASS_Ec_iML1515_core_75p37M_flux`.

   For these diagnostics, keep `plot_flux()` on the full forward pass:
   `model(c, output_subset=None)`. Do not use `output_subset` to request only
   the plotted flux. That changes the token set seen by attention and gives a
   different prediction, so the diagnostic plots become wrong. If CUDA memory is
   tight, reduce the plotting batch size instead.

3. Train a prior dense network on experimental media and growth data.

   Experimental data is loaded from:
   `./AMN_data/iML1515_EXP.csv`.

   The target column is:
   `GR_AVG`.

   The prior network receives the variable medium inputs and predicts bounded
   uptake values for the same variable input positions. These predicted medium
   values are inserted into the full FluxTransformer input tensor. Fixed medium
   components are kept at generator-aligned rates.

4. Inspect selected iML1515 pathway-token embeddings.

   The notebook includes exploratory t-SNE cells for selected reaction subsets
   such as glycolysis, pentose phosphate pathway, TCA, and fermentation. These
   cells support debugging and qualitative model inspection only. They are not
   currently thesis-facing figures; thesis t-SNE/embedding interpretation should
   come from the E. coli core experiments.

## Experimental Data Setup

The notebook normalizes experimental column names such as `EX_pi_e_i` to
`EX_pi_e` so they match FluxTransformer input names.

The current experimental dataset has 110 samples. It uses 10 variable carbon
source indicators:

- `EX_rib__D_e`
- `EX_malt_e`
- `EX_melib_e`
- `EX_tre_e`
- `EX_fru_e`
- `EX_gal_e`
- `EX_ac_e`
- `EX_lac__D_e`
- `EX_succ_e`
- `EX_pyr_e`

Cross-validation is stratified by the number of active carbon sources. Current
strata are 1, 2, 3, and 4 active carbon sources.

`AMN_data/EXP110.csv` provides `GR_STD`, which is used for plotting
experimental uncertainty around measured growth rates.

## Current Prior-Network Protocol

Current settings in the AMN notebook:

- Prior model: one-hidden-layer dense network.
- Hidden size: `512`.
- Dropout: `0.0`.
- Optimizer: `AdamW`, learning rate `1e-3`, weight decay `1e-3`.
- Loss: Huber loss with `delta=0.03`, applied to predicted biomass flux versus
  experimental `GR_AVG`.
- Epochs: `90`.
- Maximum epochs: `100`.
- Early stopping patience: `15`.
- Base batch size: `1`.
- Cross-validation: 10-fold stratified CV repeated with split seeds
  `[10, 11, 12]`.
- Final full-data fit: ensemble seeds `[10, 11, 12]`.

The dense prior predicts nonnegative bounded rates. Carbon-source outputs are
bounded by `2.2`; oxygen is bounded by `10.0`.

## Faure Identity-Line Audit

Faure et al. report that the uncertainty boxes intersect the identity line for
79% of AMN-QP predictions, 76% of AMN-LP predictions, and 74% of AMN-Wt
predictions. Their text defines each box using the standard deviations of both
measurement and prediction. In practice, that corresponds to testing whether the
measured interval, `GR_AVG +/- GR_STD`, overlaps the predicted interval,
`prediction_mean +/- prediction_std`.

This is not a clean accuracy metric. Increasing the prediction standard
deviation makes the vertical interval wider and can increase the intersection
rate even when the mean prediction is not better. Treat it as a loose
uncertainty/coverage diagnostic, not as a primary model-comparison statistic.

This does not reproduce cleanly from the available Fig. 3 source data. Using
`Data_Fig3.xlsx` from the article source-data ZIP together with measured
`GR_STD` from `AMN_data/EXP110.csv`, the same interval-overlap criterion gives:

- AMN-QP: `68/110 = 61.8%`
- AMN-LP: `69/110 = 62.7%`
- AMN-Wt: `80/110 = 72.7%`

Plausible alternatives using replicate min-max ranges also did not recover the
published `79/76/74%` pattern. The raster version of Fig. 3a is not reliable
for an exact count because the bars overlap, but it also does not visually
support `87/110` QP boxes crossing the identity line.

## Current Result Snapshot

The current notebook output suggests that the frozen-FluxTransformer prior model
is performing better than the TabPFN baseline on the same experimental media
features.

Current prior-network out-of-fold summary:

- Pooled OOF `R2`: about `0.878`.
- Pooled OOF `MAE`: about `0.0235`.
- Pooled OOF `RMSE`: about `0.0294`.

Future runs of `ecoli_iML1515_A_model_testing.ipynb` explicitly use
TabPFN-3.5 (`ModelVersion.V3_5`) for main CV, repeated CV, and uncertainty
experiments. Package version and checkpoint are printed; existing features,
splits and seeds are retained. No 3.5 evaluation has been run for this update.

Historical TabPFN baseline (predates the 3.5 switch):

- Pooled OOF `R2`: about `0.807`.
- Pooled OOF `MAE`: about `0.0297`.
- Pooled OOF `RMSE`: about `0.0370`.

Treat these values as a notebook-state snapshot, not final thesis numbers,
unless the notebook is rerun from top to bottom with the intended checkpoint and
data.

## Saved Figures

The notebook currently saves thesis-facing figures under `./insights/thesis`,
including:

- `iML1515_insample_fit.png`
- `iML1515_oof_true_vs_predicted_all_cv_folds.png`
- `iML1515_oof_true_vs_predicted_colored_by_fold.png`
- `iML1515_oof_true_vs_predicted_with_std_bars.png`
- `iML1515_oof_true_vs_predicted_with_exp_and_seed_std_bars.png`
- `iML1515_tabpfn_oof_true_vs_predicted.png`

Flux diagnostics and exploratory iML1515 pathway t-SNE plots are saved under the
date/model-specific `pic_dir` used in the notebook. Treat those t-SNE plots as
notebook diagnostics, not as planned thesis figures.

## Sanity Checks

- Confirm that the checkpoint output token order matches the loaded CSV columns.
- Confirm that all experimental input columns exist after removing the `"_i"`
  suffix.
- Confirm that binary carbon-source features remain 0/1 before TabPFN or
  stratified CV.
- Check that fixed medium rates in the notebook match
  `generate_ecoli_iML1515_A_data.py` or the shared generator used to train the
  evaluated checkpoint.
- Print and review the variable input columns learned by the prior ANN.
- Keep simulated-data FluxTransformer diagnostics separate from experimental
  growth-rate evaluation.
- Do not describe the current experiment as an exact reproduction of Faure et
  al.; it is a FluxTransformer-reservoir adaptation of the AMN idea.

## Open Questions

- Whether the prior dense network should predict only variable carbon rates or
  also selected fixed/media context channels.
- Whether oxygen should remain trainable/flexible in all final runs or be tested
  against a fixed-oxygen ablation.
- Keep fixed glycerol and amino-acid caps at the Faure value of `2.2` unless an
  explicitly named sensitivity experiment is being run.
- Whether the FluxTransformer checkpoint should be retrained only on AMN-style
  simulated media or mixed with broader iML1515 media.
- Whether final reporting should compare against Faure AMN-QP/LP/Wt numbers,
  TabPFN only, or additional pFBA baselines.

## A union B combined notebook (2026-09-08)

The AMN branch in `ecoli_iML1515_AB_union_model_testing.ipynb` trains its own
width-512 prior MLP, separately from both MINN MLPs, with no TabPFN tests. Its
union A-regime inputs use base 10, fixed glycerol/amino acids 2.2, and absent
glucose/ethanol/cobalamin zero. The loader joins GR_STD by medium composition
and verifies growth alignment. Repeated ten-fold CV uses seeds 10/11/12; an
inner 20% training split selects epochs and a fresh full outer-training refit
predicts the untouched outer fold. This corrects outer-fold early-stopping
selection for this new workflow; existing notebooks are unchanged.

Settings and exports are explicit in the notebook; helpers are in
`iml1515_ab_evaluation.py`. No production union growth results are recorded yet.

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

Notebook compatibility fix (2026-09-13): the AMN plot and final summary cells
reload an older imported evaluation module if `summarize_amn_oof` is absent.
Existing trained results remain in memory; rerunning these cells does not
require a kernel restart or retraining. Metric formulas are unchanged.

Model C combined evaluation (2026-09-14): `ecoli_iML1515_C_model_testing.ipynb` uses `iml1515_evaluation.py` with explicit C input schema and basal 50. The configured legacy 40-input checkpoint excludes cobalamin; it is not injected. Existing union defaults and training protocols are unchanged. See `iML1515_sampling_study_notes.md` for details.


## Controlled AMN fructose/oxygen sweep (2026-09-24)

`generate_AMN_sweep.py` is a standalone diagnostic-data generator for a future
AMN embedding analysis. Defaults are a deterministic 100 x 100 Cartesian grid:
fructose 0.05--2.2 and oxygen 1--10, with endpoints included and oxygen varying
fastest. It preserves the current AMN generator's 38 ordered inputs, native
2,712 reaction outputs, closed-medium reset and secretion bounds, basal uptake
10, and fixed glycerol/four amino acids at 2.2. Other variable carbons and
glucose have zero uptake. The default is FBA, matching the current AMN script;
optional pFBA retains fraction_of_optimum=0.999. This does not change the
sampling study's separate recommendation to use pFBA for new comparisons.

The model CSV is `data/iML1515_AMN_sweep_10000_samples.csv`; metadata is
`data/iML1515_AMN_sweep_10000_metadata.csv`. The notebook's current `load_data`
requires only nutrient inputs before the first flux column and only flux
columns thereafter, so metadata must remain separate. Metadata `sample_id`
is the zero-based model row number for complete grids, with
`sample_id = sweep_fructose_index * oxygen_levels + sweep_oxygen_index`.
Any later row selection/reordering must apply the same indices to metadata.

Only optimal finite solutions are retained. Failed points are reported with
rates and grid indices; no replacements are sampled. Incomplete runs raise an
error and retain explicitly named `.partial.csv` files rather than publishing
a final dataset. Partial metadata retains original grid IDs despite missing
rows. Existing outputs require `--overwrite-existing`.

Validation: a 3 x 3 FBA smoke run produced nine optimal solutions with the
expected schema. Full 10,000-point generation and notebook integration are
not performed; the training generator and notebook remain unchanged.


Glycolysis t-SNE display update: a second plot immediately follows the existing
glycolysis plot, using the existing helper's `sample_color_mode="rainbow"`
(as in TCA). It retains 4,000 contexts and perplexity 40, and saves separately
with suffix `_tsne_glycolysis_reactions_rainbow`. Rainbow colors encode the
helper's nutrient-context ordering; reaction identity remains in center markers
and labels. Plot generation was not rerun for this addition.

PPP sweep visualization: a dedicated execution cell follows the glycolysis
sweep and calls `plot_AMN_sweep_tsne` with `ppp_reactions`, perplexity 40 and
seed 10. It uses all sweep contexts, produces the two single-variable panels
then the joint bivariate figure from one fit, and retains its result as
`sweep_ppp_tsne_result`. Pathway-specific filenames preserve glycolysis plots.
The full PPP sweep t-SNE was not run for this addition.

Joint sweep styling uses light gray `#F0F0F0`, high-fructose red
`#D73027`, high-oxygen blue `#0072B2`, and high-both yellow-green `#A6CE39`.
The cloud and square legend share the same bilinear mapping. Joint-figure
point alpha is 0.90 and reaction centers are white with black edges; the
single-variable panels retain their existing colors, opacity and markers.
Existing saved figures require rerunning the plotting cells to adopt this style.


## Sweep plus random test contexts (2026-09-25)

Each glycolysis/PPP sweep joint figure is followed by a separate combined
t-SNE figure. It fits all 10,000 sweep contexts plus the same 10,000 independent
AMN test CSV rows, chosen without replacement from the full 50,000-row file
with NumPy seed 10. Input/output schemas are checked against the checkpoint;
inputs use the same nutrient-token injection as `load_data`. Existing figures
and their fits are unchanged. Combined fits use openTSNE FFT, PCA initialization,
perplexity 40 and seed 10. Coordinates differ from sweep-only fits and should
be interpreted within each combined figure.

Random points are darker peach-yellow `#D9AA73`, alpha 0.20, and drawn behind the
bivariate sweep points (alpha 0.90). White reaction centers use sweep points
only in the combined embedding. Filenames end in `_with_random_bivariate.png`.
Results retain dataset labels, original source-row IDs, reaction labels and
coordinates as `sweep_random_glycolysis_result` and `sweep_random_ppp_result`.
Nutrient-color arrays apply only to the returned `sweep_mask`.
Full 20,000-context fits were not run during implementation.

The redundant single-run TabPFN OOF plotting cell and its saved notebook
output were removed. The repeated-CV plot remains; training, metric
calculations and epoch-selection behavior are unchanged by this removal.


## Combined t-SNE density revision (2026-09-25)

Supersedes the combined-fit settings above: use 1,000 sweep contexts selected
as 25 evenly spaced fructose indices x 40 evenly spaced oxygen indices from
the validated 100 x 100 grid, including both endpoints on both axes. Preserve
original sweep sample IDs and align colors by selected row order. Both pathways
reuse this subset and the same 10,000 random test rows. Combined perplexity is
80, seed 10, with PCA initialization and FFT; fits contain 154,000 glycolysis
points or 88,000 PPP points. Sweep/random marker sizes are 10/3 and opacity
0.85/0.20; all random reactions retain one peach-yellow color.

Sweep-only point markers are size 4. Normal reaction-subset plots retain their
original marker sizes (10 generally, 30 for up to 20 reactions, and 36 for up
to 10 reactions); reaction centers and labels retain their sizes. Sweep-only
fits retain all 10,000 contexts and perplexity 40.
Stale t-SNE notebook outputs were cleared. Execution replaces the same PNG
filenames rather than creating an additional comparison variant. Existing PNGs
on disk are not regenerated until the plotting cells run.

Verified the 1,000 unique grid selections, full ranges and all four corners,
original sample IDs, nutrient-color alignment, and both combined calls with
mocked inference/t-SNE. Syntax and diff checks passed; full fits were not run.


## Fructose-only sweep with random contexts

After each glycolysis/PPP combined plot, an additional fit uses the 100
fructose levels at oxygen uptake 10 from the full sweep, plus the same 10,000
random AMN test conditions. Original sweep IDs (99, 199, ..., 9999) are
preserved. Only the sweep oxygen is fixed; random conditions retain their
original nutrient values. Perplexity 80, seed 10, point styling and inference
logic match the combined experiment. Sweep colors use the high-oxygen edge
of the existing palette with a one-dimensional fructose colorbar. Outputs
end in `_oxygen10_with_random.png`; separate result dictionaries end in
`_oxygen10_result`. Previous figures remain unchanged.

Validated the 100 full-range levels, original IDs, color mapping and both
plotting paths with mocked inference/t-SNE. Full fits were not run.

Joint sweep legend axes use Fructose and Oxygen (units retained), with
17-point labels and 19-point legend titles in sweep-only and combined plots.
The fixed-oxygen combined legend uses Fructose uptake with a 17-point label.

## Full-flux distribution-shift diagnostics

`ecoli_iML1515_A_model_testing.ipynb` now scores the full independent
50,000-row B (MINN Tazza) and E (general) test CSVs immediately after its
in-distribution overall-metrics cell. It reports pooled regression R2, MAE
and RMSE over all 2,712 checkpoint flux outputs, checks source generator
input order and exact checkpoint output order, and saves the scores and input
mapping protocol under `pic_dir`. No full test scores were computed during
implementation.

The AMN checkpoint has 38 declared training inputs, but all 27 B and 55 E
source inputs have matching reaction tokens in its 2,712-token vocabulary.
Under full-output inference, the model reads context values at every token,
so the notebook injects every source value at its matching `*_flux` token.
Three B inputs (glucose, ethanol and cobalamin) and 17 E inputs lie outside
the declared training-input set. B omits 14 of AMN's trained inputs; their
context values are zero. `input_token_indices` only guarantees that declared
inputs remain present under subset inference. The additional tokens were
never nonzero input features during AMN training, so their responses remain
extrapolations. The protocol records per-input active-row counts. All 50,000
B rows have positive glucose and cobalamin; all 50,000 E rows have positive
cobalamin and 5,594 E rows have positive glucose. One-row real-checkpoint
smoke tests for B and E verified every injected context position and matched
independent scikit-learn R2, MAE and RMSE. On the B row, removing the three
additional inputs changed the predictions, confirming that the former
projection had discarded readable context. Full 50,000-row scores remain
uncomputed.

## Validation-fold Vin passed to ordinary FBA

The AMN notebook now includes a Faure Supplementary Figure S9-style test just
before its TabPFN section. The existing prior-network CV cell saves the full
38-exchange `Vin` vector predicted by each selected fold model on its own
validation rows, with the medium row and split seed. It retains the current
three repeated stratified 10-fold splits; the full-data prior ensemble is not
used for this comparison. Capturing `Vin` requires rerunning that CV training
cell because earlier notebook executions retained growth predictions only.

The new test loads `models/iML1515.xml`, maximizes
`BIOMASS_Ec_iML1515_core_75p37M` with ordinary FBA, closes unselected uptake
as the AMN generator does, and sets each predicted positive uptake cap as a
negative exchange lower bound. It solves each of the 330 validation-fold
contexts separately before averaging the three FBA growth predictions for
each of 110 media. Pooled regression R2, MAE and RMSE and per-split variation
are reported beside the frozen FluxTransformer's existing OOF scores. A
row-linked CSV with `Vin`, FBA status and growth is saved under `pic_dir`; the
measured-versus-FBA-predicted figure follows the existing growth-plot style
and is saved under `insights/thesis`.

Faure's Figure S9 used one 10-fold CV, whereas this notebook retains its
three-repeat protocol. The inspected author's downstream E. coli comparator
uses pFBA (`Build_Dataset.ipynb` cell 20); this notebook's ordinary FBA is a
local adaptation. The current front-network procedure also selects its
epoch on the outer validation fold. This comparison preserves the notebook's
existing protocol but is not an untouched-test estimate. A small mock CV
verified that captured `Vin` comes from the selected fold model; six FBA
smoke solves on real media verified row aggregation, metrics, CSV and plot
generation. Full experimental CV/FBA results have not been run.

## Cross-task MINN reservoir growth control (2026-10-03)

`ecoli_iML1515_B_model_testing.ipynb` now evaluates the 110-point Faure-style
growth task using its frozen MINN checkpoint and a new AMN-style front MLP.
It keeps the standalone AMN notebook's medium conversion, 512-hidden-unit
network, Huber/AdamW settings, 10-fold carbon-count stratification repeated
with split seeds 10/11/12, training seed 10, 100-epoch limit and validation-fold
early stopping. All 38 AMN exchanges are mapped into the MINN checkpoint's
full reaction-token vocabulary, including 14 inputs outside its declared
training-input set. The printed pooled R2, MAE and RMSE average the three
validation predictions per medium; the single saved scatter includes
experimental horizontal and split-seed vertical standard-deviation bars.
This control tests transfer of a task-mismatched reservoir, with the same
legacy validation limitation as the reference AMN notebook. A real-checkpoint
one-step training smoke passed; the full CV result is pending.
