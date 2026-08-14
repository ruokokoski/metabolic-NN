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

- `ecoli_iML1515_AMN_model_testing.ipynb`: main AMN-style evaluation notebook.
- `generate_ecoli_iML1515_AMN_data.py`: recommended simulated iML1515 data
  generator for future Faure-style AMN FluxTransformer training data.
- `generate_ecoli_iML1515_AMN_MINN_data.py`: shared AMN/MINN simulated-data
  generator for training a FluxTransformer reservoir that sees both Faure-like
  no-glucose media and MINN Table 4-style glucose/oxygen context.
- `flux_transformer.py`: canonical FluxTransformer model definition.
- `docs/Faure etal 2023.pdf`: main paper for AMN context.
- `docs/Faure_supplementary.pdf`: supplementary AMN architecture and benchmark
  details.

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

## Simulated AMN Data

`generate_ecoli_iML1515_AMN_data.py` is the recommended generator for
new iML1515 FBA samples with media settings chosen to resemble the Faure
experimental setup.

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
- The Faure AMN sampling policy is retained, including the `2.2` caps for fixed
  glycerol and the four amino acids. The only deliberate difference in which
  AMN medium variables are sampled is oxygen: Faure keeps oxygen fixed, whereas
  the FluxTransformer generator samples it between `1.0` and `10.0` by default.
  Use `--fixed-oxygen` only for an explicit ablation.
- Each sample starts from a closed uptake medium while preserving the model's
  default exchange upper bounds for secretion. This avoids carrying stale solver
  bounds between samples while keeping unselected nutrients closed.
- The generator follows the robust MINN-style process: it loops until the
  accepted sample target is reached, writes through a timestamped temporary CSV,
  reports attempts and feasible rate, has a max-attempt guard, and periodically
  reloads the model/solver.
- The default output prefix remains `iML1515_exp_training_data`, so a default
  500,000-sample run saves `./data/iML1515_exp_training_data_500000_samples.csv`
  unless that file already exists. Use `--overwrite-existing` or a different
  `--output-prefix` deliberately.
- Outputs are all iML1515 reaction fluxes with `"_flux"` suffixes.

Use this generator as the source of truth for input column order, exchange
names, and rate conventions when checking the notebook.

## Shared AMN/MINN Reservoir Data

Cross-task generator, checkpoint, trial, and result state is maintained in
`AMN_MINN_shared_reservoir_notes.md`. Keep this section focused on AMN-specific
implications and update both notes when a shared change affects AMN behavior.

`generate_ecoli_iML1515_AMN_MINN_data.py` is a separate generator for training a
single iML1515 FluxTransformer reservoir that should be usable in both the
AMN-style experimental growth notebook and the MINN Table 4-style reservoir
workflow. It should not replace the strict Faure-style generator when the goal is
to make Faure-faithful claims.

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
    their Faure `2.2` caps, and 1-4 Faure carbon sources are sampled. Oxygen is
    the only additional randomized AMN medium variable relative to Faure.
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
`generate_ecoli_iML1515_AMN_data.py` uses an accepted-sample loop and a fully
reset medium.

## Current Notebook Workflow

The notebook currently has four main parts.

1. Load and inspect a pretrained FluxTransformer.

   The active checkpoint is currently:
   `./models/iML1515_500k_d256_h8_l3_ff1024/iML1515_500k_d256_h8_l3_ff1024_checkpoint.pth`.

   This checkpoint predates the current stable generation behavior in
   `generate_ecoli_iML1515_AMN_data.py`. Do not treat its metrics as results from
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

Current TabPFN baseline:

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
  `generate_ecoli_iML1515_AMN_data.py` or the shared generator used to train the
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
