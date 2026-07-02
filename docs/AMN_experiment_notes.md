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
- `generate_ecoli_iML1515_AMN_data.py`: simulated iML1515 data generator used to
  create Faure-style FluxTransformer training data.
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

`generate_ecoli_iML1515_AMN_data.py` generates iML1515 FBA samples with media
settings chosen to resemble the Faure experimental setup.

Key points:

- The model is `models/iML1515.xml`.
- Variable carbon sources are selected from the Faure-style carbon set:
  ribose, maltose, melibiose, trehalose, fructose, galactose, acetate,
  D-lactate, succinate, and pyruvate.
- Glycerol is fixed as an additional carbon source.
- Base medium exchanges include phosphate, CO2, protons, water, ammonia,
  oxygen, ions, sulfate, sodium, chloride, and trace elements.
- Alanine, proline, threonine, and glycine are treated as fixed amino-acid
  exchanges.
- The current generator uses uptake rates around the Faure experimental scale:
  variable carbon and amino-acid rates use `2.2`, while many base nutrients use
  `10.0`.
- Oxygen is variable in the current generator implementation, even though the
  Faure-style medium comments refer to a fixed oxygen setting. Keep this in mind
  when comparing against the paper.
- Outputs are all iML1515 reaction fluxes with `"_flux"` suffixes.

Use this generator as the source of truth for input column order, exchange
names, and rate conventions when checking the notebook.

## Current Notebook Workflow

The notebook currently has three main parts.

1. Load and inspect a pretrained FluxTransformer.

   The active checkpoint is currently:
   `./models/iML1515_500k_d256_h8_l3_ff1024/iML1515_500k_d256_h8_l3_ff1024_checkpoint.pth`.

   The simulated test data loaded for FluxTransformer diagnostics is currently:
   `./data/iML1515_test_data_50000_samples.csv`.

2. Evaluate simulated-data flux predictions.

   The notebook plots FluxTransformer diagnostics for selected fluxes, including
   the iML1515 biomass reaction:
   `BIOMASS_Ec_iML1515_core_75p37M_flux`.

3. Train a prior dense network on experimental media and growth data.

   Experimental data is loaded from:
   `./experimental_data/iML1515_EXP.csv`.

   The target column is:
   `GR_AVG`.

   The prior network receives the variable medium inputs and predicts bounded
   uptake values for the same variable input positions. These predicted medium
   values are inserted into the full FluxTransformer input tensor. Fixed medium
   components are kept at generator-aligned rates.

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

`experimental_data/EXP110.csv` provides `GR_STD`, which is used for plotting
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
- Early stopping patience: `13`.
- Base batch size: `1`.
- Cross-validation: 10-fold stratified CV repeated with split seeds
  `[42, 43, 44]`.
- Final full-data fit: ensemble seeds `[42, 43, 44]`.

The dense prior predicts nonnegative bounded rates. Carbon-source outputs are
bounded by `2.2`; oxygen is bounded by `10.0`.

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

Flux diagnostics are saved under the date/model-specific `pic_dir` used in the
notebook.

## Sanity Checks

- Confirm that the checkpoint output token order matches the loaded CSV columns.
- Confirm that all experimental input columns exist after removing the `"_i"`
  suffix.
- Confirm that binary carbon-source features remain 0/1 before TabPFN or
  stratified CV.
- Check that fixed medium rates in the notebook match
  `generate_ecoli_iML1515_AMN_data.py`.
- Print and review the variable input columns learned by the prior ANN.
- Keep simulated-data FluxTransformer diagnostics separate from experimental
  growth-rate evaluation.
- Do not describe the current experiment as an exact reproduction of Faure et
  al.; it is a FluxTransformer-reservoir adaptation of the AMN idea.

## Open Questions

- Whether the prior dense network should predict only variable carbon rates or
  also selected fixed/media context channels.
- Whether oxygen should remain trainable or be fixed to the Faure-style medium
  setting.
- Whether the FluxTransformer checkpoint should be retrained only on AMN-style
  simulated media or mixed with broader iML1515 media.
- Whether final reporting should compare against Faure AMN-QP/LP/Wt numbers,
  TabPFN only, or additional pFBA baselines.
