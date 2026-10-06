# MINN Training Notes

Shared generator, checkpoint, trial, and cross-task result state is maintained
in `AMN_MINN_shared_reservoir_notes.md`. Keep this file focused on MINN-specific
training, mapping, and pFBA behavior, and update both notes when a shared change
affects MINN behavior.

Detailed guide for `ecoli_iML1515_MINN_model_testing.ipynb` and related MINN-style experiments.

## Current MINN metric contract (2026-09-23)

MINN method comparisons must use regression `R2`, calculated with
`sklearn.metrics.r2_score(y_true, y_pred)` (1 - SSE/SST), not squared Pearson
correlation. Tazza adopts the Goncalves metrics; the formula supplied from
`docs/Goncalves2023.pdf` specifies regression R2. For the Table 4-style pFBA
comparison, calculate R2 across the 47 mapped fluxes within each held-out
condition, then report mean and population SD (`ddof=0`) across the 29
conditions. Preserve negative scores. Undefined constant-truth cases must
remain explicit (`force_finite=False`); report any excluded/failed conditions.
Pooled regression R2 is a separate diagnostic, not a replacement for this
per-condition aggregation. Pearson r squared may be reported only under its
own explicit label. MAE, RMSE, NE and the pFBA constraints remain unchanged.

**Paper versus released-code discrepancy (checked 2026-09-27):** Goncalves et
al., Section 2.5, Equation 4 (`docs/Goncalves2023.pdf`, PDF p. 3), explicitly
define R2 as `1 - SSE/SST`. Tazza et al., Section 2.4
(`docs/tazza-minn.pdf`, PDF p. 5), say they use the same metrics and call R2 a
regression coefficient; they do not define it as squared Pearson correlation.
However, `omics2flux/utils.py:r2_metric` and `MINN/src/utils/plots.py:r2_metric`
both return `linregress(y_true, y_pred).r**2` per condition. Thus the papers'
stated definition and the released implementations disagree. For the saved
29-condition iML1515 baseline predictions against the original MINN-fitted
47-flux targets, mean regression R2 is 0.658478 +/- 1.189478, while mean
squared Pearson correlation is 0.892825 +/- 0.132254. The published Tazza
Table 4 pFBA/MINN-reservoir values (0.892/0.910) should be treated as
historical paper-reported R2; numerical proximity to the code's Pearson
metric does not by itself establish how every published result was calculated.
The active `minn_fitted` file is byte-identical to the fitted file selected by
MINN's released reservoir configuration; its precise use behind the published
Table 4 is not independently established. The iML1515 pFBA model still differs
from MINN's reduced iAF1260 model.

This requirement applies to every MINN evaluation of sample spaces A, B,
A-union-B, C, D and E, including the standalone MINN notebook and shared
MINN/AMN trial. It supersedes historical instructions below that preserve
Pearson r squared as Table 4 R2. Old Pearson-labelled-as-R2 results are
historical correlation results and must not be interpreted as regression R2.

The initial 2026-09-23 correction covered only
`ecoli_iML1515_MINN_model_testing.ipynb`. Both of that notebook's pFBA metric
implementations use `r2_score(..., force_finite=False)`. Its context-mode
comparison cell can re-score cached pFBA predictions without retraining.
The shared evaluator's `metrics` and `compare_pfba` paths calculate both
metrics distinctly. The AB-union notebook's final pFBA table now uses those
per-condition results and removes its legacy Pearson-as-R2 table. C/D/E now
use the same corrected final table. Shared-trial notebooks still need their
legacy summary corrected; their old tables do not yet satisfy this contract.

The baseline was recomputed on all 29 conditions and 47 fluxes:
regression R2 **0.658478 +/- 1.189478**, MAE 0.495038 +/- 0.365933,
RMSE 0.832498 +/- 0.633807, NE 0.309400 +/- 0.373326. The old baseline
0.892825 +/- 0.132254 was Pearson r squared. Full predictions for the two
standalone reservoir variants were unavailable; the user will rerun training
and evaluation. No new measured/predicted reservoir scores are claimed.

The standalone biomass diagnostic now selects
`BIOMASS_Ec_iML1515_core_75p37M_flux`; its pFBA objective already used core.
`PLOT_ALL_46_FLUX_DIAGNOSTICS = False`. The removed trainer helper
`prepare_tensors` is replaced locally by the same 80/20 split (`random_state=42`)
and float32 tensor conversion so imports work with the current trainer.
Stale affected outputs are cleared; the baseline was recomputed separately.
Two targeted tests cover regression versus correlation (including negative
scores), constant predictions and experiment-aligned cached rescoring.

## Author MINN nutrient and Table 4 audit (2026-10-06)

The [MINN repository inspection](../reference/minn_repository_notes.md)
records verified author conditions, complete exchange bounds, metric
calculations, and publication limitations at release commit
`c2fc1098313827529b8e3cc3679649f255f16f88`. It confirms the paper/code metric
discrepancy and strengthens its provenance: the released `pFBA_ishii.csv` is
byte-identical to Goncalves's pinned artifact. On its 29 conditions and 45
original signed fluxes, Pearson r squared is **0.822702 +/- 0.156467**,
reproducing Tazza's **Table 2** pFBA row and its MAE/RMSE/NE. Regression R2
on the same predictions is **0.778061 +/- 0.228708**. This is not a Table 4
reproduction; direction conversion to selected positive split magnitudes also
changes correlation and regression R2, so encodings must match for comparisons.

The exact **Table 4** values remain unverified at artifact level. The released
reservoir notebook produces five latent controls but exports only two named
columns (ethanol/acetate), a shape mismatch, and supplies no downstream pFBA
calculation or matched Table 4 predictions. The paper specifies three extra
constraints including CO2. Released metrics use Pearson r squared; applying
that finding to the particular published Table 4 rows is a supported inference,
not a numerical reproduction. Keep the local regression-R2 contract above.

Author conditions are glucose-minimal chemostat media with five WT dilution
rates and 24 KO conditions. The 587-reaction iAF1260-FBA model has glucose/O2
imports, 16 basal imports, and export-only CO2/ethanol/acetate; it has no fixed
glycerol or alanine/proline/threonine/glycine supplements. Basal XML import
bounds are `[0,999999]`, not this repository's base-input value 50. The neural
builder does not enforce every XML bound; with `kos_genes=False`, its only
selected bound is CO2 export with `Vin=inf`. Do not confuse XML availability,
learned context, and actual downstream solver constraints.

The paper describes 2,000 randomized five-channel FBA conditions within
observed Ishii ranges, but no generation script/dataset is released. Exact
sampling distributions, raw/fitted range choice, bound assignments, and plain
FBA versus pFBA pretraining are unknown. Our integer ranges and pFBA at 0.999
are local adaptations. Author AMN instead has a verified 110-pattern × 100-draw
pFBA prior with ten variable sources and fixed organic supplements.

The released reservoir loader uses fitted glucose/O2 as features and only five
fitted exchange targets. Fitting changes glucose in 28/29 conditions, O2 in
18/29, and CO2 in all 29. Fitted maxima are O2 **26.2417** and CO2 **19.6169**,
beyond the local B/shared caps 20 and 15. The exact original fitting code is
unavailable. The inherited Goncalves pFBA baseline instead fixes raw glucose/O2
to equalities and applies KOs; its `gapC`-position/b0118 (`acnB`) mapping is
unresolved. These input/constraint differences require explicit scope when
interpreting baseline or transfer results. All detailed evidence is in the audit.

## 1) Core goal
- Use a pretrained `FluxTransformer` as a frozen reservoir.
- Train only a front MLP on omics + measured inputs.
- By default, front MLP predicts latent reservoir context/control channels for CO2, ethanol, and acetate.
- The AMN-trial notebook compares both glucose/oxygen context modes: `"measured"` copies measured glucose/oxygen into the FluxTransformer context, while `"predicted"` asks the front MLP to predict all five context channels. Both downstream branches apply only the same three CO2/ethanol/acetate caps.
- Reconstruct full transformer input as:
  - measured glucose/oxygen copied directly into their context tokens
  - latent predicted CO2/ethanol/acetate context/cap channels
  - fixed `base_exchanges`
  - zeros for other channels
- Run transformer forward with full vocab output.
- For FluxTransformer->pFBA, keep glucose/oxygen as measured inputs. Downstream extra constraints are selected by `MINN_PFBA_EXTRA_CONSTRAINT_MODE`; the reservoir training/FluxTransformer input context still includes CO2 regardless of mode.
- The standalone notebook comparison uses all three predicted CO2, ethanol, and acetate secretion caps.

## 1.1) Simulated MINN data generation file
- Simulated MINN-style training data for FluxTransformer is generated in: `generate_ecoli_iML1515_MINN_data.py`.
- Use this file as the sign/bound/order ground truth when validating notebook mappings.
- The current generator samples glucose and oxygen uptake constraints, leaves CO2/ethanol/acetate uncapped in the secretion direction, and fills their input/context columns from realized pFBA secretion fluxes after solving.
- In `generate_ecoli_iML1515_MINN_data.py`, non-variable base exchanges are fixed as medium-availability inputs with `lower_bound=-default_rate`. Glucose and oxygen are variable uptake lower bounds. CO2/ethanol/acetate are special secretion-context exchanges: `lower_bound=0`, `upper_bound` left at the nonnegative model default, then their input values are overwritten from the solved pFBA flux.
- `generate_ecoli_iML1515_AMN_MINN_data.py` is the shared AMN/MINN reservoir generator for retraining a checkpoint that should work in both the AMN growth notebook and the MINN Table 4-style workflow. It adds glucose and ethanol to the Faure-style AMN input set, always includes the five MINN context tokens, and mixes `minn`, `faure`, and `mixed` regimes. The `minn` and `mixed` regimes independently sample the same widened integer caps as the Tazza-style generator: glucose 1--15, oxygen 1--20, CO2 0--15, ethanol 0--1, and acetate 0--3. All shared regimes use non-carbon base nutrient rate 50. Use this for AMN/MINN transfer checkpoints, not for strict Faure-only claims.
- The AMN-trial MINN notebook showed why a strict no-glucose AMN checkpoint is not enough for Table 4: `EX_glc__D_e` was absent from the AMN training inputs and effectively constant zero in the simulated AMN data, while MINN uses measured glucose uptake as a central context channel.
- `generate_ecoli_iML1515_MINN_data_tazza.py` is the separate Tazza-style ablation generator. It uniformly draws integer caps from deliberately widened, rounded envelopes around the raw Ishii ranges: glucose 1--15, oxygen 1--20, CO2 0--15, ethanol 0--1, and acetate 0--3. Glucose/oxygen are uptake caps; CO2/ethanol/acetate are secretion upper caps. The sampled caps remain in the input/context columns, while realized pFBA values are written to `*_flux`. This is closer to Tazza et al.'s random five-channel reservoir `Vin` setup without restricting training to the exact experimental extrema, and it intentionally retains this repo's iML1515 model, 500k-sample scale, and pFBA target generation.
- The Tazza generator accepts the same portable run controls used by the shared generator, including `--n-samples`, `--model-dir`, `--data-dir`, `--output-prefix`, solver timeout/reset controls, bounded retry controls, and cap-range overrides. Its defaults remain 500k samples, seed 42, core biomass, pFBA at 0.999 of optimum, and the cap ranges above. Final files are named `<output-prefix>_<n-samples>_samples.csv`; timestamped temporary files are retained if generation fails.
- `scripts/fit_minn_fluxomics_minn_like.py` is the maintained iML1515 refitting script for the 29-sample MINN split fluxomics file. It starts from `MINN_data/fluxomics_iAF1260_reduced_split.csv`, uses `models/iML1515.xml`, fixes biomass by default, and writes `MINN_data/fluxomics_iML1515_minn_like_fit.csv`.
- The maintained fitting policy is MINN-like: glucose/O2 have weighted deviation terms but no hard soft-input band by default, because this matched the original MINN fitted-file behavior better than the stricter soft-input-band trial. The original fitted file `MINN_data/fluxomics_iAF1260_reduced_split_fit.csv` is only a descriptive audit reference, not a target to reproduce; use `--compare-output-to-reference` only when that extra comparison is explicitly wanted.

## 2) Data and feature setup
- Training/eval notebook: `ecoli_iML1515_MINN_model_testing.ipynb`.
- MINN-style data directory: `./MINN_data`.
- `MINN_FLUXOMICS_FILE_MODE` is the notebook fluxomics-file switch. It sets `MINN_FLUXOMICS_FILE` from `MINN_FLUXOMICS_FILE_OPTIONS`. The current active default is `minn_fitted`.
  - `minn_fitted`: `fluxomics_iAF1260_reduced_split_fit.csv`; original MINN fitted/Table 4-comparable file
  - `non_fitted`: `fluxomics_iAF1260_reduced_split.csv`; non-fitted split source robustness file
  - `iml1515_minn_like`: `fluxomics_iML1515_minn_like_fit.csv`; iML1515 fit with MINN-like soft exchange movement; preferred iML1515-specific fitted file
- Inputs include:
  - transcriptomics
  - proteomics
  - measured flux inputs (`R_EX_glc__D_e_rev`, `R_EX_o2_e_rev`)
- Full 5-channel reservoir context source order:
  - `R_EX_glc__D_e_rev`
  - `R_EX_o2_e_rev`
  - `R_EX_co2_e_fwd`
  - `R_EX_etoh_e`
  - `R_EX_ac_e`
- The first two channels are measured/copied by default; only CO2, ethanol, and acetate are predicted by the front MLP unless `MINN_GLC_O2_CONTEXT_MODE="predicted"`.
- Transformer-output training targets exclude these 5 context columns by default (`MINN_FLUX_TARGET_EXCLUDE_CONTEXT=True`) so context/cap outputs are latent controls, not exact exchange-flux regressions.
- Downstream pFBA extra-constraint mode: `co2_etoh_ac_cap` applies predicted `R_EX_co2_e_fwd`, `R_EX_etoh_e`, and `R_EX_ac_e` as secretion upper caps.

## Standalone dual-fit pFBA comparison (2026-09-27)

`ecoli_iML1515_MINN_model_testing.ipynb` now keeps the original MINN-fitted
file as the main target/input file. Its final table contains pFBA, measured
context FT+pFBA with three caps, and predicted context FT+pFBA with three
caps. The ethanol/acetate-only runs are removed. Each row reports mean and
population SD across successful conditions for regression R2 and squared
Pearson r, with distinct labels and successful-condition coverage.

A separate iML1515-fitted sensitivity test uses
`fluxomics_iML1515_minn_like_fit.csv` for both measured uptake inputs and
47-flux targets. It recomputes pFBA and retrains the measured-context front
MLP with the same HPO/LOO protocol before its three-cap FT+pFBA evaluation.
The final five-row table is the last cell. A direct run of the notebook's
baseline helper solved all 29 conditions for each fitted file. On the
iML1515 fit, mean regression R2 is 0.916927 +/- 0.067616 and mean squared
Pearson r is 0.925047 +/- 0.060296. On the original MINN fit, the same
check reproduces 0.658478 +/- 1.189478 and 0.892825 +/- 0.132254.
The expensive front-MLP runs have not been rerun, so no new FT+pFBA scores
are claimed. The iML1515-fit scores measure agreement with different fitted
targets and are not a direct Table 4 reproduction.

## AB-union/C/D/E dual-fit pFBA comparison (2026-09-28)

`ecoli_iML1515_AB_union_model_testing.ipynb` now ends with the same five
comparison methods as the standalone notebook: original-fit baseline,
measured and predicted three-cap FT+pFBA; iML1515-fit baseline and measured
three-cap FT+pFBA. The original target mode is required. The separate
iML1515-fit branch checks experiment, feature, target, and mapping order,
then retrains measured context using that file and the existing 50-trial HPO
and 29-fold LOO protocol. The existing `ev.run_pfba` per-condition metrics
provide regression R2, squared Pearson r, MAE, RMSE and NE over all 47 mapped
fluxes; the final table reports mean and population SD over successful
conditions plus defined R2/correlation counts. It verifies fitted-file truth
and sample alignment before reporting. The old AB-union Pearson-as-R2
summary is removed, while its separate pooled regression table remains.
The full AB-union notebook has not been rerun; no new FT+pFBA scores are
recorded here.

The C, D and E evaluation notebooks now use the same five-row comparison,
separate iML1515-fitted measured-context retraining, fitted-file provenance
checks, per-condition metric aggregation and defined-score coverage as
AB-union. They retain their own checkpoint and input schemas: the configured
C checkpoint has 40 inputs, while D and E each have 55. Their original and
iML1515-fitted files align on all 29 experiment IDs and 47 source mappings.
Notebook structure and Python syntax pass; no C/D/E HPO, leave-one-out fit or
new FT+pFBA score has been run for this change.

### Why the two fitted-file baselines differ (checked 2026-09-27)

The raw split file and both fitted files have the same 29 experiments and 47
flux columns in the same order. The original fitted CSV is byte-identical to
`MINN/data/ishii_data/fluxomics_iAF1260_reduced_split_fit.csv`; no local copy
or row-alignment error was found. Re-solving `WT_0.1h-1` with the local
iML1515 fitter reproduces its saved row within 1.8e-6. Tazza Section 2.1
(`docs/tazza-minn.pdf`, PDF p. 2) describes its FBA-fit data as a minimum
Euclidean-distance projection. The original fitting implementation is absent
here. `scripts/fit_minn_fluxomics_minn_like.py` instead minimizes **weighted
absolute deviation** (L1), with biomass fixed and glucose/O2, other exchange,
and internal weights 1000, 100, and 1. The two fits therefore do not use the
same known objective or GEM.

`WT_0.1h-1` explains 87.8% of the 0.258449 gap between mean per-condition
regression R2 values. Its original-fit R2 is -5.608750 (SSE 550.994 versus
SST 83.373); its iML1515-fit R2 is 0.970738. Omitting this condition gives
0.882308 for the original fit and 0.915005 for the iML1515 fit. The next
weakest original-fit condition is `fbaB` at R2 0.430072; its iML1515-fit R2
is 0.973802. All 29 conditions solve in both baseline runs.

For `WT_0.1h-1`, raw glucose/O2 uptake magnitudes are 1.34/2.20. The
original fit moves them to 5.8620/6.8486, whereas the iML1515 fit moves
them to 2.0280/3.1644; both fitted biomass targets remain 0.1000. The
notebook treats uptake values as *caps* and maximizes iML1515 biomass before
pFBA. It consequently predicts biomass 0.320904 and substantially excessive
glycolysis/acetate fluxes against the original fit, but biomass 0.100000 and
closely matched fluxes against the iML1515 fit. On this condition, fixing
biomass to 0.1000 while retaining original fitted uptake caps improves R2 to
0.4173; also fixing uptake fluxes gives 0.5698. Thus the mismatch includes
the biomass objective and remaining flux-solution differences. The included
reduced iAF1260 model also scores poorly (-5.3490) under the same simplified
uptake-cap/biomass-maximization setup, so switching GEM alone does not resolve
this example. That counterfactual is not a reproduction of MINN's full
published evaluation protocol.

The higher iML1515-fit R2 does **not** mean that fit is closer to the raw
measurements: across all 29 x 47 entries its mean absolute raw-to-fit change
is 0.4623, versus 0.2134 for the original MINN fit. The comparison changes
both pFBA uptake inputs and evaluation targets, while fitting and evaluation
use the same iML1515 GEM. The original file is an authentic upstream result;
the local evidence does not identify a typo or explain why its source fitting
procedure selected the high `WT_0.1h-1` glucose uptake. This is a protocol
and target-compatibility issue, not evidence that 0.916927 reproduces Tazza's
original benchmark.

## 3) Mapping/sign conventions (critical)
- Keep explicit source->token mapping and signs consistent with generation scripts.
- `*_rev` source columns are positive magnitudes in the split data but represent uptake direction.
- During training, full context magnitudes are recovered from signed context targets using `context_signs` and clamped to nonnegative magnitudes; measured glucose/oxygen are copied into the FluxTransformer input context by default.
- Context/cap outputs are latent controls only; the active notebook does not use an exact cap-loss term or weight sweep.
- During FluxTransformer->pFBA evaluation, do not use front-MLP predictions for glucose/oxygen; use the measured `R_EX_glc__D_e_rev` and `R_EX_o2_e_rev` columns.
- For pFBA constraints in COBRA, apply correct bound direction/sign (especially for uptake-style exchanges).

## 3.1) Transformer source file
- `FluxTransformer` class is defined in: `flux_transformer.py`.
- When validating forward behavior or input/output shape assumptions, check this file first.
- For FluxTransformer-based MINN experiments in this repo, `iML1515` is the correct model/GEM context. Do not switch the FluxTransformer pFBA path to the paper's iAF1260-FBA reduced GEM just because Table 4 used it.
- FluxTransformer uses the unsplit, unpruned iML1515 reaction/token vocabulary. Do not split reversible reactions or prune reactions when mapping FluxTransformer inputs, outputs, or pFBA reactions.

## 4) Model architecture
- Wrapper: frozen transformer + trainable front MLP.
- The active notebooks use the historical one-hidden-layer, width-512 ReLU
  front MLP. The smaller two-stage normalized-loss trial was rejected because
  it worsened both direct reservoir and FluxTransformer-to-pFBA metrics.
- Front-MLP context outputs use `softplus` to produce nonnegative latent context/cap values.
- Transformer remains frozen (no optimizer params from transformer).
- Reservoir forward passes must still use the full output vocabulary
  (`output_subset=None`). Output subsetting changes the attention token set and
  is not a valid speed optimization for this workflow.

## 5) Training protocol
- Outer CV: Leave-One-Out (LOO).
- Inner CV: KFold hyperparameter evaluation.
- HPO:
  - `drop_rate`, `learning_rate`, `weight_decay`
  - one-time global HPO mode is available (`MINN_HPO_ONCE=True`) to reduce runtime.
  - current default trials: `minn_cv_max_trials=50`
  - current drop-rate search space: `[0.0, 0.05, 0.1, 0.2, 0.3, 0.35, 0.4]`
  - current learning-rate search range: `5e-4` to `1e-2` log-sampled
  - fixed run mode is available with `MINN_USE_FIXED_HYPERPARAMS=True`; this skips Optuna trials and uses `MINN_FIXED_BEST_PARAMS` directly.
  - current fixed best hyperparameters: `{"drop_rate": 0.25, "learning_rate": 0.0009736777696601047, "weight_decay": 4.738391288843704e-05}`
  - best hyperparameters must be printed immediately after HPO trials complete
  - objective is stability-aware:
    - `mean(inner_fold_val_loss)`
    - `+ MINN_INNER_CV_STD_PENALTY * std(inner_fold_val_loss)`
- Loss:
  - flux loss on transformer target channels (`y_minn_np`) that exclude the 5 context columns by default
  - no separate exact cap/context loss; front-MLP context outputs are learned only through the frozen FluxTransformer target-flux loss
  - the active AMN/MINN trial uses raw, unnormalized Huber loss
- Training stabilization:
  - gradient clipping (`MINN_GRAD_CLIP_MAX_NORM`)
  - LR warmup + cosine decay (`MINN_LR_WARMUP_EPOCHS`, `MINN_LR_COSINE_MIN_FACTOR`)
- AMP:
  - use `torch.amp.autocast(...)`
  - use `torch.amp.GradScaler(...)`

### 5.1) AMN/MINN trial legacy pipeline

`ecoli_iML1515_MINN_AMN_model_testing_trial.ipynb` uses the legacy front-MLP
workflow restored from commit `6b1e3bf`, with the
`AMN_MINN_500k_d256_h8_l3_ff1024`
FluxTransformer checkpoint, `minn_fitted` data mode, context-target exclusion,
full-vocabulary reservoir forward, and `co2_etoh_ac_cap` pFBA mode.

Its active settings are:

- one hidden layer with width 512
- ReLU activation
- raw, unnormalized Huber loss
- 50 HPO trials evaluated with the full 150-epoch protocol
- historical CUDA batch configuration (requested batch 5, reduced to 2)

The predicted-context result (`R2=0.895539`, `MAE=0.486514`) slightly beats
baseline pFBA (`R2=0.892825`, `MAE=0.495038`). Preserve this configuration as
the reference when testing further improvements. The former two-layer GELU,
per-flux normalized-loss pipeline is retained only in Git history.

## 6) Early stopping (implemented)
- Config keys:
  - `minn_cv_early_stopping_patience`
  - `minn_cv_early_stopping_min_delta`
- Current default: `minn_cv_early_stopping_patience=25`
- Behavior:
  - evaluate validation loss each epoch
  - keep best front-MLP state
  - stop after patience with no improvement
  - restore best state before final validation prediction export

## 7) Outputs to preserve
- OOF transformer-target predictions (`oof_pred`) and truths (`oof_true`).
- OOF full 5-channel reservoir context (`oof_pred_constraints`) for diagnostics and FluxTransformer+pFBA cells; glucose/O2 columns are measured copies by default, CO2/ethanol/acetate columns are latent front-MLP predictions.
- OOF observed context magnitudes (`oof_true_constraints`) for diagnostics only.
- Per-LOO metrics (R2/MAE/RMSE/NE).
- Compact Optuna trials table (`last_optuna_trials_df`).
- Per-LOO validation-loss table (`loo_val_loss_df`).
- Per-LOO R2/RMSE panel plot: save to `./pics/minn_loo_metric_panels.png` by default and still show it in the notebook; set `MINN_LOO_METRIC_PLOT_SAVE_PATH=None` for display-only behavior.
- FluxTransformer->pFBA cap-binding diagnostics:
  - `pfba_cap_binding_diagnostics_df`
  - `pfba_cap_binding_sample_summary_df`
  - `pfba_cap_binding_result_summary_df`
  - the diagnostic section aggregates `MINN_CAP_BINDING_DIAGNOSTICS_DF`, `MINN_CAP_BINDING_SAMPLE_SUMMARY_DF`, and `MINN_CAP_BINDING_RESULT_SUMMARY_DF`
- Epoch diagnostics:
  - per-LOO `epochs_trained`
  - summary of mean/min/max trained epochs

## 8) pFBA evaluation notes
- Baseline pFBA and FluxTransformer+pFBA should be separate sections.
- Keep experiment order aligned exactly with OOF rows when merging constraints.
- Ensure token indices and iML1515 reaction order are unchanged.
- Use `models/iML1515.xml` for FluxTransformer->pFBA evaluation in the MINN notebook.
- The iML1515 SBML default pFBA objective is `BIOMASS_Ec_iML1515_core_75p37M`; use this for the Table 4-style experimental pFBA comparison and the simulated-data generator unless intentionally running a separate WT-objective experiment.
- Always print the chosen pFBA objective, and map the biomass metric to the same iML1515 biomass reaction used as the pFBA objective.
- Keep FluxTransformer pFBA mapping in the unsplit/unpruned reaction space; only convert MINN split source-column signs into the corresponding unsplit iML1515 bounds or flux signs.
- FluxTransformer->pFBA should use measured glucose/oxygen uptake as lower-bound uptake caps, not exact fixed fluxes.
- `MINN_PFBA_EXTRA_CONSTRAINT_MODE="co2_etoh_ac_cap"` applies predicted CO2, ethanol, and acetate as nonnegative secretion upper caps. This is the only active cap set in the AMN-trial notebook.
- Predicted nonnegative secretion caps use `lower_bound=max(0, min(current_lower_bound, prediction))` and `upper_bound=max(0, prediction)`.
- Do not include cap-calibration trials in the final comparison by default; direct cap-MAE calibration can over-shrink upper caps, and the pFBA-tuned safety calibration selected identity scales in testing.
- The final comparison cell reports the three original MINN-fitted rows plus iML1515-fitted pFBA and measured-context FT+pFBA, all with the `co2_etoh_ac_cap` cap set where applicable.
- Keep the per-sample cap-binding diagnostic cell before the final comparison. It separates bad cap prediction from pFBA overconstraint by checking whether each predicted secretion cap binds, whether it is below the fitted target (`binding_low_cap`), and whether the cap improves or worsens the fitted-target error versus baseline pFBA.
- Verify feasibility counts and print failed samples for debugging.
## 8.1) Table 2 benchmark notebook
- Table 2-style benchmarks are now in the separate notebook: `ecoli_iML1515_MINN_Table2.ipynb`.
- `ecoli_iML1515_MINN_model_testing.ipynb` should not be described as evaluating Table 2 metrics; its active experimental comparison is the Table 4-style iML1515 pFBA workflow with FluxTransformer-to-pFBA variants and cap-binding diagnostics.
- The Table 2 notebook mirrors the Goncalves/ML2Flux benchmark format used in Tazza et al. Table 2.
- Use `MINN_data/fluxomics.csv` for the original signed Ishii/Goncalves flux targets, not the split/FBA-fit MINN fluxomics file.
- Inputs:
  - transcriptomics
  - proteomics
  - measured uptake fluxes (`R_EX_glc_e_`, `R_EX_o2_e_`)
- Targets:
  - all fluxomics columns except the two fixed uptake fluxes
  - expected shape: 45 target fluxes over 29 samples
- Table 2 notebook rows include:
  - TabPFN ML-to-flux benchmark (separate from the standalone AMN notebook,
    which now explicitly selects TabPFN-3.5; this MINN notebook is unchanged)
  - Goncalves-style pFBA baseline recomputed with `models/iML1515.xml`
  - MLP + frozen FluxTransformer ML-to-flux benchmark
  - final Table 2-style comparison against published Tazza rows
- Protocol:
  - Leave-One-Out CV using `KFold(n_splits=len(X), shuffle=True, random_state=12345)` where applicable
  - fit scalers inside each fold only
  - fit one `TabPFNRegressor` per target flux because TabPFN regression is single-output
  - report R2, MAE, RMSE, and NE as mean +/- std across LOO samples, Table 2 style

## 8.2) Goncalves-style pFBA in the Table 2 notebook
- `ecoli_iML1515_MINN_Table2.ipynb` includes the Goncalves/omics2flux Ishii pFBA baseline adapted to `models/iML1515.xml`.
- Keep the Goncalves protocol:
  - fixed measured glucose and oxygen uptake from `MINN_data/fluxomics.csv`
  - pFBA over the same 45 non-uptake Table 2 flux targets
  - R2, MAE, RMSE, and NE as mean +/- std across the 29 Ishii samples
- Important mapping detail:
  - local `fluxomics.csv` rows use gene-symbol sample names
  - `omics2flux/pfba.py` uses an ordered b-number knockout list
  - therefore the Table 2 notebook must map sample names to the original Goncalves b-numbers before knockout
- iML1515 does not contain the original Goncalves `b4395` gene used for the `gpmB` sample. The adapted iML1515 benchmark maps it to the iML1515 PGM isozyme `b3612` so all 29 samples solve.
- Sanity check: the iML1515 Goncalves-style pFBA cell should report `Successful pFBA samples: 29/29` and an empty failed-sample table.

## 9) Common failure modes
- CUDA OOM in diagnostics/training:
  - lower batch size
  - keep full-output forward where required
- Mapping mismatch between source columns and output tokens.
- Wrong sign when converting predicted magnitudes to pFBA bounds.
- Misaligned experiment ordering when attaching OOF predictions.
- Missing TabPFN benchmark dependencies in the active Python environment (`pandas`, `scikit-learn`/`sklearn`, `tabpfn`) when running `ecoli_iML1515_MINN_Table2.ipynb`.

## 10) Recommended sanity checks
- Assert all required token names exist in `outputs`.
- Print source->token mapping with signs.
- Print c0/cap value versus observed-context magnitude summaries for the 5 context channels.
- Check OOF sample count equals dataset count in LOO context.
- Confirm pFBA evaluated sample count and metric table shape.
- Confirm FluxTransformer->pFBA `pred_vin_df` contains only the predicted extra constraints for the selected mode, never predicted glucose/oxygen.
- For the Table 2 notebook, confirm the benchmark prints 29 samples, 141 features, and 45 targets.

## A union B combined notebook (2026-09-08)

`ecoli_iML1515_AB_union_model_testing.ipynb` independently trains measured- and
predicted-glucose/O2 context MLPs, in addition to its separate AMN MLP. The
primary data has 29 conditions, 141 features and 42 non-context training
targets; pFBA retains all 47 mapped source fluxes plus a non-context summary.
The union B neural context uses base 50 and cobalamin, with A-only inputs zero.

HPO and feature scaling are confined to outer-training data. Each LOO refit
uses the winning trial's median inner best epoch, with no outer-test early
stopping or global HPO. Both modes use observed glucose/O2 pFBA uptake caps
and OOF secretion caps. All pFBA methods retain the same SBML background
medium and explicit 0.999 fraction. Optional target-file and ethanol/acetate
cap-only sensitivities are paired across both modes. No TabPFN tests are added.

The legacy pFBA `R2` metric is squared Pearson correlation. The new helper
exports `Pearson_r2` separately from regression `R2`, preserves undefined
metrics, and reports own-success/common-success counts and failed samples.
Current main/trial notebooks are unchanged; no production union result is
recorded. See the sampling-study note for implementation scope and provenance.

## Union MINN numerical stability (2026-09-12)

A reported MINN trial stopped at gradient clipping because its gradient norm
was nonfinite with CUDA AMP enabled. The trace establishes gradient failure;
mixed-precision overflow is a suspected cause, not a reproduced diagnosis.
The union notebook now defaults to FP32. If AMP is explicitly enabled, a
nonfinite fit restarts from its original seed in FP32 before any result is
accepted. Invalid gradients never update weights. Numerical HPO failures are
recorded as failed trials and do not terminate remaining trials; if all fail,
the run stops explicitly. Fit histories record actual AMP use.

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

Model C combined evaluation (2026-09-14): `ecoli_iML1515_C_model_testing.ipynb` uses `iml1515_evaluation.py` with explicit C input schema and basal 50. The configured legacy 40-input checkpoint excludes cobalamin; it is not injected. Existing union defaults and training protocols are unchanged. See `iML1515_sampling_study_notes.md` for details.

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
HPO/LOO, and fresh pFBA model copies per condition. The final table includes
the original per-condition Pearson-r-squared/MAE/RMSE/NE mean and population SD;
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

## Standalone MINN distribution-shift overall metrics (2026-10-02)

`ecoli_iML1515_MINN_model_testing.ipynb` now places an A/E distribution-shift
test immediately after its existing in-distribution overall-metrics cell. It
scores all 50,000 rows of the independent AMN A and general E test CSVs against
the 2,712 flux outputs in the MINN checkpoint's exact order. Each source
nutrient bound is placed at its matching `*_flux` context token, including
tokens outside MINN's 27 declared training inputs; remaining context slots
are zero. Full-output inference uses `output_subset=None`. This mapping is
possible because the checkpoint carries a shared 2,712-token reaction
vocabulary; declaring only 27 training inputs does not mask other context
positions in a full-output forward pass.

A contributes 38 source inputs: 24 shared with MINN, 14 additional, while
MINN-trained glucose, ethanol and cobalamin inputs are absent and zero-filled.
E contributes 55 inputs: all 27 MINN inputs plus 28 additional channels. The
cell reports pooled regression R2, MAE and RMSE over every row and flux,
and records source-input activity, zero-filled inputs and the token mapping
in its protocol JSON. The current MINN diagnostic overall metrics instead
use the 20% split of its Tazza test CSV. A uses the AMN FBA target policy and
E uses its broader pFBA sampling policy, so the shift scores mix context
generalization with different target generation. The full A/E score runs
are left for notebook execution. Notebook schema and code syntax passed;
one real-checkpoint row from each source verified token placement and finite
metrics. A two-row A check matched pooled R2, MAE and RMSE against scikit-learn
on the same predictions.

## MINN reservoir on the 110-point AMN growth task (2026-10-03)

The standalone MINN notebook now includes a cross-task growth-transfer control
after its t-SNE plots, before MINN's own front-MLP experiment. It uses the same
110 `AMN_data/iML1515_EXP.csv` media, `GR_AVG` targets, aligned `GR_STD` from
`AMN_data/EXP110.csv`, front-network architecture, 10-fold carbon-count
stratification, split seeds 10/11/12, training seed 10, Huber delta 0.03,
batch sizes 1/2, 100-epoch limit and patience 15 as the standalone AMN
notebook. The only intended model change is the frozen
`MINN_1M_d256_h8_l4_ff1024` reservoir. The MLP learns ten carbon uptake
caps at most 2.2 plus oxygen at most 10; AMN fixed basal bounds are 10 and
glycerol/amino-acid bounds 2.2.

All 38 AMN medium inputs are injected at their matching tokens in the full
2,712-token MINN vocabulary. Fourteen AMN sources lie outside MINN's 27
declared training inputs; glucose, ethanol and cobalamin are absent from the
AMN medium and remain zero. The code averages one held-out-fold prediction
per sample across the three split seeds, prints pooled regression R2, MAE and
RMSE over 110 samples, and saves one true-versus-predicted plot with measured
horizontal and split-seed vertical standard-deviation bars. The validation
fold also chooses the epoch, following the reference AMN notebook; the
result is a legacy out-of-fold comparison, not a strict untouched-test score.
The comparison includes input-coverage and bound-policy shifts: in particular,
AMN uses acetate as an uptake cap, while MINN trained it as a secretion cap.

The notebook's original 63 cells and outputs were preserved. Schema and code
syntax passed; a real-checkpoint one-step front-MLP smoke run verified input
mapping and finite growth output. Mocked 30-fold aggregation populated all
110 OOF rows and rendered the two-error-bar plot. The full 30-fold training
run and final metrics remain pending notebook execution.

## Fitted sensitivity restart dependency (2026-10-04)

The iML1515-fitted measured-context sensitivity cell no longer requires the
original measured/predicted FT+pFBA comparison to have run in the current
kernel. After a restart, run the MINN setup and reusable training/evaluator
definition cells, the original baseline pFBA cell (which defines `flux_df`,
the 47-flux mapping and `evaluate_pfba_baseline`), and the iML1515-fitted
baseline cell before the sensitivity cell. The sensitivity cell still performs
its own fresh HPO, LOO training and downstream pFBA as configured; it restores
the pre-run dataset and context state whether or not an original comparison
result is available. The final cross-file comparison table still requires the
original measured and predicted FT+pFBA results. This change avoids rerunning
those expensive original fits merely to reach the fitted sensitivity test.
