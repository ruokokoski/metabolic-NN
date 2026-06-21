# AGENTS.md

## MINN training quick guide (ecoli_iML1515_MINN_model_testing.ipynb)

This document captures the main points an agent should follow when working on the MINN-style training/evaluation pipeline in this repo.

## 1) Core goal
- Use a pretrained `FluxTransformer` as a frozen reservoir.
- Train only a front MLP on omics + measured inputs.
- By default, front MLP predicts latent reservoir context/control channels for CO2, ethanol, and acetate.
- `MINN_GLC_O2_CONTEXT_MODE="measured"` is the default: measured glucose/oxygen are copied into the FluxTransformer context. Set it to `"predicted"` only for the older ablation where the front MLP predicts all 5 context channels.
- Reconstruct full transformer input as:
  - measured glucose/oxygen copied directly into their context tokens
  - latent predicted CO2/ethanol/acetate context/cap channels
  - fixed `base_exchanges`
  - zeros for other channels
- Run transformer forward with full vocab output.
- For FluxTransformer->pFBA, keep glucose/oxygen as measured inputs. Downstream extra constraints are selected by `MINN_PFBA_EXTRA_CONSTRAINT_MODE`; the reservoir training/FluxTransformer input context still includes CO2 regardless of mode.
- `etoh_ac_cap` means CO2 remains a predicted FluxTransformer context/input channel, but its pFBA upper cap is not applied. Only ethanol and acetate are constrained downstream.

## 1.1) Simulated MINN data generation file
- Simulated MINN-style training data for FluxTransformer is generated in: `generate_ecoli_iML1515_MINN_data.py`.
- Use this file as the sign/bound/order ground truth when validating notebook mappings.
- The current generator samples glucose and oxygen uptake constraints, leaves CO2/ethanol/acetate uncapped in the secretion direction, and fills their input/context columns from realized pFBA secretion fluxes after solving.
- In `generate_ecoli_iML1515_MINN_data.py`, non-variable base exchanges are fixed as medium-availability inputs with `lower_bound=-default_rate`. Glucose and oxygen are variable uptake lower bounds. CO2/ethanol/acetate are special secretion-context exchanges: `lower_bound=0`, `upper_bound` left at the nonnegative model default, then their input values are overwritten from the solved pFBA flux.
- `generate_ecoli_iML1515_MINN_data_minn_mimic.py` is the separate MINN-mimic ablation generator. It randomly caps CO2/ethanol/acetate secretion during simulation and writes those sampled caps to the input/context columns instead of overwriting them with realized pFBA secretion fluxes. This is less biologically natural than leaving secretion products unconstrained, but closer to the paper's described random reservoir `Vin` setup.
- `fit_minn_fluxomics_soft_inputs.py` is a separate experimental fitter for the 29-sample MINN split fluxomics file. It starts from `MINN_data/fluxomics_iAF1260_reduced_split.csv`, uses soft measured glucose/O2 constraints by default, and compares the fitted output against the non-fitted source by default. The original fitted file `MINN_data/fluxomics_iAF1260_reduced_split_fit.csv` is only a descriptive audit reference, not a target to reproduce; use `--compare-output-to-reference` only when that extra comparison is explicitly wanted.
- `fit_minn_fluxomics_minn_like.py` is the MINN-like refitting-policy trial in iML1515. It uses `models/iML1515.xml`, fixes biomass by default, disables the hard glucose/O2 soft-input band by default, and writes `MINN_data/fluxomics_iML1515_minn_like_fit.csv`.

## 2) Data and feature setup
- Training/eval notebook: `ecoli_iML1515_MINN_model_testing.ipynb`.
- MINN-style data directory: `./MINN_data`.
- `MINN_FLUXOMICS_FILE` is a simple fluxomics-file switch:
  - current active fitted/Table 4-comparable file: `fluxomics_iAF1260_reduced_split_fit.csv`
  - non-fitted robustness file: `fluxomics_iAF1260_reduced_split.csv`
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
- Downstream pFBA extra-constraint modes:
  - `etoh_ac_cap`: predicted `R_EX_etoh_e`, `R_EX_ac_e` as secretion upper caps; CO2 remains in the reservoir input but is unconstrained in pFBA
  - `co2_etoh_ac_cap`: predicted `R_EX_co2_e_fwd`, `R_EX_etoh_e`, `R_EX_ac_e` as secretion upper caps

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
- Front MLP:
  - 1 hidden layer
  - hidden size 512
- Front-MLP context outputs use `softplus` to produce nonnegative latent context/cap values.
- Transformer remains frozen (no optimizer params from transformer).

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
- Training stabilization:
  - gradient clipping (`MINN_GRAD_CLIP_MAX_NORM`)
  - LR warmup + cosine decay (`MINN_LR_WARMUP_EPOCHS`, `MINN_LR_COSINE_MIN_FACTOR`)
- AMP:
  - use `torch.amp.autocast(...)`
  - use `torch.amp.GradScaler(...)`

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
- Per-LOO R2/MAE/RMSE panel plot: save to `./pics/minn_loo_metric_panels.png` by default and still show it in the notebook; set `MINN_LOO_METRIC_PLOT_SAVE_PATH=None` for display-only behavior.
- FluxTransformer->pFBA cap-binding diagnostics:
  - `pfba_cap_binding_diagnostics_df`
  - `pfba_cap_binding_sample_summary_df`
  - `pfba_cap_binding_result_summary_df`
  - final comparison aggregates `MINN_CAP_BINDING_DIAGNOSTICS_DF`, `MINN_CAP_BINDING_SAMPLE_SUMMARY_DF`, and `MINN_CAP_BINDING_RESULT_SUMMARY_DF`
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
- `MINN_PFBA_EXTRA_CONSTRAINT_MODE` controls downstream extra constraints:
  - `etoh_ac_cap`: predicted ethanol and acetate are nonnegative secretion upper caps; predicted CO2 is still supplied to FluxTransformer but not applied as a pFBA cap
  - `co2_etoh_ac_cap`: predicted CO2, ethanol, and acetate are nonnegative secretion upper caps
- Predicted nonnegative secretion caps use `lower_bound=max(0, min(current_lower_bound, prediction))` and `upper_bound=max(0, prediction)`.
- The notebook includes a follow-up etoh/ac-cap retraining cell after the measured-vs-predicted pFBA comparison. It selects the better glucose/O2 FluxTransformer context mode from the `co2_etoh_ac_cap` comparison, retrains with `MINN_PFBA_EXTRA_CONSTRAINT_MODE="etoh_ac_cap"`, keeps CO2 in the FluxTransformer context/input, and evaluates downstream pFBA with only ethanol/acetate caps.
- Do not include cap-calibration trials in the final comparison by default; direct cap-MAE calibration can over-shrink upper caps, and the pFBA-tuned safety calibration selected identity scales in testing.
- The final comparison cell should report: baseline pFBA, FluxTransformer to pFBA measured `co2_etoh_ac_cap`, FluxTransformer to pFBA predicted `co2_etoh_ac_cap`, and FluxTransformer to pFBA better-context `etoh_ac_cap`.
- Keep the per-sample cap-binding diagnostic cell after the final comparison. It separates bad cap prediction from pFBA overconstraint by checking whether each predicted secretion cap binds, whether it is below the fitted target (`binding_low_cap`), and whether the cap improves or worsens the fitted-target error versus baseline pFBA.
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
  - TabPFN ML-to-flux benchmark
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
