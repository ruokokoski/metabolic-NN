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
- For FluxTransformer->pFBA, keep glucose/oxygen as measured inputs. Downstream extra constraints are selected by `MINN_PFBA_EXTRA_CONSTRAINT_MODE`; the reservoir training context still includes CO2 regardless of mode.

## 1.1) Simulated MINN data generation file
- Simulated MINN-style training data for FluxTransformer is generated in: `generate_ecoli_iML1515_MINN_data.py`.
- Use this file as the sign/bound/order ground truth when validating notebook mappings.
- The current generator samples glucose and oxygen uptake constraints, leaves CO2/ethanol/acetate uncapped in the secretion direction, and fills their context columns from realized pFBA secretion fluxes after solving.

## 2) Data and feature setup
- Training/eval notebook: `ecoli_iML1515_MINN_model_testing.ipynb`.
- MINN-style data directory: `./MINN_data`.
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
  - `etoh_ac_cap`: predicted `R_EX_etoh_e`, `R_EX_ac_e` as secretion upper caps
  - `co2_etoh_ac_cap`: predicted `R_EX_co2_e_fwd`, `R_EX_etoh_e`, `R_EX_ac_e` as secretion upper caps

## 3) Mapping/sign conventions (critical)
- Keep explicit source->token mapping and signs consistent with generation scripts.
- `*_rev` source columns are positive magnitudes in the split data but represent uptake direction.
- During training, full context magnitudes are recovered from signed context targets using `context_signs` and clamped to nonnegative magnitudes; measured glucose/oxygen are copied into the FluxTransformer input context by default.
- `MINN_CAP_SUPERVISION_MODE="latent"` is the default: there is no exact auxiliary loss on cap values. Set it to `"exact"` only as an ablation to add direct Huber supervision toward observed context magnitudes.
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
- Transformer remains frozen (no optimizer params from transformer).

## 5) Training protocol
- Outer CV: Leave-One-Out (LOO).
- Inner CV: KFold hyperparameter evaluation.
- HPO:
  - `drop_rate`, `learning_rate`, `weight_decay`
  - one-time global HPO mode is available (`MINN_HPO_ONCE=True`) to reduce runtime.
  - optional per-aux retune toggle: `MINN_REDO_HPO_PER_AUX_WEIGHT`
  - current default trials: `minn_cv_max_trials=50`
  - fixed run mode is available with `MINN_USE_FIXED_AUX_AND_HYPERPARAMS=True`; this skips both the aux-weight grid search and Optuna trials, using `MINN_FIXED_CONSTRAINT_AUX_WEIGHT` and `MINN_FIXED_BEST_PARAMS` directly.
  - current fixed aux weight: `0.0` in latent mode, `0.5` in exact auxiliary mode
  - current fixed best hyperparameters: `{"drop_rate": 0.25, "learning_rate": 0.0009736777696601047, "weight_decay": 4.738391288843704e-05}`
  - current aux-weight grid is `[0.0]` in latent mode; exact auxiliary mode uses `[0.3, 0.35, 0.4, 0.45, 0.5]`
  - current one-time HPO anchor aux weight is `0.0` in latent mode and `0.4` in exact auxiliary mode
  - best hyperparameters must be printed immediately after HPO trials complete
  - objective is stability-aware:
    - `mean(inner_fold_val_loss)`
    - `+ MINN_INNER_CV_STD_PENALTY * std(inner_fold_val_loss)`
- Loss:
  - flux loss on transformer target channels (`y_minn_np`) that exclude the 5 context columns by default
  - optional auxiliary front-MLP context loss only when `MINN_CAP_SUPERVISION_MODE="exact"`
  - `MINN_AUX_CHANNEL_WEIGHT_MODE` controls exact auxiliary loss distribution across whichever context channels are predicted by the MLP:
    - `equal`: all 5 possible context channels weighted equally
    - `etoh_ac_emphasized`: ethanol/acetate receive 2x the per-channel weight of glucose/O2/CO2
    - `etoh_ac_only`: only ethanol/acetate contribute to the auxiliary loss
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
  - `etoh_ac_cap`: predicted ethanol and acetate are nonnegative secretion upper caps
  - `co2_etoh_ac_cap`: predicted CO2, ethanol, and acetate are nonnegative secretion upper caps
- Predicted nonnegative secretion caps use `lower_bound=max(0, min(current_lower_bound, prediction))` and `upper_bound=max(0, prediction)`.
- Verify feasibility counts and print failed samples for debugging.
- Aux-weight selection mode:
  - `MINN_AUX_WEIGHT_SELECTION_MODE="pfba"` selects by final FluxTransformer->pFBA metrics.
  - pFBA-based aux selection must initialize/use `base_cobra_model` and `cobra_pfba` inside the training/selection cell; do not rely on the later baseline pFBA cell having already run.
  - fallback mode `"oof"` selects by pooled OOF metrics.

## 8.1) TabPFN ML-to-flux benchmark
- The TabPFN section at the end of `ecoli_iML1515_MINN_model_testing.ipynb` should mirror the Goncalves/ML2Flux benchmark used in Tazza et al. Table 2.
- Use `MINN_data/fluxomics.csv` for the original signed Ishii/Goncalves flux targets, not the split/FBA-fit MINN fluxomics file.
- Inputs:
  - transcriptomics
  - proteomics
  - measured uptake fluxes (`R_EX_glc_e_`, `R_EX_o2_e_`)
- Targets:
  - all fluxomics columns except the two fixed uptake fluxes
  - expected shape: 45 target fluxes over 29 samples
- Protocol:
  - Leave-One-Out CV using `KFold(n_splits=len(X), shuffle=True, random_state=12345)`
  - fit `StandardScaler` for `X` and `y` inside each fold only
  - fit one `TabPFNRegressor` per target flux because TabPFN regression is single-output
  - report R2, MAE, RMSE, and NE as mean +/- std across LOO samples, Table 2 style

## 8.2) Goncalves-style pFBA with iML1515
- The final pFBA cell repeats the Goncalves/omics2flux Ishii pFBA baseline, but swaps the GEM to `models/iML1515.xml`.
- Keep the Goncalves protocol:
  - fixed measured glucose and oxygen uptake from `MINN_data/fluxomics.csv`
  - pFBA over the same 45 non-uptake Table 2 flux targets
  - R2, MAE, RMSE, and NE as mean +/- std across the 29 Ishii samples
- Important mapping detail:
  - local `fluxomics.csv` rows use gene-symbol sample names
  - `omics2flux/pfba.py` uses an ordered b-number knockout list
  - therefore the notebook must map sample names to the original Goncalves b-numbers before knockout
- iML1515 does not contain the original Goncalves `b4395` gene used for the `gpmB` sample. The adapted iML1515 benchmark maps it to the iML1515 PGM isozyme `b3612` so all 29 samples solve.
- Sanity check: the iML1515 Goncalves-style pFBA cell should report `Successful pFBA samples: 29/29` and an empty failed-sample table.

## 9) Common failure modes
- CUDA OOM in diagnostics/training:
  - lower batch size
  - keep full-output forward where required
- Mapping mismatch between source columns and output tokens.
- Wrong sign when converting predicted magnitudes to pFBA bounds.
- Misaligned experiment ordering when attaching OOF predictions.
- Missing TabPFN benchmark dependencies in the active Python environment (`pandas`, `scikit-learn`/`sklearn`, `tabpfn`).

## 10) Recommended sanity checks
- Assert all required token names exist in `outputs`.
- Print source->token mapping with signs.
- Print c0/cap value versus observed-context magnitude summaries for the 5 context channels.
- Check OOF sample count equals dataset count in LOO context.
- Confirm pFBA evaluated sample count and metric table shape.
- Confirm FluxTransformer->pFBA `pred_vin_df` contains only the predicted extra constraints for the selected mode, never predicted glucose/oxygen.
- For the TabPFN benchmark, confirm the benchmark prints 29 samples, 141 features, and 45 targets.
