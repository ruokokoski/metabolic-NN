# AGENTS.md

## MINN training quick guide (ecoli_iML1515_MINN_model_testing.ipynb)

This document captures the main points an agent should follow when working on the MINN-style training/evaluation pipeline in this repo.

## 1) Core goal
- Use a pretrained `FluxTransformer` as a frozen reservoir.
- Train only a front MLP on omics + measured inputs.
- Front MLP predicts the same 5-channel reservoir context used by MINN.
- Reconstruct full transformer input as:
  - predicted 5 channels
  - fixed `base_exchanges`
  - zeros for other channels
- Run transformer forward with full vocab output.
- For FluxTransformer->pFBA, keep glucose/oxygen as measured inputs. Downstream extra constraints are selected by `MINN_PFBA_EXTRA_CONSTRAINT_MODE`; the reservoir training context still includes CO2 regardless of mode.

## 1.1) Simulated MINN data generation file
- Simulated MINN-style training data for FluxTransformer is generated in: `generate_ecoli_iML1515_MINN_data.py`.
- Use this file as the sign/bound/order ground truth when validating notebook mappings.

## 2) Data and feature setup
- Training/eval notebook: `ecoli_iML1515_MINN_model_testing.ipynb`.
- MINN-style data directory: `./MINN_data`.
- Inputs include:
  - transcriptomics
  - proteomics
  - measured flux inputs (`R_EX_glc__D_e_rev`, `R_EX_o2_e_rev`)
- Front-MLP 5 reservoir context source channels:
  - `R_EX_glc__D_e_rev`
  - `R_EX_o2_e_rev`
  - `R_EX_co2_e_fwd`
  - `R_EX_etoh_e`
  - `R_EX_ac_e`
- Downstream pFBA extra-constraint modes:
  - `etoh_ac`: predicted `R_EX_etoh_e`, `R_EX_ac_e`
  - `co2_exact_etoh_ac`: predicted `R_EX_co2_e_fwd`, `R_EX_etoh_e`, `R_EX_ac_e` as tight constraints
  - `co2_band_etoh_ac`: predicted `R_EX_co2_e_fwd` as a banded constraint plus tight predicted `R_EX_etoh_e`, `R_EX_ac_e`

## 3) Mapping/sign conventions (critical)
- Keep explicit source->token mapping and signs consistent with generation scripts.
- `*_rev` source columns are positive magnitudes in the split data but represent uptake direction.
- During training, auxiliary constraint targets for the 5 reservoir context channels are recovered from signed flux targets using `target_signs` and clamped to nonnegative magnitudes.
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
  - current fixed aux weight: `0.5`
  - current fixed best hyperparameters: `{"drop_rate": 0.25, "learning_rate": 0.0009736777696601047, "weight_decay": 4.738391288843704e-05}`
  - current aux-weight grid, used only when fixed run mode is disabled: `[0.3, 0.35, 0.4, 0.45, 0.5]`
  - current one-time HPO anchor aux weight: `MINN_HPO_ANCHOR_AUX_WEIGHT=0.4`
  - best hyperparameters must be printed immediately after HPO trials complete
  - objective is stability-aware:
    - `mean(inner_fold_val_loss)`
    - `+ MINN_INNER_CV_STD_PENALTY * std(inner_fold_val_loss)`
- Loss:
  - flux loss on transformer target channels
  - plus auxiliary front-MLP constraint loss (weighted)
  - `MINN_AUX_CHANNEL_WEIGHT_MODE` controls how that auxiliary loss is distributed across the 5 reservoir-context channels:
    - `equal`: current baseline behavior, all 5 channels weighted equally
    - `etoh_ac_emphasized`: ethanol/acetate receive 2x the per-channel weight of glucose/oxygen/CO2
    - `etoh_ac_only`: only ethanol/acetate contribute to the auxiliary loss
  - set `MINN_AUX_CHANNEL_WEIGHT_MODE="equal"` to revert to the pre-experiment auxiliary objective exactly
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
- OOF front-MLP 5-channel reservoir context (`oof_pred_constraints`) for diagnostics and FluxTransformer+pFBA cells.
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
- The iML1515 SBML default pFBA objective is `BIOMASS_Ec_iML1515_core_75p37M`; use this for the Table 4-style experimental pFBA comparison unless intentionally testing the simulated-data `wt` objective from `generate_ecoli_iML1515_MINN_data.py`.
- Always print the chosen pFBA objective, and map the biomass metric to the same iML1515 biomass reaction used as the pFBA objective.
- Keep FluxTransformer pFBA mapping in the unsplit/unpruned reaction space; only convert MINN split source-column signs into the corresponding unsplit iML1515 bounds or flux signs.
- FluxTransformer->pFBA should use measured glucose/oxygen uptake as lower bounds.
- `MINN_PFBA_EXTRA_CONSTRAINT_MODE` controls downstream extra constraints:
  - `etoh_ac`: tight nonnegative secretion constraints on `EX_etoh_e`, `EX_ac_e`
  - `co2_exact_etoh_ac`: tight nonnegative secretion constraints on `EX_co2_e`, `EX_etoh_e`, `EX_ac_e`
  - `co2_band_etoh_ac`: `EX_etoh_e` and `EX_ac_e` stay tight, while `EX_co2_e` uses `prediction +/- PFBA_CO2_BAND_HALF_WIDTH`
- Tight nonnegative secretion constraints use `lower_bound=max(0, prediction - PFBA_EXTRA_CONSTRAINT_TOL)` and `upper_bound=prediction + PFBA_EXTRA_CONSTRAINT_TOL`.
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

## 8.3) Direct FluxTransformer Table 2-style benchmark
- The final notebook section can add a direct `FluxTransformer` row to the Table 2-style comparison.
- This is not FluxTransformer+pFBA:
  - no COBRA/pFBA optimization is run
  - a direct Table 2 front MLP is retrained once per LOO fold
  - by default, run one global inner KFold Optuna HPO inside this final cell and reuse those best parameters across all outer LOO folds
  - `FLUXTRANSFORMER_TABLE2_HPO_PER_OUTER_FOLD=True` enables the expensive nested variant where each outer LOO fold runs its own inner KFold Optuna HPO
  - do not reuse `best_res`, `best_params_per_split`, `MINN_FIXED_BEST_PARAMS`, selected aux weights, or other training results from earlier notebook cells
  - the front MLP predicts four exchange-context values for the current Table 2 benchmark: glucose, oxygen, ethanol, and acetate
  - measured glucose/oxygen uptake are input features to the front MLP; they are not copied directly into the FluxTransformer input vector in this mode
  - the four predicted exchange-context magnitudes are inserted into the corresponding FluxTransformer input-token positions
  - CO2 is not predicted in this trial; its FluxTransformer input token is kept at the free/base upper cap (`DEFAULT_BASE_EXCHANGE_RATE`, usually 50)
  - base exchanges are fixed to the same default medium value used by `generate_ecoli_iML1515_MINN_data.py`
  - the frozen FluxTransformer full-vocabulary output is collected for the held-out sample
  - the same 45 non-uptake Goncalves/Tazza Table 2 targets are extracted from the full output
- Default supervision:
  - default test settings are 25 Optuna trials and 100 max epochs
  - assemble the FluxTransformer input in the same exchange-constraint style as the simulated pretraining data
  - optimize the front MLP through the frozen FluxTransformer on the 45 Table 2 target fluxes
  - add an auxiliary loss on the four predicted exchange-context magnitudes
  - scale the Table 2 target loss within each fold using training-fold mean/std to avoid large fluxes dominating
  - print best hyperparameters immediately after HPO finishes
- Do not feed the 45 evaluated target fluxes as FluxTransformer inputs. The pretrained FluxTransformer was trained on exchange constraints, not on internal Table 2 flux targets.
- Do not reuse earlier FluxTransformer+pFBA results for this Table 2 row. MINN Table 2 is not the reservoir model; this direct FluxTransformer row trains its own front MLP and evaluates the frozen transformer's full-vocabulary output.
- The cell must use:
  - standalone Optuna-selected front-MLP hyperparameters from the Table 2 FluxTransformer training cell
  - original signed `MINN_data/fluxomics.csv` as the Table 2 truth table
  - the same R2, MAE, RMSE, and NE mean +/- std metric convention as the TabPFN and iML1515 pFBA rows
- Keep the expensive FluxTransformer Table 2 training/HPO cell separate from the lightweight results display cell.
- Keep `TabPFN` as the last row in the combined final comparison table.

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
- Print prediction/target magnitude summaries for the 5 constraints.
- Check OOF sample count equals dataset count in LOO context.
- Confirm pFBA evaluated sample count and metric table shape.
- Confirm FluxTransformer->pFBA `pred_vin_df` contains only the predicted extra constraints for the selected mode, never predicted glucose/oxygen.
- For the TabPFN benchmark, confirm the benchmark prints 29 samples, 141 features, and 45 targets.
