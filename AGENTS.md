# AGENTS.md

This repository contains FluxTransformer experiments for metabolic flux
prediction and analysis. Use this file as the repo-wide map. Detailed experiment
notes live in `docs/experiment_notes/`.

Update this `AGENTS.md` after major repository changes so future agents start
from the current project state.

## Repository Purpose

- Build neural surrogates for FBA/pFBA-generated flux distributions.
- Train FluxTransformer models on simulated metabolic conditions.
- Evaluate whether the learned flux predictions and token embeddings preserve
  biologically meaningful reaction structure.
- Compare direct neural flux prediction, FluxTransformer reservoir use, and
  pFBA-based downstream workflows.

## Repository Structure

- Root `*.py` files contain the main model definitions, training scripts, data
  generators, and reaction-list helpers.
- Root `*.ipynb` files are active experiment, evaluation, and exploration
  notebooks. Most thesis-facing analysis currently happens in notebooks rather
  than standalone scripts.
- `docs/` contains paper notes, thesis PDFs, and long-form guidance.
- `docs/experiment_notes/` contains maintained experiment notes that should stay
  in sync with major experiment changes.
- `data/` contains generated simulated FBA/pFBA CSV datasets used for training
  and testing FluxTransformer models.
- `AMN_data/` contains measured external datasets, including Faure-like
  iML1515 growth data and associated uncertainty files.
- `MINN_data/` contains the MINN benchmark tables and measured exchange-flux
  inputs used by the MINN-style iML1515 workflows.
- `models/` contains saved model weights, checkpoints, and training logs, usually
  grouped by model name.
- `scripts/` contains supporting utilities for plotting, data combination, and
  one-off or batch-style experiment helpers.
- `insights/` and `pics/` contain generated plots and analysis artifacts.
  Thesis-relevant figures usually live under `insights/thesis/`.
- `old/` contains historical notebooks and scripts kept for comparison only.
- `venv/`, `__pycache__/`, and other environment/cache folders are not part of
  the experiment source of truth.

## Core Files

- `flux_transformer.py`: canonical `FluxTransformer` implementation. Check this
  first for model architecture, input/output shape assumptions, and forward
  behavior.
- `train_flux_transformer.py`: older/basic in-memory training script.
- `train_flux_transformer_with_cache.py`: cached/memmap training script; use this
  for larger E. coli core runs with ordinary Huber loss.
- `train_flux_transformer_norm_loss.py`: per-flux normalized Huber-loss training
  trial. This file should contain normalized loss only; do not reintroduce the
  activity-head/tail-loss trial code here.
- `ecoli_core_model_testing.ipynb`: main E. coli core evaluation notebook.
- `ecoli_core_transformer.ipynb`: E. coli core architecture-size sweep notebook;
  trains multiple current-architecture FluxTransformer sizes on one data file
  and compares best-validation-loss metrics plus combined biomass panels.
- `ecoli_iML1515_exp_model_testing.ipynb`: iML1515 experimental-data workflow.
- `ecoli_iML1515_MINN_model_testing.ipynb`: MINN-style FluxTransformer reservoir
  workflow.
- `ecoli_iML1515_MINN_Table2.ipynb`: separate Table 2-style MINN benchmark
  notebook.
- `ecoli_iML1515_AMN_model_testing.ipynb`: AMN-style iML1515 experimental
  growth-rate workflow inspired by Faure et al.
- `yeast9_model_testing.ipynb`: main Yeast9 evaluation notebook. It intentionally
  defines the older non-injected FluxTransformer class locally for the saved
  Yeast9 checkpoint.

## Exploration Notebooks

- `ecoli_core_exploration.ipynb`: inspect the E. coli core COBRA model,
  exchange reactions, FBA/pFBA/FVA behavior, anaerobic growth, and nutrient
  source tests before changing core generators or evaluation logic.
- `ecoli_iML1515_exploration.ipynb`: inspect the iML1515 GEM, exchange/demand
  reactions, FBA/pFBA/FVA behavior, medium-source tests, and curated pathway
  reaction-ID lists for subset diagnostics.
- Yeast9 exploration uses `yeast_gem_exploration.ipynb` for GEM/model,
  exchange, and medium inspection, and `yeast9_explore_samples.ipynb` for
  generated-sample distributions, single-reaction checks, and sampling weights.

## Detailed Notes

- `docs/experiment_notes/README.md`: index for maintained experiment notes.
- `docs/experiment_notes/MINN_training_notes.md`: detailed MINN
  training/evaluation guide moved out of the root instructions.
- `docs/experiment_notes/ecoli_core_experiment_notes.md`: E. coli core
  FluxTransformer experiment notes and current interpretation.
- `docs/experiment_notes/AMN_experiment_notes.md`: initial notes for the
  Faure-inspired iML1515 AMN-style experiments.
- `docs/experiment_notes/yeast9_experiment_notes.md`: Yeast9 FluxTransformer
  evaluation and training-efficiency notes.

## Data Generation

- E. coli core simulated data is currently generated with
  `generate_ecoli_core_data.py`.
- Older E. coli core history used `generate_ecoli_core_data_complex.py`; when
  comparing old commits, check which generator produced the data.
- iML1515 simulated data is generated by files such as
  `generate_ecoli_iML1515_data.py` and `generate_ecoli_iML1515_MINN_data.py`.
- AMN-style iML1515 simulated media data should use
  `generate_ecoli_iML1515_AMN_data_stable.py` for new runs. The older
  `generate_ecoli_iML1515_AMN_data.py` is legacy and kept for traceability of
  existing checkpoints/data.
- Shared AMN/MINN iML1515 reservoir data should use
  `generate_ecoli_iML1515_AMN_MINN_data.py`. This is not the strict Faure-style
  generator: it adds glucose and ethanol context inputs and mixes Faure-like
  no-glucose media with MINN Table 4-style glucose/oxygen media.
- Yeast experiments use the `yeast9_*` files.
- Use the generator as the source of truth for reaction order, sign conventions,
  exchange bounds, and sampled constraints.

## Yeast9 Experiments

- The active Yeast9 evaluation notebook is `yeast9_model_testing.ipynb`.
- The active checkpoint is
  `models/yeast9_d256_h8_l3_ff1024/yeast9_d256_h8_l3_ff1024_checkpoint.pth`.
- This checkpoint was trained with the older non-injected FluxTransformer API.
  The notebook defines that older model class locally. Do not replace it with
  the current `flux_transformer.py` injected-token implementation unless the
  Yeast9 checkpoint is retrained.
- Yeast9 diagnostics should include biomass (`r_2111`), selected high-flux
  examples (`r_1110`, `r_2100`, `r_1672`), pooled full-output metrics, and
  difficult-flux tables/plots.
- For Yeast9, use pooled metrics to compare with smaller models, but keep the
  reaction-level caveat: many outputs are sparse or low-variance, and accuracy is
  uneven across the full reaction set.
- Yeast9 output-subset training experiments belong in the `yeast9_rs_*` files.
  Treat subset training as a speed-accuracy tradeoff.

## Token and Mapping Invariants

- Treat token order as a contract between data, model checkpoints, notebooks, and
  plots.
- Always assert that loaded test-data token order matches the checkpoint output
  order before evaluating a saved model.
- Inputs are usually mapped into matching `*_flux` tokens; verify this mapping
  explicitly before changing data generation or model loading.
- Do not split reversible reactions or rename reaction tokens in evaluation code
  unless the generator and checkpoint were built with the same convention.
- For pFBA/COBRA workflows, check bound direction and sign carefully, especially
  for uptake reactions and split `*_rev` source columns.

## Training Guidance

- Prefer `train_flux_transformer_with_cache.py` or
  `train_flux_transformer_norm_loss.py` for current E. coli core training.
- Training scripts save a separate log file beside the model and checkpoint:
  `./models/{model_name}/{model_name}_training.log`.
- Logs should mirror user-facing training prints and include the training file,
  `MODEL_NAME`, resolved `model_name`, data path, device, architecture, training
  settings, and loss settings.
- Cached training scripts print first-batch GPU memory with `print_gpu_memory()`
  to help diagnose batch-size OOM risk.
- Current larger core runs use `num_epochs=100` and `patience=10` where early
  stopping is implemented.
- Cached current training restores/saves the best epoch, not simply the final
  epoch. Temporary best checkpoints should be removed after final save.
- `train_flux_transformer_norm_loss.py` uses per-flux normalized Huber loss:
  predictions and targets are divided by a per-flux scale before Huber loss.
  Supported scale methods include `q95_q05`, `std`, and `iqr`; `q95_q05` is the
  current default.
- Do not confuse the normalized-loss trial with the older activity-head/tail
  trial. The activity-head and stratified-loss trials were not adopted as the
  main model.

## E. coli Core Experiments

- Main notebook: `ecoli_core_model_testing.ipynb`.
- Architecture sweep notebook: `ecoli_core_transformer.ipynb`; use it for
  capacity comparisons, not as the main thesis-facing evaluation notebook.
- Current result interpretation emphasizes both prediction quality and embedding
  structure:
  - pooled model metrics,
  - per-flux metric table,
  - biomass/GLUDy/GLUN diagnostics,
  - full colored t-SNE,
  - gray token-cluster t-SNE with black initial embeddings,
  - pathway-grouped, Spearman-grouped, and stoichiometric-grouped plots.
- Gray t-SNE plots should show post-layer token embeddings for the selected
  contexts and tokens. Black points represent initial token embeddings in the
  same t-SNE fit when `show_pre_embeddings=True`.
- Do not tie black pre-embedding plotting to diverse-sample highlighting; those
  are separate options.
- Pathway groupings are interpretive. Small biologically feasible group changes
  are acceptable only when they improve plot coherence without violating known
  metabolism.
- Current normalized-loss results improve pooled `R^2`, pooled RMSE, and the
  number of very high-`R^2` fluxes compared with the normal Huber-loss commit
  `faaa3de`, but worsen pooled MAE and many per-flux MAEs. Describe this as a
  tradeoff, not a clean upgrade.
- Difficult fluxes such as `GLUDy`, `GLUN`, `GLUSy`, `NADTRHD`, `MDH`, `ME2`,
  `PPCK`, and `PYK` should be analyzed by failure mode rather than treated as
  one generic "hard flux" class.

## MINN Experiments

- The detailed MINN guide is in
  `docs/experiment_notes/MINN_training_notes.md`.
- For MINN-style experiments, use a pretrained FluxTransformer as a frozen
  reservoir and train only the front MLP unless the user explicitly asks for a
  different ablation.
- The active MINN AMN-trial notebook compares measured versus predicted
  glucose/oxygen FluxTransformer context. Both branches use the same three
  latent front-MLP CO2/ethanol/acetate context/cap outputs; no alternative cap
  sets are evaluated.
- The active MINN pFBA comparison is Table 4-style iML1515 pFBA, not the Table
  2 benchmark. Table 2-style work belongs in `ecoli_iML1515_MINN_Table2.ipynb`.
- MINN reservoir forward passes must keep `output_subset=None`; requesting only
  target fluxes changes the attention token set and therefore changes the frozen
  reservoir prediction.
- `ecoli_iML1515_MINN_AMN_model_testing_trial.ipynb` uses the restored
  `6b1e3bf` one-layer width-512 ReLU/raw-Huber workflow with the AMN/MINN
  FluxTransformer, `minn_fitted` targets, full-vocabulary forward, and
  `co2_etoh_ac_cap` pFBA mode. Its predicted-context result slightly beats
  baseline pFBA and is the reference configuration. The rejected two-layer
  GELU/per-flux-normalized-loss version remains available in Git history.

## AMN Experiments

- The initial AMN guide is in
  `docs/experiment_notes/AMN_experiment_notes.md`.
- Current AMN-style work uses the Faure et al. Artificial Metabolic Network idea
  but adapts it to this repo by using a frozen FluxTransformer as the metabolic
  reservoir.
- The active notebook target is experimental growth rate (`GR_AVG`) from
  `AMN_data/iML1515_EXP.csv`; uncertainty plots use `GR_STD` from
  `AMN_data/EXP110.csv`.
- iML1515 AMN notebook t-SNE/pathway-subset plots are exploratory diagnostics
  only. Current thesis t-SNE/embedding figures should come from the E. coli core
  experiments, not iML1515.
- The stable AMN generator keeps the Faure-style 38-exchange nutrient identity
  and order, but follows the more robust MINN-style generation process: closed
  uptake reset, preserved exchange upper bounds, accepted-sample target loop,
  attempt/feasible-rate logging, temporary output file, solver timeout, and
  periodic model reload.
- AMN oxygen is flexible by default. The stable generator samples `EX_o2_e`
  unless `--fixed-oxygen` is explicitly used for an ablation.
- Fixed glycerol and amino-acid caps default to `2.2` in the stable AMN
  generator. This intentionally avoids the very high background carbon
  availability caused by setting these carbon-containing fixed supplements to
  `10`; document any change to this policy in the AMN notes and thesis notes.
- `generate_ecoli_iML1515_AMN_MINN_data.py` is the shared-reservoir generator
  for a checkpoint intended to work in both AMN-style growth prediction and MINN
  Table 4-style reservoir experiments. It includes `EX_glc__D_e` and
  `EX_etoh_e` beyond the Faure 38-input set, keeps oxygen flexible, and samples
  separate `minn`, `faure`, and `mixed` regimes. Do not present this shared
  generator as an exact Faure setup.
- When changing AMN experiment generation, update
  `docs/experiment_notes/AMN_experiment_notes.md` and this `AGENTS.md`
  together.

## Evaluation and Plotting Hygiene

- Keep notebooks deterministic where possible: fixed sample counts, perplexity,
  cluster counts, and model names should be visible in the cell.
- Save thesis-relevant plots to `pic_dir` using filenames that include
  `model_name`.
- Do not overwrite trial results with main-model results; copy notebooks before
  changing them for uncertain trials.
- When comparing models, use both pooled metrics and per-flux metrics. Pooled
  `R^2`/RMSE can improve while MAE worsens.
- For Spearman correlation grouping, choose cluster count based on the structure
  being explained. More requested groups can split off singletons rather than
  reveal new broad modules.
- t-SNE/UMAP plots are qualitative evidence. Do not claim mechanism from a plot
  alone; connect visual structure to flux correlations, stoichiometry, pathway
  membership, and known biology.

## Common Failure Modes

- CUDA OOM during training or diagnostics: reduce batch size or sample count and
  check first-batch GPU memory.
- Token-order mismatch between CSV, checkpoint, and notebook.
- Wrong reaction sign or uptake/secretion bound direction in pFBA constraints.
- Accidentally evaluating a trial checkpoint as if it were the main model.
- Reintroducing removed trial logic such as activity-head/tail loss into the
  normalized-loss training file.
- Treating high pooled metrics as proof of exact feasibility. Neural predictions
  should still be checked for mass-balance residuals, bound violations, and
  objective consistency when feasibility matters.

## Git and Artifact Hygiene

- Do not revert user changes unless explicitly asked.
- Ignore unrelated dirty files.
- Keep large generated artifacts out of commits unless the user explicitly wants
  them tracked.
