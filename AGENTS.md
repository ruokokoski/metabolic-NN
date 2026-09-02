# AGENTS.md

This repository contains FluxTransformer experiments for metabolic-flux
prediction and analysis. Use this file as the repo-wide map and invariant list.
Read the relevant maintained note under `docs/experiment_notes/` before changing
an experiment. Keep detailed settings, results, and trial history in those notes
rather than expanding this file.

Update this file after major structural changes. Update the matching experiment
note whenever an experiment's data generation, model, evaluation, or current
interpretation changes.

## Repository Purpose

- Build neural surrogates for FBA/pFBA-generated flux distributions.
- Train FluxTransformer models on simulated metabolic conditions.
- Test whether predictions and token embeddings preserve biologically meaningful
  reaction structure.
- Compare direct neural flux prediction, frozen-reservoir workflows, and pFBA
  downstream workflows.

## Repository Map

- Root `*.py`: model definitions, training scripts, data generators, and
  reaction-list helpers.
- Root `*.ipynb`: active experiment, evaluation, and exploration notebooks.
- `docs/experiment_notes/`: maintained experiment state and interpretation.
- `data/`: generated FBA/pFBA training and test datasets.
- `AMN_data/`: measured Faure-like iML1515 growth data and uncertainties.
- `MINN_data/`: MINN benchmark tables and measured exchange-flux inputs.
- `models/`: checkpoints, weights, and training logs grouped by model name.
- `scripts/`: plotting, data-combination, fitting, and batch utilities.
- `insights/` and `pics/`: generated analysis artifacts; thesis figures usually
  live under `insights/thesis/`.
- `old/`: historical notebooks and scripts for comparison only.
- `venv/`, `__pycache__/`, and other cache/build directories are not sources of
  experiment truth.

## Read First by Task

- Model architecture or tensor behavior: `flux_transformer.py`.
- E. coli core work: `docs/experiment_notes/ecoli_core_experiment_notes.md`.
- Shared AMN/MINN generator, checkpoint, or trial work:
  `docs/experiment_notes/AMN_MINN_shared_reservoir_notes.md`.
- MINN-only reservoir or Table 2 work:
  `docs/experiment_notes/MINN_training_notes.md`.
- AMN-only growth prediction or data:
  `docs/experiment_notes/AMN_experiment_notes.md`.
- Yeast9 evaluation or output-subset training:
  `docs/experiment_notes/yeast9_experiment_notes.md`.
- Notes index and repository-wide experiment history:
  `docs/experiment_notes/README.md`.

## Main Entry Points

- `flux_transformer.py`: canonical current `FluxTransformer` implementation.
- `train_flux_transformer.py`: cached/memmap training with ordinary Huber loss.
- `train_flux_transformer_norm_loss.py`: per-flux normalized Huber-loss trial;
  keep removed activity-head and tail-loss experiments out of this file.
- `yeast9_data.py`: shared Yeast9 data-loading and deterministic split helpers.
- `yeast9_rs_transformer.py`: Yeast9 trainer with full-output training by
  default and optional correlation-based output-subset training.
- `ecoli_core_model_testing.ipynb`: main E. coli core evaluation.
- `ecoli_core_transformer.ipynb`: E. coli core architecture-size sweep.
- `ecoli_iML1515_MINN_model_testing.ipynb`: MINN frozen-reservoir workflow.
- `ecoli_iML1515_MINN_Table2.ipynb`: separate Table 2-style benchmark.
- `ecoli_iML1515_AMN_model_testing.ipynb`: AMN-style experimental growth-rate
  workflow.
- `ecoli_iML1515_AMN_MINN_model_testing_trial.ipynb`: AMN branch using the
  shared AMN/MINN reservoir.
- `ecoli_iML1515_MINN_AMN_model_testing_trial.ipynb`: MINN branch using the
  shared AMN/MINN reservoir.
- `yeast9_model_testing.ipynb`: main Yeast9 evaluation; it intentionally defines
  the older non-injected FluxTransformer locally for its saved checkpoint.

Exploration notebooks are `ecoli_core_exploration.ipynb`,
`ecoli_iML1515_exploration.ipynb`, `yeast_gem_exploration.ipynb`, and
`yeast9_explore_samples.ipynb`. Use them to inspect model structure, exchange
bounds, media, FBA/pFBA/FVA behavior, and sample distributions before changing
biological constraints or generators.

## Data Generator Routing

- E. coli core: `generate_ecoli_core_data.py`.
- General iML1515: `generate_ecoli_iML1515_data.py`.
- MINN: `generate_ecoli_iML1515_MINN_data.py`; use
  `generate_ecoli_iML1515_MINN_data_tazza.py` only for its documented ablation.
- AMN and shared AMN/MINN: `generate_ecoli_iML1515_AMN_data.py` and
  `generate_ecoli_iML1515_AMN_MINN_data.py`.
- Yeast9: `generate_yeast9_data.py`.

Read the matching experiment note before running or modifying a generator.

## Token, Mapping, and Constraint Invariants

- Treat token order as a contract between generators, CSV files, checkpoints,
  notebooks, and plots.
- Assert that loaded test-data token order matches checkpoint output order before
  evaluating a saved model.
- Verify the mapping from inputs to matching `*_flux` tokens explicitly.
- Do not rename tokens or split reversible reactions unless the generator,
  checkpoint, and evaluation code use the same convention.
- Use each generator as the source of truth for reaction order, sampled
  constraints, exchange bounds, and sign conventions.
- In COBRA/pFBA workflows, verify uptake/secretion direction and the conversion
  of positive `*_rev` magnitudes into signed bounds.
- Keep model architecture and input/output dimensions consistent with checkpoint
  metadata. Do not silently adapt incompatible checkpoints.

## Data and Training Invariants

- Keep training and test datasets separate and record the exact generator,
  arguments, seed, reaction order, and data path used for reported runs.
- Preserve numerical tolerances, random seeds, preprocessing, dataset splits,
  loss scaling, and objective definitions unless the experiment explicitly
  changes them.
- Save training logs beside checkpoints as
  `models/{model_name}/{model_name}_training.log` and include data, device,
  architecture, training, and loss settings.
- Where early stopping is implemented, restore and save the best validation
  epoch rather than the final epoch.
- Keep temporary checkpoints and partial output files out of final artifacts.
- Diagnose CUDA OOM by checking batch size, sample count, and first-batch memory
  before changing model semantics.

## Evaluation and Plotting Invariants

- Report pooled and per-flux metrics together. Pooled `R^2` or RMSE can improve
  while individual fluxes or MAE worsen.
- Analyze difficult reactions by failure mode, including sparsity, low variance,
  tail compression, alternative pathways, reversibility, and redox sensitivity.
- Do not treat high regression metrics as proof of metabolic feasibility. Check
  mass-balance residuals, bound violations, and objective consistency when
  feasibility matters.
- Keep notebooks reproducible: expose model names, paths, sample counts, seeds,
  perplexity, cluster counts, and other material settings in cells.
- Save thesis plots through `pic_dir` with filenames that identify the model.
- Keep uncertain trial notebooks and outputs separate from main-model results.
- Treat t-SNE and UMAP as qualitative evidence. Connect visual structure to
  correlations, stoichiometry, pathway membership, and known biology; do not
  infer mechanism from a plot alone.
- Keep pathway groupings biologically defensible even when adjusting them for
  readability.

## Change Coordination

- When changing a generator, update every affected notebook mapping and the
  matching experiment note.
- Update `docs/experiment_notes/AMN_MINN_shared_reservoir_notes.md` in the same
  change whenever the shared generator, either shared trial notebook, a shared
  training configuration, dataset, checkpoint, model trial, result, or
  cross-task interpretation changes. Apply this requirement to newly added
  files whose names contain `AMN_MINN` or `MINN_AMN`.
- A new shared model trial must record its data provenance, checkpoint,
  architecture, changed variable, validation protocol, main results, and
  keep/reject decision in the shared note.
- When changing model tokenization or architecture, identify all incompatible
  checkpoints and notebooks before editing consumers.
- When changing AMN generation, update
  `docs/experiment_notes/AMN_experiment_notes.md` in the same change.
- When changing MINN context, cap, mapping, or benchmark behavior, update
  `docs/experiment_notes/MINN_training_notes.md` in the same change.
- When changing E. coli core training, evaluation, or plotting behavior, update
  `docs/experiment_notes/ecoli_core_experiment_notes.md`.
- When changing Yeast9 checkpoint handling or subset training, update
  `docs/experiment_notes/yeast9_experiment_notes.md`.

## Verification

- Run the narrowest relevant tests, notebook checks, or script validation.
- For Python edits, at minimum confirm syntax and inspect the final diff.
- For notebooks, validate JSON structure and review changed code cells and saved
  outputs deliberately.
- For data or checkpoint work, validate shapes, token order, dtypes, devices,
  signs, bounds, and representative samples.
- State what was verified and what remains unverified.

## Git and Artifact Hygiene

- Preserve user changes and ignore unrelated dirty files.
- Do not commit large generated datasets, checkpoints, plots, caches, or
  environments unless explicitly requested.
- Do not replace main results with trial outputs or evaluate a trial checkpoint
  as the main model.
- Keep generated and historical artifacts out of the source-of-truth path.
