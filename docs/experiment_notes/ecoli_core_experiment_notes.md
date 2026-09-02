# E. coli Core Experiment Notes

This file captures the current E. coli core FluxTransformer work so the root
`AGENTS.md` can stay as a repository map. Longer notes are in
`docs/working_notes/ecoli_core_plot_notes.md` and should be treated as local draft material
unless the user explicitly asks otherwise.

## Scope

- Main model family: `FluxTransformer` trained on simulated E. coli core pFBA/FBA
  data.
- Main evaluation notebook: `ecoli_core_model_testing.ipynb`.
- Architecture-size sweep notebook: `ecoli_core_transformer.ipynb`.
- Core-model exploration notebook: `ecoli_core_exploration.ipynb`.
- Current data generator: `generate_ecoli_core_data.py`.
- Older historical generator: `generate_ecoli_core_data_complex.py`, available
  in Git history rather than the current working tree. Check the relevant old
  commit before comparing old plots or metrics, because generator changes can
  explain differences in embedding figures.

## Core Files

- `flux_transformer.py`: canonical model implementation.
- `train_flux_transformer.py`: cached/memmap training with ordinary Huber loss.
- `train_flux_transformer_norm_loss.py`: current normalized-loss training trial.
  Keep this file focused on per-flux normalized Huber loss; activity-head and
  tail-loss code was removed after testing.
- `ecoli_core_model_testing.ipynb`: main analysis notebook. Use this for
  current metrics, flux diagnostics, and embedding plots.
- `ecoli_core_transformer.ipynb`: architecture-size sweep notebook for training
  multiple current-architecture FluxTransformer sizes on the same E. coli core
  data file. Use it to compare model capacity with best-validation-loss metrics
  and combined biomass prediction panels; it should not save per-model test
  artifacts.
- `ecoli_core_exploration.ipynb`: sandbox notebook for understanding and
  experimenting with the E. coli core COBRA model itself before changing
  training or evaluation code.

## Current Training Setup

- Current large core runs use `num_epochs=100` and `patience=10`.
- Best-epoch saving is expected: final saved model/checkpoint should correspond
  to the best validation epoch, not simply the last epoch.
- Training scripts write a companion log file beside the model/checkpoint:
  `./models/{model_name}/{model_name}_training.log`.
- Cached training prints first-batch GPU memory with `print_gpu_memory()` at
  `epoch == start_epoch and batch_idx == 0`, which helps diagnose whether batch
  size is likely too large.
- `train_flux_transformer_norm_loss.py` scales each output flux before applying
  Huber loss. Supported scale choices include `q95_q05`, `std`, and `iqr`;
  `q95_q05` is the current default.

## Evaluation Notebook Checks

- Assert that loaded test-data token order matches the checkpoint output order
  before computing metrics or plots.
- Keep `model_name`, checkpoint path, data path, and `outputs` visibly tied
  together in notebook cells.
- Include both pooled metrics and per-flux metrics. Pooled metrics alone can
  hide flux-specific regressions.
- Keep the hard-flux audit cell: it is useful for identifying low-`R^2` or high
  error fluxes before proposing new loss functions.
- Do not overwrite main-model analysis with uncertain trial results. Copy the
  notebook before switching to a trial checkpoint.

## Core Exploration Notebook

- `ecoli_core_exploration.ipynb` is for experimenting with the base E. coli core
  metabolic model, not for reporting final FluxTransformer results.
- Use it to inspect the COBRA model structure: reactions, metabolites, genes,
  exchange reactions, bounds, objective reaction, and specific reactions or
  metabolites.
- It includes model-map visualization, ordinary FBA, pFBA, FVA, anaerobic growth
  tests, alternative carbon-source tests, alternative nitrogen-source tests,
  base-exchange essentiality checks, and nutrient-uptake sensitivity analysis.
- It also contains training-data inspection cells. These are useful when asking
  whether a difficult flux is caused by the simulated data distribution, low
  variance, many zero values, rare active states, or pathway degeneracy.
- Use this notebook before changing the generator or loss function when the
  question is biological or constraint-related. Use
  `ecoli_core_model_testing.ipynb` when the question is about a trained
  checkpoint, prediction metrics, or learned embeddings.

## Current Results Interpretation

- The normalized-loss model improves pooled `R^2`, pooled RMSE, and the number
  of very high-`R^2` fluxes compared with the earlier normal Huber-loss model
  from commit `faaa3de`.
- The same normalized-loss model worsens pooled MAE and many individual per-flux
  MAEs. Treat it as a tradeoff, not a simple overall win.
- Current threshold counts from the provided 95-flux table, using `>=`, are:
  `R^2 >= 0.90`: 78/95, `R^2 >= 0.95`: 68/95, `R^2 >= 0.99`: 37/95.
- Biomass prediction is strong and suitable as evidence that the surrogate
  captures the main growth objective under many simulated conditions.
- Difficult fluxes still need careful discussion. Important examples include
  `GLUDy`, `GLUN`, `GLUSy`, `NADTRHD`, `MDH`, `ME2`, `PPCK`, and `PYK`.

## Trial History

- Activity-head/tail-aware trial: not adopted. It did not solve the difficult
  fluxes well enough and should not be reintroduced into
  `train_flux_transformer_norm_loss.py`.
- Stratified hard-flux training trial: not adopted. It improved some tail-error
  behavior but weakened broader metrics, so the file was removed.
- Hurdle-style loss was discussed but not prioritized, because several difficult
  fluxes have different failure modes and should first be identified and grouped
  more carefully.
- Better next trials should start from the original architecture unless there is
  clear evidence that architectural changes are needed.

## Plotting and Embedding Notes

- Gray t-SNE plots from `plot_post_attn_tsne_gray` should show post-layer token
  embeddings for the selected contexts and selected tokens. They should not be
  reduced to one post-token center per reaction.
- Black points represent initial token embeddings when
  `show_pre_embeddings=True`. They should be controlled independently from
  diverse-sample highlighting.
- For thesis-style gray figures, the preferred interpretation is the joint t-SNE
  view where black initial embeddings and gray post-layer token states are in
  the same fit.
- Full colored t-SNE plots show how all sampled post-layer token embeddings
  distribute in the learned representation space.
- Pathway-grouped plots are interpretive. Only move reactions between pathway
  groups when the move is biologically feasible and also makes same-color
  regions more coherent in the actual plot.
- Spearman-grouped plots should be explained as correlation-based response
  modes. Increasing the requested cluster count can split off singleton
  reactions instead of revealing new broad modules.

## Pathway Grouping Guidance

- Keep pathway names readable but not overly short. For example, avoid labels
  like `Ethanol`; prefer names that still identify the pathway or process.
- Small groups can be merged into larger biologically related groups when this
  improves plot readability without misrepresenting metabolism.
- Do not optimize pathway labels only for visual clustering. The grouping still
  has to be biologically defensible.
- Watch for duplicate reaction entries in grouping dictionaries. Duplicate keys
  do not normally change the final color assignment, but they make the code
  harder to audit.

## Difficult Flux Improvement Ideas

- First identify hard fluxes systematically by `R^2`, MAE, RMSE, activity rate,
  and high-flux tail errors.
- Separate failure modes before changing training:
  - mostly inactive or low-variance fluxes,
  - high-tail fluxes where large values are underpredicted,
  - reversible or alternative-pathway fluxes with ambiguous optima,
  - coupled redox/energy reactions sensitive to small constraint changes.
- Candidate future trials:
  - per-flux or group-weighted losses based on hard-flux audit results,
  - regime-aware sampling or batch balancing,
  - targeted oversampling of rare active/high-flux states,
  - auxiliary diagnostics for activity state without making it the main model,
  - feasibility-aware post-processing or penalties when biological feasibility
    matters more than pure regression error.
