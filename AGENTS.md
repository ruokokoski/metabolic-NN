# AGENTS.md

## Purpose and Scope

Develop FluxTransformer toward a universal neural surrogate for metabolic flux
prediction across diverse simulated conditions and downstream tasks. Current
experiments use separate reaction vocabularies and checkpoints for E. coli core,
iML1515, and Yeast9; cross-model universality is a development goal, not an
established capability.

Use this file as a concise repository map and set of development invariants.
Keep experiment settings, results, and trial history in `docs/experiment_notes/`.
Read the relevant note before changing an experiment, update it when generation,
training, evaluation, or interpretation changes, and update this map after
structural changes.

## Repository Map and Experiment Notes

- Root Python files and notebooks: models, training, generation, and evaluation.
- `docs/experiment_notes/README.md`: experiment-note index. Relevant notes:
  - `ecoli_core_experiment_notes.md`: core training and evaluation.
  - `iML1515_sampling_study_notes.md`: A/B/A ∪ B/C/D/E distributions and transfer.
  - `AMN_experiment_notes.md`: AMN growth prediction and experimental-pattern data.
  - `MINN_training_notes.md`: MINN flux prediction, context, and pFBA protocols.
  - `AMN_MINN_shared_reservoir_notes.md`: shared C and cross-task workflows;
    update alongside affected task notes when shared experiments change.
  - `yeast9_experiment_notes.md`: Yeast9 evaluation and output-subset training.
- `docs/reference/`: source audits, biological mappings, and dataset provenance.
- `docs/working_notes/fungi_experiment_notes.md`: private, ignored fungi / FluxFormer
  working notes; keep fungi findings here until public documentation is requested.
- `data/`: simulated datasets; `AMN_data/` and `MINN_data/`: experimental inputs.
- `models/`: checkpoints and training logs; `tests/`: focused regression checks.
- `scripts/`: utilities and cluster jobs, including `scripts/roihu/`.
- `insights/` and `pics/`: generated analyses and figures; `old/`: historical code.
  Generated artifacts, caches, and environments are not implementation authority.

## Main Entry Points

- `flux_transformer.py`: shared injected-token `FluxTransformer`, including
  output-subset inference. Trainers and some notebooks also define local model
  classes; verify the actual class used before changing architecture or loading
  checkpoints.
- `train_flux_transformer.py`: cached/memmap training with ordinary Huber loss.
- `train_flux_transformer_norm_loss.py`: per-flux normalized Huber-loss training;
  loss scales are estimated from the training split.
- `yeast9_rs_transformer.py`: Yeast9 training, full-output by default with
  optional correlation-based output subsets. `yeast9_data.py` provides shared
  data loading and deterministic splits; `yeast9_reactions.py` defines tokens.
- `ecoli_iML1515_reactions.py`: shared iML1515 reaction/token helpers.
- `iml1515_evaluation.py`: shared AB-union/C/D/E simulated-fidelity, AMN, MINN,
  and pFBA evaluation helpers.
- `ecoli_iML1515_AB_union_model_testing.ipynb`,
  `ecoli_iML1515_C_model_testing.ipynb`, `ecoli_iML1515_D_model_testing.ipynb`,
  and `ecoli_iML1515_E_model_testing.ipynb`: combined AMN/MINN transfer evaluation.
- `ecoli_iML1515_A_model_testing.ipynb` and
  `ecoli_iML1515_B_model_testing.ipynb`: standalone reservoir evaluations,
  including distribution-shift and cross-task controls.
- `ecoli_iML1515_AMN_model_testing.ipynb`: separate AMN_11k experimental-pattern
  experiment with independent simulated test data; check artifact availability
  and provenance in the AMN note before running.
- `ecoli_iML1515_MINN_Table2.ipynb`: separate MINN Table 2-style benchmark.
  `ecoli_iML1515_AMN_MINN_model_testing_trial.ipynb` and
  `ecoli_iML1515_MINN_AMN_model_testing_trial.ipynb`: shared-reservoir task trials.
- `ecoli_core_model_testing.ipynb`: main core evaluation;
  `ecoli_core_10M_ordinary_huber_eval.ipynb`: ordinary-Huber comparison;
  `ecoli_core_transformer.ipynb`: architecture-size sweep.
- `yeast9_model_testing.ipynb`: main Yeast9 evaluation, with a local older
  non-injected model class required by its saved checkpoint.

Use `ecoli_core_exploration.ipynb`, `ecoli_iML1515_exploration.ipynb`,
`yeast_gem_exploration.ipynb`, and `yeast9_explore_samples.ipynb` to inspect
metabolic constraints and sample distributions before changing generators.
`fungi_exploration.ipynb` loads the SBML models in `metabolic_models/` and prints
basic structural summaries, baseline FBA feasibility, and permitted uptake bounds.
`metabolic_models/Growth_data_for_ML_Helsinki_Uni.csv` contains accompanying
growth summaries; model/strain mappings and interpretation limits are in the fungi note.

## Data Generator Routing

- E. coli core: `generate_ecoli_core_data.py`.
- iML1515 A: `generate_ecoli_iML1515_A_data.py`, AMN-oriented source subsets.
- iML1515 B: `generate_ecoli_iML1515_B_data.py`, Tazza-style five-channel cap
  sampling; consult code and notes for current distributions.
- iML1515 A ∪ B: `generate_ecoli_iML1515_AB_union_data.py`, balanced literal
  mixture of A and B regimes.
- iML1515 C: `generate_ecoli_iML1515_C_data.py`, shared AMN/MINN and bridge regimes.
- iML1515 D: `generate_ecoli_iML1515_D_data.py`, broad task-aware distribution.
- iML1515 E: `generate_ecoli_iML1515_E_data.py`, curated broad task-agnostic
  distribution. `generate_ecoli_iML1515_E_data_old.py` is archival.
- Experimental-pattern AMN: `generate_ecoli_iML1515_AMN_data.py`, condition-based
  allocation; separate from A. `generate_AMN_sweep.py` generates diagnostic sweeps.
- Yeast9: `generate_yeast9_data.py`.

## Model, Data, and Constraint Contracts

- Treat ordered inputs, output vocabulary, input-to-`*_flux` mappings, reversible
  token conventions, and architecture as a contract across generation, training,
  checkpoints, and evaluation. Assert compatibility before scoring; identify
  incompatible consumers before changing tokenization or architecture.
- Use generators as authority for sampling, bounds, and signs. Verify positive
  `*_rev` magnitudes against signed COBRA uptake bounds and secretion caps.
  Concentrations are not flux bounds; an unmeasured nutrient is not absent.
- Record the metabolic model, generator arguments, solver/objective, pFBA
  fraction, seeds, token order, and data paths. Keep training, validation, and
  test data separate; fit preprocessing and loss scales on training data only.
- Preserve numerical tolerances, seeds, splits, loss definitions, and defaults
  unless intentionally changing the experiment. Generator revisions do not
  retroactively change existing datasets or checkpoint provenance.
- Save best-validation checkpoints when using early stopping, with architecture,
  token mappings, and reproducible settings. Keep companion training logs at
  `models/{model_name}/{model_name}_training.log`.

## Evaluation and Verification

- Report pooled and per-flux metrics; inspect sparse, low-variance, tail, and
  reversible fluxes when interpreting errors. Evaluate held-out conditions and
  distribution shifts when claiming broader generalization, and distinguish
  simulated fidelity from experimental downstream transfer.
- Check mass balance, bound violations, and objective consistency when assessing
  feasibility. Track neural context inputs separately from downstream pFBA caps.
  Embedding visualizations alone do not establish biological mechanism.
- Keep notebook paths, model identity, seeds, and material settings explicit.
  Distinguish current code, saved outputs, archival results, and pending reruns;
  keep trial artifacts separate from established results.
- Update affected consumers and experiment notes together. Shared AMN/MINN
  changes also require the shared-reservoir note; sampling-study changes require
  the sampling-study note. Record trial provenance, changed variables, validation
  protocol, results, and keep/reject decisions in the relevant note.
- Run focused existing tests and relevant validation. For Python, check syntax;
  for notebooks, validate JSON and inspect changed cells and outputs; for data
  and checkpoints, check shapes, token order, dtypes/devices, signs, and bounds.
  Review the final diff and report verification limits.
- Preserve unrelated user changes. Keep large datasets, checkpoints, plots,
  caches, and environments out of commits unless explicitly requested.
