# Experiment Notes

Maintained notes for active FluxTransformer experiments.

The root `AGENTS.md` contains the repository map and cross-experiment
invariants. Keep model settings, generator details, current result snapshots,
trial outcomes, and experiment-specific interpretation in the files below.

- `ecoli_core_experiment_notes.md`: E. coli core flux prediction and embedding diagnostics.
- `AMN_experiment_notes.md`: iML1515 AMN-style growth prediction experiments.
- `AMN_MINN_shared_reservoir_notes.md`: shared iML1515 generator, checkpoint,
  cross-task trials, and current AMN/MINN result snapshot.
- `MINN_training_notes.md`: iML1515 MINN-style reservoir and pFBA experiments.
- `yeast9_experiment_notes.md`: Yeast9 flux prediction and training-efficiency experiments.

## Repository-wide history

- The August 2026 legacy-model cleanup removed old E. coli core and iML1515
  checkpoints with no current or recent notebook references.
- `models/ecoli_core_d128_h8_l3_ff640_1M/` was intentionally retained.
- All Yeast9 model directories remained relevant at the time of that cleanup.
