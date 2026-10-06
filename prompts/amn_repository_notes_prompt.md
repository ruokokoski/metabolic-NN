# Prompt: Inspect the Faure AMN E. coli experiments

Inspect the Faure et al. AMN repository to establish the media, experimental
data, and training procedure used for E. coli growth prediction. Compare the
implementation with the paper and supplement. Focus on the findings needed to
understand and reproduce this experiment.

## Locations and output

Start from the root of this repository (`metabolic-NN`). All paths below are
relative to that root:

- AMN repository: `../amn_release` — go one directory up, then into the sibling
  `amn_release` repository.
- Paper: `docs/Faure etal 2023.pdf`.
- Supplement: `docs/Faure_supplementary.pdf`.
- Single findings file: `docs/reference/amn_repository_notes.md`.

Create `docs/reference/` if needed. Keep the AMN repository read-only and preserve
existing changes. Write only the findings file; do not retrain models or
regenerate datasets. Small calculations to verify counts and metrics are useful.

## Where to look and how to support findings

Read applicable repository instructions and the AMN README. Trace the relevant
notebook calls into `Library/` and inspect the data and saved parameters they use.
Start with:

- `Build_Experimental.ipynb`, `Build_Dataset.ipynb`, and their library functions;
  `Dataset_experimental/EXP110.csv` and `Dataset_input/iML1515_EXP.csv`.
- `Build_Model_AMN.ipynb`, `Build_Model_ANN_Dense.ipynb`,
  `Build_Model_RC.ipynb`, and `Library/Build_Model.py`.
- Relevant files in `Dataset_model/`, `Reservoir/`, and `Result/`; use
  `Figures.ipynb` to identify the artifacts behind paper results.

Record the AMN commit and relevant local modifications. Distinguish committed
code from local additions and edits when comparing with the publication. Follow
actual execution paths and notebook overrides rather than relying on comments
or function defaults. Cite code paths with functions/lines or notebook cells,
data files, and paper pages/sections beside the findings. Mark missing evidence
or uncertain conclusions explicitly; do not substitute settings from this
repository's FluxTransformer experiments.

## 1. Medium sources and uptake bounds

Return complete **lists** of:

- Variable carbon/organic sources.
- Fixed organic supplements, including glycerol and amino acids if present.
- Fixed base-medium components and other exchange inputs.
- Relevant closed/excluded sources, including glucose if applicable.

For each source, give its name, exact AMN exchange ID, corresponding unsplit
exchange ID where relevant, physical concentration if documented, and actual
uptake bound with units. Explain sign conventions and split-reaction mappings.
State how absent sources, oxygen, and any implicitly open exchanges are handled.

Trace how experimental composition or presence indicators become model inputs
and uptake constraints, including scaling constants. Distinguish physical
concentrations, nominal caps, and learned bounds. If experimental prediction,
simulation/pretraining, and mechanistic baselines use different medium settings,
report those differences. Explain the fixed supplements' contribution to the
available carbon supply.

## 2. Experimental data and condition counts

Identify the strain, assay, growth-rate target, and relevant growth conditions.
Describe how raw measurements become growth rates, how replicates are averaged,
and how uncertainty and filtering are handled. Trace the final experimental
file into the model's inputs and targets.

Compute from the data:

- Total unique medium conditions, training rows, and replicate counts.
- For **each variable carbon source**, conditions containing it, conditions with
  it as the sole variable source, and conditions containing it in a mixture.
- Fixed organic supplement coverage and condition counts by number of variable
  sources. Keep fixed supplements separate when counting mixture size.

Reconcile counts with the paper and explain overlapping source-membership
counts, exclusions, or duplicate media. Keep the authors' measured growth dataset
separate from other E. coli datasets; state whether those are used for training,
pretraining, or external evaluation.

## 3. Exact training and evaluation procedure

Describe the experimental-growth workflow for AMN-Wt, AMN-LP, AMN-QP, and relevant
ANN/reservoir comparisons. Separate simulated-data pretraining from training on
experimental growth rates. Give the effective settings for each variant:

- Input features and target preprocessing; architecture and trainable/frozen
  components.
- Exact optimized loss, including formulas, weighting, regularization, and
  mechanistic terms. Distinguish the training loss from evaluation metrics and
  the inner LP/QP objective.
- Optimizer, learning rate, batch size, epochs, early stopping, best-weight
  restoration, and relevant solver settings.
- Hyperparameter selection and the data used for it; seeds where available.

Verify the **10-fold CV** procedure from code: splitter, stratification if any,
shuffle, repetitions, and actual training/validation/test counts. Establish
whether conditions or replicates are split, whether weights are reset for each
fold, and whether preprocessing or tuning uses held-out data. Report any
replicate overlap or leakage supported by evidence.

Explain how held-out predictions and repeated runs are combined, how `Q2`/`R2`
and uncertainty are calculated, and which saved artifacts correspond to the
paper. Summarize the verified training sequence with actual parameter values.

## 4. Differences between the paper and implementation

Report material differences in media/bounds, data counts or processing, CV,
losses, model settings, and evaluation. For each difference, state the paper's
claim, the code/data evidence, and its consequence. Distinguish contradictions
from missing details, local modifications, and artifacts of uncertain provenance.
Check key reported metrics against saved predictions when practical.

## Report style

Organize the single findings file around the four topics above, ending with
important limitations and unresolved questions. Keep source lists explicit and
use compact tables only where they improve comparison. Avoid repeating findings
across lists, tables, and prose. Include enough evidence and calculation details
to verify the conclusions without documenting every unrelated code branch.
