# Prompt: Inspect the Tazza MINN nutrient conditions and Table 4 metrics

Inspect the Tazza et al. MINN repository to establish the **actual nutrient and
exchange conditions used in code**, compare them with the paper, and explain
how they differ from Faure's AMN experiment. Nutrient conditions and their
sampling are the main aim. A second required aim is to establish whether the
R2 reported in **Tazza Table 4** is regression R2 or squared Pearson
correlation, including the metric inherited from Goncalves et al.

## Locations and outputs

Start from the root of this repository (`metabolic-NN`). Paths below are
relative to that root:

- MINN repository: `../MINN` — one directory up, then into the sibling MINN
  repository. Check the actual filenames; the README's notebook names may
  differ from the files present.
- Tazza paper: `docs/tazza-minn.pdf`; `docs/tazza-minn.extracted.txt` is a
  navigation aid, but verify equations and tables against the PDF.
- Goncalves paper: `docs/Goncalves2023.pdf`.
- AMN comparison: `docs/reference/amn_repository_notes.md`,
  `docs/Faure etal 2023.pdf`, and `docs/Faure_supplementary.pdf`.
  Read `../amn_release` only when needed to verify an AMN comparison.
- Single primary findings file: `docs/reference/minn_repository_notes.md`.
- Maintained experiment notes: `docs/experiment_notes/MINN_training_notes.md`,
  `docs/experiment_notes/AMN_MINN_shared_reservoir_notes.md`,
  `docs/experiment_notes/iML1515_sampling_study_notes.md`, and
  `docs/experiment_notes/AMN_experiment_notes.md` where applicable.

Create `docs/reference/` if needed. Keep MINN, AMN, and any Goncalves source
repository read-only. Preserve existing changes in all repositories. Write the
findings file and update the relevant experiment notes when the audit yields
new findings, corrections, or stronger evidence, especially about Table 4.
Do not change generators, notebooks, models, datasets, or evaluation behavior;
do not retrain models or regenerate full datasets. Small in-memory calculations
on existing data and predictions are encouraged. If an artifact is unavailable,
record the resulting limit rather than inventing it or reconstructing a full
experiment.

## Where to look and how to support findings

Read applicable repository instructions, the MINN README, and the existing
MINN metric contract before investigating. Record the MINN commit and relevant
local modifications. Distinguish committed source, notebook overrides, saved
outputs, checkpoint metadata, and local additions. Start with:

- `MINN_reservoir_main.ipynb` and `MINN_c_balanced_main.ipynb`.
- `conf/MINN_reservoir.yaml`, the relevant balanced-model configurations,
  and their composed `conf/dataset/`, `conf/gem/`, `conf/model/`, and HPO
  settings. Trace Hydra overrides to the effective configuration.
- `src/utils/import_data.py`, `import_GEM.py`, `training_ishii.py`,
  `training.py`, `plots.py`, and the called model code in `src/nn_model/`.
- The actually loaded files in `GEMs/`, `data/ishii_data/`, `data/faure/`,
  and the reservoir checkpoint. Locate any simulation-generation,
  flux-fitting, reduction, or pFBA-comparison code reached from these paths.
- Existing flux targets and prediction artifacts, including
  `fluxomics.csv`, `fluxomics_iAF1260_reduced_split.csv`,
  `fluxomics_iAF1260_reduced_split_fit.csv`, `pFBA_ishii.csv`, and
  `minn_results/`; verify their roles rather than assuming their filenames
  establish Table 4 provenance.

Cite code paths with functions/lines or notebook cells, effective configuration
values, data files, and paper pages/sections beside substantive claims. Define
notebook cell numbering. Use primary sources if missing supplements or the
Goncalves code must be located online. Do not substitute this repository's
iML1515 FluxTransformer settings for the authors' MINN settings. Existing notes
are leads to verify, not proof of what generated a published table.

## 1. Nutrient conditions, exchange bounds, and simulated sampling

Return complete lists of variable carbon/organic sources, fixed organic
supplements, fixed inorganic/base-medium components, oxygen, and relevant
closed/excluded sources. Include compounds that remain implicitly available
through the GEM's original bounds. Do not stop at the learned reservoir's
input channels: identify the full effective medium seen by the solver.

For each component give its name, exact MINN exchange ID, corresponding
unsplit ID where relevant, physical concentration if documented, units,
direction, and effective lower/upper bounds. Explain positive split-reverse
magnitudes and their conversion to signed uptake bounds. Distinguish absence
from an unmeasured or unconstrained exchange. State how model loading, medium
resets, knockouts, preprocessing, and later overrides change the bounds.

Separate the following execution paths, using an explicit bound table:

1. Measured experimental conditions and their raw exchange-flux inputs.
2. Flux-fitted experimental inputs/targets and the fitting constraints.
3. Simulated reservoir pretraining inputs and flux labels.
4. Experimental MINN predictions and learned reservoir-control values.
5. Plain mechanistic baselines and MINN-reservoir-to-pFBA Table 4 comparisons.

Trace glucose, oxygen, CO2, ethanol, and acetate wherever present. Establish
whether each value is measured, fitted, sampled, learned, or a realized solver
flux, and whether it fixes a flux, supplies an uptake bound, or supplies a
secretion bound. Verify which learned channels become downstream constraints;
an input to the reservoir need not be a cap applied in the final solver.

For simulated data establish the complete sampling rule: eligible exchanges,
number active, range endpoints, integer/discrete/continuous distributions,
dependencies between channels, fixed supplements, seeds, requested/accepted
sample counts, infeasible-sample handling, and any train/test split. Determine
whether simulations are restricted to measured experimental conditions or
independently draw new combinations/caps. Verify this from code and saved
sample metadata where available. If only a pretrained checkpoint is released,
state which generation claims can and cannot be established.

For every solver path identify the actual GEM variant, objective reaction,
FBA versus pFBA call, optimum fraction, and any flux fitting or alternative
objective. Distinguish plain FBA, parsimonious optimization, neural mechanistic
updates, and optimization used to construct fitted targets. Function defaults
and the generic word "FBA" in the paper are insufficient evidence.

Compare author MINN conditions directly with **author AMN conditions** in a
compact table: glucose and other variable sources, fixed glycerol/amino acids,
oxygen, base nutrients, source-pattern selection, cap scaling, solver, and
target type. Verify the AMN finding that its iML1515 simulation uses the 110
experimental source patterns with 100 random cap draws each; do not describe
our unrestricted one-to-four-source sampler as the author AMN workflow.
Explain the biological and numerical implications of the differences. Put any
comparison with this repository's A/B/C generators in a separately labelled
adaptation column or paragraph.

## 2. Experimental data, model fitting, and condition coverage

Identify the experimental organism/strain, source study, media, growth or
cultivation conditions, perturbations/knockouts, and the units and roles of
omics and flux measurements. Keep Ishii conditions, Faure media, other
benchmarks, and simulated pretraining samples separate.

Compute condition/row counts and identify replicates, exclusions, duplicates,
KO groups, and measured glucose/oxygen coverage. Give the observed ranges of
the relevant exchange measurements and compare them with simulated bounds.
Do not equate exchange-flux measurements with supplied nutrient concentrations.

Trace raw fluxomics through splitting, mapping, filtering, scaling, and fitting
to the exact Table 4 reference targets. Establish which GEM and constraints
produced fitted targets, whether measured exchanges or biomass could move,
and whether the generation script is released. Quantify relevant raw-versus-
fitted changes. Do not present a locally reconstructed fit as proof of the
historical fitting procedure. Explain which inputs and targets each compared
method receives and whether they are equivalent.

## 3. Table 4 R2: paper definition, executed metric, and artifact provenance

Read Tazza's metric description and Table 4 caption/notes alongside the cited
Goncalves metric definition. Check the Goncalves equation and, if accessible,
its original metric implementation (`omics2flux` is a lead in the existing
notes; verify its identity, version, and relevant call path). Give a direct,
evidence-qualified answer for **each Table 4 row** that can be traced.

Distinguish these two formulas explicitly:

```text
Regression R2 = 1 - sum_i (y_i - yhat_i)^2 / sum_i (y_i - mean(y))^2
Pearson r^2   = corr(y, yhat)^2
```

Trace the metric from the notebook/table-producing call through every helper
and aggregation. Inspect `src/utils/plots.py:r2_metric` and `metrics_table`,
but also search for inline metrics, alternate utilities, Q2 labels, and saved
outputs. A function name or import of `r2_score` does not establish the executed
definition. Determine whether the code refits a line with slope/intercept,
clips predictions, reverses arguments, normalizes values, or handles exceptions
and constant vectors by returning zero. Verify how these choices affect scores.

Establish the evaluation axis, exact flux subset/order, raw versus fitted
targets, inclusion of supplied exchanges/biomass, per-condition versus
per-flux scoring, mean/population-or-sample SD, repeated-run averaging, and
pooled versus fold-wise calculations. Reconcile actual condition/flux counts
with the paper rather than assuming 29 conditions and 47 fluxes universally.
Inspect leave-one-out and inner-fold selection, preprocessing, weight resets,
and which prediction artifacts genuinely contain held-out outputs.

Where existing matched predictions are available, recompute **both regression
R2 and Pearson r squared on the same rows and fluxes**, first preserving the
author processing/aggregation to reproduce the reported result. Separately
show the effect of material processing choices and preserve negative
regression scores and explicitly undefined cases in the audit. Include MAE,
RMSE, and normalized error as checks that the artifact and aggregation align.
Use a small shifted/scaled prediction example if useful to show why the two
R2 definitions differ; do not confuse squared correlation with predictive
accuracy on the identity line.

Compare recalculated means/SDs with the published Table 4 values at their
reported precision. Classify conclusions as: paper-stated definition,
verified released-code behavior, reproduced artifact calculation, or inference
about the published number. Numerical closeness alone cannot prove how a
published row was computed. If publication artifacts or exact settings are
missing, give the strongest supported answer and the remaining uncertainty.
Keep Table 2 results separate from Table 4.

## 4. Differences, implications, and maintained-note updates

Report material differences in nutrient availability, sampling, fitted targets,
GEMs, solver settings, evaluation, and R2 definitions. For each give the
paper's statement, implementation/data evidence, consequence, and confidence.
Keep contradictions separate from undocumented details, local modifications,
and differences introduced by our FluxTransformer adaptations.

Update the maintained notes **only with verified new or corrected findings**:

- `MINN_training_notes.md`: MINN conditions, fitting/solver provenance, and
  the Table 4 metric conclusion with its evidence limits.
- `AMN_MINN_shared_reservoir_notes.md` and
  `iML1515_sampling_study_notes.md`: implications for comparing reservoirs,
  sampling distributions, and historical versus current metrics.
- `AMN_experiment_notes.md`: any verified correction to the AMN comparison.

Search for affected claims throughout `docs/experiment_notes/`. Reconcile
contradictions, link to the primary findings file, and distinguish superseded
historical results from current interpretation. Preserve the current local
regression-R2 evaluation contract unless an explicit later task authorizes a
behavior change. Do not relabel old Pearson-based results as regression R2
or claim an exact published-score reproduction without matching evidence.

## Report style and verification

Organize `docs/reference/minn_repository_notes.md` around the four topics above,
with nutrient conditions first, a direct Table 4 metric verdict, and a closing
list of important limitations/unresolved questions. Keep complete source lists
and effective bound values explicit. Use compact tables for execution-path
and AMN/MINN comparisons; avoid documenting unrelated branches or repeating
the same finding in several sections. Include enough evidence and calculation
details to reproduce the conclusions. Finish by checking the documentation
diff, links, and `git diff --check`, and report which notes changed and what
remains unverified.
