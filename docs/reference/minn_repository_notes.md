# Tazza MINN nutrient conditions and Table 4: repository inspection

Inspected 6 October 2026. MINN's biological benchmark consists of **29 glucose-minimal chemostat conditions**, not AMN's 110 mixtures of ten variable carbon sources with fixed organic supplements. The released MINN reservoir notebook uses **fitted** glucose/oxygen inputs and five fitted exchange targets. Its metric is **squared Pearson correlation**, as is Goncalves's released metric, despite the regression-R2 formula in the latter paper. The exact two **Table 4** rows cannot be regenerated from the released artifacts: the downstream pFBA calculation and its predictions are missing, and the reservoir export has a five-value/two-column mismatch.

## Scope and provenance

- Read-only MINN checkout: `../MINN`, commit **`c2fc1098313827529b8e3cc3679649f255f16f88`**, 12 May 2025, “Update Dockerfile”; clean at inspection start and finish. This release predates the final paper. No local additions were substituted for author code.
- Tazza et al. (2025), *MINN: A metabolic-informed neural network for integrating omics data into genome-scale metabolic modeling*: local `docs/tazza-minn.pdf`, [published article](https://doi.org/10.1016/j.csbj.2025.08.004), [publisher supplement](https://ars.els-cdn.com/content/image/1-s2.0-S2001037025003265-mmc1.pdf). Paper page numbers below are PDF positions, with printed pages 3609--3617; supplement pages refer to its own pagination. The Table 4 page, computational-setup page, and Goncalves equation were inspected visually.
- Goncalves et al. (2023), *Predicting metabolic fluxes from omics data via machine learning: Moving from knowledge-driven towards data-driven approaches*: `docs/Goncalves2023.pdf`, [article](https://doi.org/10.1016/j.csbj.2023.10.002). Its linked `dmgoncal/omics2flux` source was inspected at **`5366d6955f0bbb51eb9fed06ce5d616bda1b4042`**, 16 October 2023. Downloaded `utils.py`, `pfba.py`, `analysis.py`, `data.py`, and `models.py` match that commit's Git blob hashes.
- `MINN:` paths below are relative to `../MINN`; `Goncalves:` paths refer to that pinned `omics2flux` commit. Notebook cells are **one-based positions including markdown cells**, not execution counts. Both released MINN notebooks have no saved execution outputs.
- AMN comparison: [AMN repository inspection](amn_repository_notes.md), checked again against author `Build_Dataset.ipynb` cell 13 and `Dataset_model/iML1515_UB.npz`. The AMN checkout's existing notebook metadata/output changes and local additions were preserved.
- Applicable repository instructions and existing experiment notes were read. Research Base supplied caveats about fitted targets, inputs, and checkpoint provenance; current author files determine the findings below. No training, full dataset generation, flux refitting, or model changes were performed. Calculations used existing CSVs, XMLs, metric-function bodies, and static checkpoint metadata. Project changes are limited to this report and relevant experiment notes; durable caveats were also added to Research Base's existing `wiki/experiments/iml1515-frozen-reservoir-transfer.md` page.

## 1. Nutrient conditions, exchange bounds, and simulated sampling

### The five channels describe uptake and secretion, not five carbon sources

The paper's reservoir channels are glucose, oxygen, CO2, ethanol, and acetate. Glucose is the supplied organic carbon source; oxygen is an uptake; CO2, ethanol, and acetate are secretion channels. The released 587-reaction model does not allow import of those three products. None of AMN's fixed glycerol, alanine, proline, threonine, or glycine supplements is present in this model. These are different biological and numerical environments, not different subsets of one common organic-source pool.

Split exchange fluxes are nonnegative. A `_rev` import with positive magnitude `u` corresponds to unsplit flux `-u`; a `_fwd` export has unsplit flux `+v`. The `R_` prefix is part of the author SBML ID; corresponding unsplit BiGG IDs below omit it. For two retained directions, net unsplit flux is forward minus reverse. CBMPy's split reaction stoichiometry, rather than suffix alone, establishes the direction. [MINN: `src/utils/import_GEM.py:5--12`; `GEMs/iAF1260_split_FBA_reduction.xml`, exchange reactants/products.]

Fluxes and exchange bounds are in **mmol gDW^-1 h^-1**; biomass is h^-1. The benchmark provides measured flux rates, not nutrient-concentration-to-uptake conversions. Tazza specifies glucose-minimal cultivation, but neither these CSVs nor the inspected release provides a quantified physical recipe for each modeled base component. Do not reinterpret any bound below as a supplied concentration.

| Component | MINN split ID | Unsplit BiGG ID | Raw observed magnitude range | Fitted magnitude range | Bounds in released FBA-reduced XML |
|---|---|---|---:|---:|---|
| D-glucose uptake | `R_EX_glc__D_e_rev` | `EX_glc__D_e` | 1.34--13.34 | 2.7630--12.8901 | `[0,8]` import; no glucose-export exchange retained |
| Oxygen uptake | `R_EX_o2_e_rev` | `EX_o2_e` | 2.20--15.53 | 6.0204--26.2417 | `[0,999999]` import; no oxygen-export exchange retained |
| CO2 secretion | `R_EX_co2_e_fwd` | `EX_co2_e` | 0.31--10.83 | 0.8565--19.6169 | `[0,999999]` export; no CO2-import exchange retained |
| Ethanol secretion | `R_EX_etoh_e` | `EX_etoh_e` | 0--0.15 | 0--0.1407 | `[0,999999]` export only |
| Acetate secretion | `R_EX_ac_e` | `EX_ac_e` | 0--1.93 | 0--1.9350 | `[0,999999]` export only |

Ranges are calculated over all 29 rows of `data/ishii_data/fluxomics_iAF1260_reduced_split.csv` and `_fit.csv`. XML bounds are **stored model defaults**, not demonstrated settings of the unpublished simulation generator or Table 4 solver. In particular, the fitted `WT_0.7h-1` glucose exceeds the XML default of 8, so a reproduction cannot silently inherit that default and claim to impose all fitted values.

### Complete basal import list in the FBA-reduced model

All 16 following inward exchanges have XML bounds **`[0,999999]`**, with no fixed positive flux. Their concentrations are undocumented in the release. They are available components, not independently sampled reservoir inputs and not fixed input-feature values of 2.2, 10, or 50.

| Component | Exact inward MINN ID | Unsplit BiGG ID |
|---|---|---|
| Calcium | `R_EX_ca2_e_rev` | `EX_ca2_e` |
| Chloride | `R_EX_cl_e_rev` | `EX_cl_e` |
| Cobalt(II) | `R_EX_cobalt2_e_rev` | `EX_cobalt2_e` |
| Copper(II) | `R_EX_cu2_e_rev` | `EX_cu2_e` |
| Magnesium | `R_EX_mg2_e_rev` | `EX_mg2_e` |
| Manganese(II) | `R_EX_mn2_e_rev` | `EX_mn2_e` |
| Molybdate | `R_EX_mobd_e_rev` | `EX_mobd_e` |
| Ammonium | `R_EX_nh4_e_rev` | `EX_nh4_e` |
| Iron(II) | `R_EX_fe2_e_rev` | `EX_fe2_e` |
| Iron(III) | `R_EX_fe3_e_rev` | `EX_fe3_e` |
| Phosphate | `R_EX_pi_e_rev` | `EX_pi_e` |
| Zinc(II) | `R_EX_zn2_e_rev` | `EX_zn2_e` |
| Sulfate | `R_EX_so4_e_rev` | `EX_so4_e` |
| Water | `R_EX_h2o_e_rev` | `EX_h2o_e` |
| Protons | `R_EX_h_e_rev` | `EX_h_e` |
| Potassium | `R_EX_k_e_rev` | `EX_k_e` |

Together with glucose and oxygen, these give **18 inward exchanges**. To account for all **34 exchange reactions** in the 587-reaction XML, the complete outward list is: formate (`R_EX_for_e`), allantoin (`R_EX_alltn_e`), acetate (`R_EX_ac_e`), dihydroxyacetone (`R_EX_dha_e`), ethanol (`R_EX_etoh_e`), xanthine (`R_EX_xan_e`), pyruvate (`R_EX_pyr_e`), succinate (`R_EX_succ_e`), KDO(2)-lipid IV(A) (`R_EX_kdo2lipid4_e`), D-lactate (`R_EX_lac__D_e`), CO2 (`R_EX_co2_e_fwd`), ammonium (`R_EX_nh4_e_fwd`), iron(III) (`R_EX_fe3_e_fwd`), phosphate (`R_EX_pi_e_fwd`), water (`R_EX_h2o_e_fwd`), and protons (`R_EX_h_e_fwd`). Each outward reaction has `[0,999999]`; its unsplit ID follows by removing `R_` and any `_fwd` suffix. These export opportunities are not nutrient additions. [Evidence: complete XML exchange enumeration, including stoichiometric direction and bound-parameter references.]

**Excluded AMN sources:** ribose, maltose, melibiose, trehalose, fructose, galactose, glycerol, alanine, proline, threonine, and glycine have no exchange reaction in this reduced model. Acetate, D-lactate, succinate, and pyruvate have export-only exchanges. Thus none of AMN's ten variable sources is an additional import here. Cobalamin, sodium, tungstate, and selenate/selenite are also absent as exchange reactions in this reduced model; do not supply them as retained MINN basal inputs merely because they appear in another GEM.

### Different model files have different effective availability

`conf/MINN_reservoir.yaml` selects `gem: iAF1260_split_FBA`; `src/utils/import_data.py:58--59` loads **`GEMs/iAF1260_split_FBA_reduction.xml`**. Its dimensions are 587 reactions and 493 species, with biomass objective `R_BIOMASS_Ec_iAF1260_core_59p81M`. This is the paper's Table 4 reservoir model, not iML1515.

- `GEMs/iAF1260_reduced_split.xml`, the default balanced-model GEM, has 1,873 reactions, 1,032 species, and 117 exchange directions. Its complete inward availability is the same 16 basal imports plus glucose `[0,8]`, oxygen `[0,18.5]`, and **CO2 import `[0,999999]`**. Extra export directions do not introduce extra organic imports.
- `GEMs/iAF1260.xml` has 2,382 reactions, 1,668 species, and 299 exchanges. In addition to those 19 inward components, it permits sodium and tungstate at lower bound `-999999`, and **cob(I)alamin at `-0.01`**. Most unsplit exchange upper bounds are `999999`. Its stored glucose/oxygen lower bounds are `-8`/`-18.5`.
- Goncalves's original `sbml/iAF1260.xml` has the corresponding 22 available imports with older names such as `EX_glc_e_`, `EX_o2_e_`, and `EX_cbl1_e_`; it is a separate full-model baseline source. Its default uptake magnitudes are likewise 8/18.5 for glucose/oxygen, 0.01 for cobalamin, and 999999 for the other available components.

These are XML availability audits. **The released neural code does not enforce all these XML bounds.** `GEM.build_GEM_matrices` extracts `S`, `Pin`, and `Pref`; it does not turn every lower/upper bound into neural constraints. With `kos_genes: False`, `load_ishii` sets `Vin=inf` and `Pin` selects only CO2 export. That upper-bound penalty is therefore inactive. Eight soft mechanistic updates reduce stoichiometric and negativity errors; they do not certify a feasible flux solution or fix measured glucose/oxygen. Bounds and unavailable reaction directions must be distinguished from what the learned layer actually constrains. [MINN: `src/utils/import_GEM.py:29--64`; `import_data.py:69--91`; `src/nn_model/amn_qp.py:17--76,125--133`; `amn_qp_old_code.py:16--66,94--134`.]

### Execution-path and solver distinctions

| Path | Inputs/targets and actual constraint role | What is established about the solver |
|---|---|---|
| Raw Ishii observations | Glucose/O2 are measured uptake rates; other fluxes and biomass are measured/MFA targets | Experimental data, not FBA-generated targets |
| Fitted experimental file | Whole 47-flux vectors adjusted; glucose/O2/CO2 can change substantially; biomass unchanged in saved file | Paper describes minimum Euclidean-distance feasible fitting; generation code/settings are absent |
| Reservoir pretraining | Paper describes five randomized exchange values and 2,000 simulated full-flux solutions | Paper calls these FBA simulations; neither a generation script nor its dataset is released, so plain FBA versus pFBA and detailed constraints are unverified |
| Released reservoir experimental training | Omics plus two fitted uptakes as features; five fitted exchanges as targets; front network produces five latent values | Frozen neural approximator and soft mechanistic updates; no FBA/pFBA solve in the notebook |
| Goncalves pFBA baseline copied into MINN | Glucose and O2 set to measured **equalities**, relevant gene knocked out, other full-model bounds retained | Actual `cobra.flux_analysis.pfba(model)` call, default optimum fraction 1.0; 45 evaluated fluxes |
| Tazza Table 4 pFBA / reservoir+pFBA | Paper says two uptake inputs versus those inputs plus CO2/ethanol/acetate constraints, evaluated on 47 fluxes | pFBA is the stated method; actual downstream code, equality-versus-cap choices, KO application, fraction, and prediction artifacts are not released |

The Goncalves baseline genuinely fixes `reaction.bounds=(measured,measured)`, rather than merely setting a maximum uptake cap. Its script retains a dilution field but does not constrain biomass to it. Gene deletion occurs within a per-condition `with model:` context, and `pfba(model)` maximizes the model biomass objective before minimizing total flux. This does **not** prove the unpublished Table 4 baseline used identical settings. [Pinned Goncalves `pfba.py:57--92`, [source](https://github.com/dmgoncal/omics2flux/blob/5366d6955f0bbb51eb9fed06ce5d616bda1b4042/pfba.py#L57).]

### What can be established about reservoir sampling

The main paper, Section 2.3.1 and Fig. 2 (PDF p. 4), specifies **2,000 FBA simulations** with randomly assigned glucose, oxygen, CO2, ethanol, and acetate values within observed Ishii ranges. It describes new numerical exchange conditions, not draws restricted to a catalog of 29 measured vectors. However, it does not identify the exact distribution, independence, range source (raw/fitted), integer rounding, zero handling, feasibility/retry policy, split, seed, or FBA/pFBA implementation. The supplement adds computational and HPO details but no reproducible sampling specification.

The tracked release contains **no reservoir simulation generator or 2,000-row iAF1260 training dataset**. The only reservoir artifact is `pretrained_block_reservoir_state_dict.pth`. Static ZIP/pickle-opcode inspection confirms weights `(500,5)`, bias `(500,)`, weights `(587,500)`, bias `(587,)`, consistent with the code's 5→500→587 network. It contains no sample count, channel order, media, solver, bounds, seed, or training-split metadata. No untrusted checkpoint objects were executed.

`data/faure/Reference_FBA_simulated_data.csv` is **1,000 × 28**, with toy channels `EX_C6_rev_ub`, `EX_C2_rev_ub`, and `EX_N_rev_ub`; it is not this reservoir dataset. `data/faure/EXP110.csv` and `iML1515_EXP.csv` are separate Faure tables, not extra Ishii pretraining conditions. A broad search of tracked notebooks, Python files, and configurations found no generation or downstream pFBA/fitting path. Absence in this release does not establish absence in the authors' private historical workflow.

### Author MINN versus author AMN

| Property | Author MINN | Author AMN iML1515 |
|---|---|---|
| Biological task | 29 glucose-minimal chemostat conditions, WT growth-rate series and 24 KOs; multi-flux prediction | 110 DH5-alpha media; experimental growth-rate prediction |
| Variable organic supply | Glucose; three reservoir channels are products, not extra nutrient sources | Ten variable carbon sources: ribose, maltose, melibiose, trehalose, fructose, galactose, acetate, D-lactate, succinate, pyruvate |
| Fixed organic supplements | None of glycerol/alanine/proline/threonine/glycine in the FBA-reduced model | All five present, each simulated inward cap 2.2 |
| Oxygen | Experimental uptake varies; part of the five-channel simulation described in the paper | Fixed simulated inward cap 2.2 |
| Base availability | Sixteen retained basal imports have XML cap 999999; neural code does not enforce these caps | All 28 fixed simulated inputs have cap 2.2 |
| Simulation source selection | Paper: 2,000 randomized five-channel exchange conditions; exact generator unavailable | Verified: 110 experimental source patterns × 100 cap draws = 11,000 rows |
| Variable numeric values | Paper: within observed exchange ranges; raw/fitted choice and distribution unverified | Verified discrete caps `(k+1)*j*2.2/99`, `j=1,...,99` |
| Solver provenance | Pretraining FBA variant unverified; downstream pFBA stated but implementation missing | iML1515 pretraining and inspected downstream comparator explicitly pFBA |
| Reservoir structure | Frozen 5→500→587 network plus inherited soft QP refinement; iAF1260-FBA reduction | Frozen simulated 38-input, 550-output iML1515 reservoir |

The AMN presence restriction was rechecked directly: exactly 110 patterns, each with 100 simulated rows, and saved method `pFBA`. It is not arbitrary random one-to-four-source selection. MINN's glucose pathway and secretion context therefore differ fundamentally from the author's AMN prior; sharing the term “reservoir” does not establish common source support or caps.

**Local adaptations:** our Tazza-style B generator uses iML1515, independently sampled integer glucose/O2/CO2/ethanol/acetate caps of 1--15/1--20/0--15/0--1/0--3, base rate 50, and pFBA at 0.999. These are deliberately widened approximations, not verified original MINN bounds. Their upper envelopes cover the **raw** values above but not the fitted oxygen maximum 26.2417 or CO2 maximum 19.6169. Our A and shared-C `faure` regimes independently select one-to-four sources, unlike author AMN; their base caps and solver choices also differ. [Local generators and `docs/experiment_notes/iML1515_sampling_study_notes.md`; author comparison established above.]

## 2. Experimental conditions, preprocessing, and fitting

### Coverage and biological scope

`fluxomics.csv` and its 47-flux split/fitted counterparts each contain **29 distinct condition rows**: `REF` (WT at D=0.2 h^-1), WT at D=0.1/0.4/0.5/0.7 h^-1, and 24 KO conditions at D=0.2 h^-1. The KO labels are `galM`, `glk`, `pgm`, `pgi`, `pfkA`, `pfkB`, `fbp`, `fbaB`, `gapC`, `gpmA`, `gpmB`, `pykA`, `pykF`, `ppsA`, `zwf`, `pgl`, `gnd`, `rpe`, `rpiA`, `rpiB`, `tktA`, `tktB`, `talA`, `talB`. No duplicate full flux vectors or missing values were found in these three tables. These are condition-level summaries, not a released replicate-level assay table. Replicate counts/filtering cannot be reconstructed from the MINN CSVs. Exact K-12 substrain and nutrient concentrations are not specified by the inspected release; do not describe all rows as unmodified MG1655.

The source study is Ishii et al. (2007), *Multiple high-throughput analyses monitor the response of E. coli to perturbations*, as identified by Tazza Section 2.1. Its 47 outputs comprise 37 central-metabolic fluxes, nine exchanges, and biomass. MINN releases 79 transcriptomic features and 60 proteomic features, plus the two uptake features: **141 input columns** after `load_ishii` merges by experiment ID. Omics uncertainty tables are not included in this loader. All three feature/target/`Vin` arrays are cast to float32. [MINN `src/utils/import_data.py:26--93`; source CSV dimensions; Tazza PDF p. 2.]

All 29 conditions have positive raw glucose, O2, and CO2 flux magnitudes. Raw ethanol secretion is positive in five conditions (fitted: three); raw and fitted acetate are positive in three. These zero-containing secretion measurements are not binary media-design variables. There is no one-to-four-carbon-source mixture count analogous to AMN.

### Raw, split, and fitted data are distinct

The MINN raw 29×47 flux values exactly match the upstream fluxomics source referenced by Goncalves `data.py`, after transposition. Raw negative PGK/PGM/RPI/SUCOAS/ACKr/LDH_D/ACALD/ALCD2x and uptake rates become positive magnitudes in the selected reverse columns; aliases are also updated. Verified sign conversion gives numerical equality between corresponding raw and unfitted split values. The splitting script itself is not released. [MINN raw/split CSVs; [Goncalves data source declaration](https://github.com/dmgoncal/omics2flux/blob/5366d6955f0bbb51eb9fed06ce5d616bda1b4042/data.py#L2).]

The fitted file differs from the unfitted split file in **840 of 1,363 entries** at absolute tolerance `1e-8`. It is not simply a renamed version of the measured data:

| Channel | Conditions changed / 29 | Maximum absolute change |
|---|---:|---:|
| Glucose | 28 | 4.5220 |
| Oxygen | 18 | 11.3717 |
| CO2 | 29 | 8.7869 |
| Ethanol | 5 | 0.0100 |
| Acetate | 2 | 0.0050 |
| Biomass | 0 | 0 |

For `WT_0.1h-1`, glucose changes **1.34→5.862**, O2 **2.2→6.8486**, and CO2 **0.31→0.8565**, while biomass remains 0.1. This is important when interpreting a growth-maximizing downstream solver: fitted substrate availability need not reproduce the supplied target growth automatically. The paper's Euclidean fitting statement and unchanged biomass do not establish a particular unpublished weighting scheme or prove which quantities were held fixed rather than coincidentally preserved. No fitting script is present. [Tazza Section 2.1, PDF p. 2; source CSV comparisons.]

The local `MINN_data/fluxomics_iAF1260_reduced_split_fit.csv` is byte-identical to the release's fitted file; SHA-256 **`fbdd62504669e3f5c05178c788798973efde9985e75761a052136d7ac836e728`**. This verifies copying and the configured reservoir input/target source. It does **not** independently prove that this file generated the publication's Table 4 targets. A local iML1515 weighted-L1 refit is a sensitivity analysis, not a recreation of the author's minimum-Euclidean historical fitting procedure.

### Effective released experimental reservoir training

The composed `MINN_reservoir` configuration selects dataset file `conf/dataset/only_2_EX_&_5_reference.yaml`, whose `dataset_name` is `ref_47_fluxes_fit`. `load_ishii` therefore reads the fitted 47-flux CSV, appends fitted glucose/O2 to omics, and drops 42 targets, leaving **five reference targets in order: glucose, O2, CO2, ethanol, acetate**. Biomass and all 37 internal fluxes are excluded from this reservoir's experimental training loss. The configuration's `metric_fluxes` nevertheless lists all 47 columns, but the reservoir notebook never calls the final 47-flux metric table.

`train_test_evaluation` builds a front MLP **141→h→5**, ReLU, dropout, linear output, with h selected from 200/250/300. The frozen network is 5→500→587 with dropout 0.25 and a linear output, followed by inherited eight soft QP updates (step 0.01, decay 0.9). Its weights are frozen, but its dropout follows the outer train/eval mode. The five front outputs have no positivity activation or hard bound before entering the frozen block. The checkpoint lacks channel-identity metadata; the intended channel order comes from the paper/target list, not its tensor shapes.

The active outer loss is normalized Euclidean prediction error over the **five projected exchange targets**, because `conf/model/amn_qp_reservoir.yaml` uses `model_name: amn_qp_divided_loss`; `get_loss` selects `L=L1` for that name. Mechanistic loss is calculated diagnostically and used inside the frozen block's soft updates, not added to this outer loss. Supplied glucose/O2 features are also reference targets; they are not hard-injected as equalities into the predicted flux vector. [MINN `src/utils/training.py:103--132`; `src/nn_model/amn_qp.py:17--76,172--222`; old QP implementation; reservoir model configuration.]

The notebook performs 29 outer leave-one-out splits (28 train/1 held out), with an inner **five-fold shuffled KFold**, seed 12345. On 28 rows, inner folds train/validate on 22/6 for three folds and 23/5 for two. MinMax scaling fits only each fold's training features; targets are unscaled. A new front network is instantiated per fold and the same pretrained weights are reloaded. HPO uses seeded **Optuna TPESampler**, 15 trials, selecting mean inner-fold data-driven test loss; the supplement describes random search instead. Adam trains for 100 epochs, batch size 5; candidate LR 0.0002/0.0005/0.0007, dropout 0.1/0.25/0.5, weight decay 0/0.0001/0.001. No early stopping or best-epoch restoration is implemented in the called trainer; testing occurs after epoch 100. A misleading comment about choosing a best epoch does not change that. [MINN `hpo.py:26--53`, `training.py:78--163`; Tazza supplement pp. 6,8.]

The fitted target creation occurs before CV and is not fold-specific. It adjusts each condition using that condition's measured flux vector, including the uptake features. Therefore the declared measured-input deployment scenario and the released fitted-input benchmark must be distinguished. The evidence establishes target-derived input preprocessing; it does not establish leakage of other conditions' targets into a held-out fold. No training was rerun and no held-out claim is inferred merely from a prediction filename.

**Other released branch:** the default balanced notebook composes `MINN_c_balanced`, dataset `ref_47_fluxes`, GEM `iAF1260_reduced_split`. Despite that dataset name, `load_ishii`'s non-fit branch reads `fluxomics_iAF1260_reduced_split_filtered_iNF517.csv` (36 fluxes), not the full 47-flux split CSV. Its removal list is empty, so it trains against 36 targets while its final metric configuration selects 45. This is a released-loader discrepancy, not evidence of the historical paper's intended supervision. [MINN `import_data.py:28--39`; `conf/dataset/ref_47_fluxes.yaml`; balanced notebook cells 5,7,9.]

## 3. Table 4 R2: definition, implementation, and reproducibility

### Direct metric verdict

**Released MINN and Goncalves code calculate Pearson r squared, not predictive regression R2.** Goncalves Section 2.5, Eq. 4 (PDF p. 3/printed 4962) explicitly supplies `1-SSE/SST`. Tazza Section 2.4 (PDF p. 5/printed 3613) adopts the same metrics and calls R2 a regression coefficient. Neither passage specifies squared Pearson correlation. Their code disagrees with that stated definition.

```text
Regression R2 = 1 - sum((truth-prediction)^2) / sum((truth-mean(truth))^2)
Pearson r^2   = linregress(truth, prediction).rvalue^2
```

Both helpers iterate over **conditions/rows**, calculate a correlation across the selected fluxes within that condition, and return `r**2`. `linregress` also estimates slope and intercept, but neither is used to correct the supplied predictions. Exceptions become zero; other constant-vector behavior depends on SciPy. Ordinary dataset rows in these calculations are nonconstant. A small audit example gives Pearson r²=1 for `[1,2,3]→[11,12,13]`, while regression R²=-149. High correlation therefore does not establish calibrated flux predictions. [MINN `src/utils/plots.py:107--122`; [pinned Goncalves `utils.py:196--220`](https://github.com/dmgoncal/omics2flux/blob/5366d6955f0bbb51eb9fed06ce5d616bda1b4042/utils.py#L196).]

For a nonconstant truth vector, a constant prediction still has a defined regression R2; Pearson correlation is undefined when either vector is constant. Constant truth makes the regression denominator zero. Author exception/zero handling can therefore hide degenerate cases, and it cannot be carried over as a valid predictive-R2 score. None of the matched 45-flux rows above requires an undefined-case exclusion.

MINN's notebook cell 7 uses this helper for its LOO **five-reference-flux** Q2 diagnostic. The balanced notebook's final `metrics_table` computes per-row MAE, RMSE, normalized error, and the same correlation; it first clips predictions to nonnegative values, then reports mean and **population SD (`np.std`, ddof=0)**. NE replaces undefined/infinite values with zero. No predictive `r2_score` call occurs in these executed paths. Goncalves `pfba.py`, `models.py`, and `analysis.py` call its same correlation helper. These source traces establish release behavior, not the absent Table 4 downstream computation.

### Published rows and exact evidence boundaries

| Tazza Table 4 row | Published R2 / MAE / RMSE / NE, mean ± SD | Audit conclusion |
|---|---|---|
| pFBA | 0.892±0.127 / 0.496±0.353 / 0.836±0.625 / 0.306±0.367 | Common released metric is Pearson r²; no matched 29×47 Table 4 pFBA artifact or calculation is supplied |
| MINN-reservoir + pFBA | 0.910±0.091 / 0.445±0.261 / 0.740±0.421 / 0.253±0.159 | Same inherited-metric evidence; extra-constraint predictions and final pFBA solutions are absent |

The caption specifies means/SDs over 29 LOO conditions; Section 3.3 says 47 fluxes. The reservoir notebook ends at **cell 9**, exporting `Vin_reservoir_final` into a DataFrame with only `['R_EX_etoh_e','R_EX_ac_e']`. However, the called trainer has output size **5**, and `test_step` appends all five front-network values. A `(29,5)` result cannot be assigned two column names; an equivalent in-memory shape check raises `ValueError`. This line is not evidence that a two-constraint pFBA workflow was successfully run. The paper specifies three additional constraints, including CO2, but the release provides neither their actual mapping/bound assignment nor the following pFBA solve/metric table.

The strongest supported answer is: **the implemented metric inherited from Goncalves is Pearson r², so that is the evidence-based interpretation of MINN's released R2/Q2 reporting; Table 4 most likely follows it, but its particular numbers remain an inference rather than an artifact-level proof.** There is no source support for treating those numbers as independently verified `1-SSE/SST` scores. Conversely, the missing final workflow prevents a categorical claim about every private historical computation behind Table 4.

### Recalculation of the predictions that actually are released

`MINN:data/ishii_data/pFBA_ishii.csv` is **45×29**, excluding glucose and oxygen. It is byte-identical to pinned Goncalves `results/pFBA_ishii.csv` (SHA-256 **`529ca084c65dc94368370d5698f90f5dc43a06afa682f04249977c1365d0ff5f`**). These are full-iAF1260 predictions from the inherited baseline, not the reduced reservoir+pFBA outputs.

Alignment was verified against raw target column names, original source values, Goncalves's 29 ordered `dict_of_settings` entries, and every glucose/O2 value. The simulation headers use locus IDs while targets use gene names; 22 KO aliases match the MINN GEM gene annotations directly. Two require provenance caveats: `b4395` is labelled `ytjC` by this GEM versus `gpmB` in the data, and **`b0118` is annotated `acnB` but occupies the `gapC` data position**. The latter is a source-level KO-identity discrepancy, not a safe automatic gene-name conversion. The recalculation preserves the author's positional metric comparison; it does not validate that biological mapping or change any knockout. [Goncalves `pfba.py:13--44,75--92`; original linked fluxomics; MINN full-model gene products.]

All calculations below use the same 29 conditions and the specified **45 fluxes**; they are not Table 4 scores. Regression R2 was computed directly from SSE/SST and checked against scikit-learn on these nonconstant rows. The actual author metric-function bodies were also evaluated without importing training or remote-service code.

| Existing artifact / target encoding | Pearson r² mean ± SD | Regression R2 mean ± SD | MAE mean ± SD | RMSE mean ± SD | NE mean ± SD |
|---|---:|---:|---:|---:|---:|
| Goncalves pFBA / original signed raw targets | **0.822702±0.156467** | **0.778061±0.228708** | 0.692122±0.732996 | 1.057549±1.028894 | 0.380735±0.184620 |
| Same pFBA / raw targets converted to selected positive split magnitudes | 0.753712±0.207453 | 0.666452±0.343767 | 0.692122±0.732996 | 1.057549±1.028894 | 0.380735±0.184620 |
| Same pFBA / fitted split targets, predictions clipped after direction mapping | 0.801206±0.215932 | 0.729678±0.348475 | 0.630331±0.693920 | 0.923960±0.946706 | 0.332481±0.208942 |
| Released balanced-MINN 29×45 predictions / raw split targets, author nonnegative clipping | 0.944780±0.062826 | 0.709765±0.742013 | 0.481273±0.483451 | 0.695425±0.680191 | 0.276072±0.279884 |

The first row reproduces **Tazza Table 2's copied Goncalves pFBA row**, including all four reported means/SDs at three decimals: **0.823±0.156, 0.692±0.733, 1.058±1.029, 0.381±0.185**. Regression R2 does not reproduce that R2 entry. This provides artifact-level confirmation of the inherited Pearson metric, but specifically for **Table 2**. `pFBA_ishii.csv` is not an unexplained approximation to Table 4's 0.892 row.

The sign-conversion row demonstrates another comparison hazard: applying component-wise sign reversals to truth and predictions preserves MAE/RMSE/NE but changes their centered variance and correlation. A signed-flux benchmark and a selected-positive-split benchmark can therefore have different R2/correlation for physically corresponding fluxes. Flux direction conventions and the evaluated subset must match before comparing reported scores.

The balanced artifact, `data/ishii_data/minn_results/df_pred_minn_balanced.csv`, has 45 named flux columns but no condition identifiers or fold metadata. Its comparison assumes the released experimental row order; its recalculated metrics do not exactly reproduce the paper's balanced Table 2 row. Removing author prediction clipping gives Pearson r²=0.944679±0.062661, regression R2=0.709310±0.742300, MAE=0.487075±0.485740, RMSE=0.696163±0.680314, NE=0.276404±0.280002. This artifact is not a reservoir-cap or final-pFBA artifact and cannot identify Table 4's second row.

The local iML1515 baseline previously recorded as Pearson r² **0.892825±0.132254** (regression R2 **0.658478±1.189478**) is a separate adaptation against the MINN-fitted targets. It uses another GEM and does not match Table 4's baseline SD of 0.127. Those historical local results were not rerun in this audit and remain scoped to their earlier verification. Numerical proximity is not publication provenance.

## 4. Differences, implications, and maintained-note updates

| Statement or assumption | Verified release evidence | Consequence / confidence |
|---|---|---|
| MINN randomly simulates conditions within observed exchange ranges | Paper gives 2,000 rows and five channels; generator/data absent | Numerical randomization is paper-described; exact implemented ranges/distribution/FBA variant cannot be certified |
| Reservoir inputs are measured glucose/O2 | Active configuration loads fitted CSV and merges its uptake columns | Fitted-input benchmark differs from a raw-measurement deployment scenario; high confidence |
| Table 4 uses three learned extra constraints | Paper lists CO2/ethanol/acetate; export names two despite five latent outputs; no solve follows | Released notebook is incomplete/inconsistent; exact downstream constraints unverified |
| Reported R2 follows regression definition | Both metric implementations return `linregress(...).r**2`; copied Table 2 baseline reproduces only with that metric | Paper/code mismatch established; Table 4 metric lineage strongly supported but exact rows unverified |
| Original-model bounds define neural feasibility | Neural builder exports only S/Pin/Pref; CO2 `Vin=inf`; XML bounds not imported as general penalties | Do not call learned outputs bound-feasible or assume fixed uptake equalities |
| Supplied uptake conditions are equivalent across baselines | Goncalves fixes raw glucose/O2 exactly; reservoir loader supplies fitted values as NN features | These are different inputs and different enforcement mechanisms |
| `ref_47_fluxes` selects the full original targets | Non-fit loader instead selects iNF517-filtered 36-column file | Current balanced release does not instantiate the straightforward 45-target paper benchmark |
| Hyperparameters use random search / best epoch | Code uses TPE and trains 100 epochs before testing, with no best restoration | Effective released settings differ from supplement/comment descriptions |
| All Faure/MINN reservoirs use the same nutrient sampling and pFBA convention | AMN sampling and pFBA verified; MINN pretraining generator absent; GEM/input support differ | Preserve author-specific provenance and label A/B/C generation as local adaptations |

The accompanying updates to `MINN_training_notes.md`, `AMN_MINN_shared_reservoir_notes.md`, `iML1515_sampling_study_notes.md`, and `AMN_experiment_notes.md` record these findings and link here. They preserve the local predictive-regression-R2 contract and historical Pearson labels. No generator, evaluation, loss, seed, or checkpoint behavior was changed.

### Limitations and unresolved questions

- Obtain the actual 2,000-row pretraining dataset/generator to establish its raw/fitted range source, random distribution, channel order, equalities/caps, objective, solver variant, feasibility policy, and pretraining split/settings.
- Obtain the final Table 4 pFBA code and matched baseline/enhanced predictions, including targets, selected flux order, KO handling, and bounds. Until then, exact row-level metric certification and publication reproduction are unavailable.
- Obtain the historical fitting/reduction code to determine exact constraints/weights; a supplied fitted CSV and unchanged biomass are insufficient to reconstruct them.
- Resolve the original `gapC` versus `b0118/acnB` condition mapping before claiming a biologically matched knockout baseline; distinguish alias differences from actual gene substitutions.
- Establish fold/condition provenance of the saved balanced-MINN predictions. No saved notebook outputs document a successful run of the current released reservoir export.
- Physical nutrient concentrations, exact K-12 substrain, and raw replicate processing are not established by the inspected MINN release and paper's glucose-minimal description. The full medium above is a **model-availability** audit, not an experimentally verified recipe.

Verification performed: source/configuration call tracing; visual PDF checks; static tensor-shape inspection; complete exchange/direction/bound enumeration; raw/split/fitted counts and value checks; byte/hash comparisons; both metric recalculations with identical samples/fluxes; author-function checks; and source-repository cleanliness checks. No missing publication score or solver setting is inferred solely from a similarly named artifact.
