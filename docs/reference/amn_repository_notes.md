# Faure AMN E. coli experiment: repository inspection

Inspected 6 October 2026. The authors' experiment comprises **110 unique DH5-alpha media**, each represented by one mean maximum specific growth rate. There are **10 variable carbon sources plus 28 fixed model inputs**, including five fixed organic inputs. Direct AMN predictions use repeated condition-level cross-validation; the main reservoir-to-FBA result is a fit to the whole experimental dataset. The release contains material differences between publication descriptions, executable notebook settings, checkpoints, and source-data artifacts. These distinctions matter for reproduction.

## Scope, provenance, and citation conventions

- Read-only source checkout: `../amn_release`, commit **`10db2a62e8ea17b303ce21a9a1261b7328f861e9`**, dated 14 November 2023, merge commit on the authors' `brsynth/amn_release` repository.
- Publication: Faure, Mollet, Liebermeister, and Faulon (2023), *A neural-mechanistic hybrid approach improving the predictive power of genome-scale metabolic models*, Nature Communications 14, 4669. [Publisher record](https://www.nature.com/articles/s41467-023-40380-0); inspected local paper `docs/Faure etal 2023.pdf` and supplement `docs/Faure_supplementary.pdf`. Page numbers below are PDF page numbers, which agree with printed pagination.
- `AMN:` paths below are relative to `../amn_release`. Notebook cell numbers are **one-based positions counting both markdown and code cells**, not execution counts. RC cell 9 exists at the same position in HEAD and the working copy; local inserted cells shift later RC positions.
- Applicable `metabolic-NN/AGENTS.md` and AMN `README.md` were read. No sibling AMN `AGENTS.md` was found. Research Base supplied general context; all specific findings below were checked against the authors' files. No FluxTransformer settings were substituted.
- Local AMN modifications: `Build_Dataset.ipynb` has changed execution/output and kernel metadata, with **identical cell source** to HEAD. `Build_Model_RC.ipynb` has changed outputs/metadata, an interrupted E. coli reservoir execution, and two added inspection cells (working-copy cells 10–11) that count unique compositions. Its E. coli training cell source is unchanged. Untracked additions are `Faure etal 2023.pdf`, `Faure_supplementary.pdf`, and `generate_ecoli_iML1515_experimental_data.py`. These additions are not evidence of the published workflow. The library files, experimental CSV, both relevant NPZs, QP checkpoint, and figure workbooks inspected here match committed HEAD byte-for-byte.
- The original audit wrote only this findings file. A follow-up on 6 October 2026 recorded the sampling and solver distinctions in the AMN, shared AMN/MINN, and iML1515 sampling-study experiment notes. No training, FBA regeneration, or dataset regeneration was performed. Raw growth processing was recomputed in memory to validate saved values.

## 1. Medium sources and uptake bounds

### Units, reaction mapping, and the effective bound regimes

Unsplit BiGG exchange fluxes conventionally encode uptake as negative flux. AMN splits these exchanges into positive inward `_i` and outward `_o` reactions. For the exchanges listed below, signed unsplit flux is `v(EX_x_e) = v(EX_x_e_o) - v(EX_x_e_i)`; an inward cap `b` corresponds to a signed unsplit lower bound `-b`. Internal reversible reactions use `_for` and `_rev`. These suffixes must not be confused with FluxTransformer token conventions. The implementation determines exchange orientation from stoichiometry, negates the reverse reaction's coefficients, and sets nonnegative bounds. [AMN: `Library/Duplicate_Model.py`, `duplicate_model`, lines 164–265; paper p. 9, “Making metabolic networks suitable for neural computations”.]

Physical concentrations are in g/L, mg/L, or mol/L. COBRA uptake bounds are in **mmol gDW^-1 h^-1**; biomass flux is in h^-1. Binary composition indicators have no physical units. There is **no measured concentration-to-flux conversion** in these files. The same source has different numerical roles in different execution paths:

| Execution path | Fixed inputs | Present variable source | Absent variable source |
|---|---|---|---|
| Direct experimental AMN-Wt/LP/QP | Binary `1`; treated numerically as an inward nominal cap of 1 | Binary `1`, nominal cap 1 | `0`, nominal cap 0 |
| Simulated UB training data | Inward cap **2.2** for each of 28 fixed inputs | `b(k,j) = (k+1) × j × 2.2/99`, `j` an integer 1–99, `k` the number of variable sources | 0 |
| Experimental AMN-Reservoir | Fixed inward caps **2.2**; divided by reservoir scaler **11** before the frozen network | Learned nonnegative, condition-dependent inward cap; no fixed upper clamp of 2.2 | Presence mask gives a learned cap of exactly 0 |
| Out-of-the-box pFBA comparison | **2.5** for every fixed component | **2.5** | **0.0001**, from an explicit epsilon |
| pFBA given reservoir-exported caps | Fixed 2.2; variable values from exported solution | Learned/exported cap, with baseline scaler set to 1 | Adds 0.0001 whenever the supplied cap is below 0.0001 |

The direct AMNs use soft constraint penalties after their updates; nominal caps do not certify final feasibility. QP/LP initialization clips predicted inward fluxes to the input cap, but UB updates can change them. Wt has no hard final cap projection. The RC presence mask constrains its *input caps*; downstream pFBA explicitly permits epsilon uptake even when a cap is zero. [AMN: `Dataset_input/iML1515_EXP.csv`; `Library/Build_Model.py`, `input_AMN` lines 288–342, `get_V0` lines 403–443, `input_RC`/`RC` lines 851–957; `Build_Dataset.ipynb` cell 20.]

### Complete variable-source list

All ten are binary experimental variables and are individually added at **0.4 g/L when present**, according to the paper; mixture concentration therefore increases with source count. Their numerical caps follow the common regimes above, independently of compound molecular mass. “Lactate” in the paper maps specifically to the **D-lactate exchange** in the code; the publication's reagent wording alone does not establish the supplied reagent's stereochemical composition.

| Variable source | Exact AMN inward exchange | Corresponding unsplit exchange |
|---|---|---|
| D-ribose | `EX_rib__D_e_i` | `EX_rib__D_e` |
| Maltose | `EX_malt_e_i` | `EX_malt_e` |
| Melibiose | `EX_melib_e_i` | `EX_melib_e` |
| Trehalose | `EX_tre_e_i` | `EX_tre_e` |
| D-fructose | `EX_fru_e_i` | `EX_fru_e` |
| D-galactose | `EX_gal_e_i` | `EX_gal_e` |
| Acetate | `EX_ac_e_i` | `EX_ac_e` |
| D-lactate | `EX_lac__D_e_i` | `EX_lac__D_e` |
| Succinate | `EX_succ_e_i` | `EX_succ_e` |
| Pyruvate | `EX_pyr_e_i` | `EX_pyr_e` |

Evidence: AMN `Build_Experimental.ipynb` cell 11; `Dataset_experimental/EXP110.csv` first ten data columns; `Dataset_input/iML1515.csv` last ten medium columns. Physical concentrations: paper pp. 11–12, “Generation of an experimental training set” and “Culture conditions”; supplement p. 25, Fig. S11.

### Complete fixed organic-input list

Each is `1` in **all 110 model-input rows** and is a fixed 2.2-cap input in simulated training/RC. The direct nominal cap is 1 and the common-scaler pFBA cap is 2.5, as above.

| Fixed model organic input | Exact AMN inward exchange | Unsplit exchange | Documented physical addition |
|---|---|---|---|
| Glycerol | `EX_glyc_e_i` | `EX_glyc_e` | **No growth-medium concentration documented.** Glycerol appears in model medium and frozen-stock handling; stock glycerol is not evidence of a quantified culture-medium supplement. |
| L-alanine | `EX_ala__L_e_i` | `EX_ala__L_e` | 5 mg/L |
| L-proline | `EX_pro__L_e_i` | `EX_pro__L_e` | 5 mg/L |
| L-threonine | `EX_thr__L_e_i` | `EX_thr__L_e` | 5 mg/L |
| Glycine | `EX_gly_e_i` | `EX_gly_e` | 5 mg/L |

The paper also says “0.04 g/L amino acid mix”, while four constituents at 5 mg/L sum to **0.020 g/L**. The component concentrations and stated mix total are internally inconsistent; the available evidence does not resolve this. [Paper p. 12, “Culture conditions”; AMN `Dataset_input/iML1515.csv`, `iML1515_EXP.csv`.]

These inputs supply substantial model carbon even in a “single-source” condition. Using carbon atom counts 3, 3, 5, 4, and 2 for glycerol, alanine, proline, threonine, and glycine, respectively, the fixed 2.2 caps allow up to `2.2 × (3+3+5+4+2) = 37.4 mmol carbon gDW^-1 h^-1` of **potential organic carbon uptake**. The corresponding nominal totals are 17 for direct binary AMNs and 42.5 for common-scaler pFBA. These are summed capacity ceilings, not observed consumption or guaranteed jointly achieved fluxes. The four amino acids alone contribute 30.8 at cap 2.2. Physically, the four stated 5 mg/L additions contain approximately **0.687 mmol carbon/L** in total, calculated from their standard molecular masses; a model cap cannot be equated with that finite concentration. Physical glycerol supply remains undocumented. “Sole variable source” below always leaves the fixed inputs separate.

### Complete fixed base-medium and other exchange-input list

All **23** rows below are binary `1` in all 110 input media; their per-exchange caps are 1 for direct experimental AMNs, 2.2 for simulation/RC, and 2.5 for common-scaler pFBA. Numerical oxygen availability follows the same rule as the other fixed inputs. The physical column gives the paper's reagent preparations, not inferred ion activities or measured uptake rates.

| Component | Exact AMN inward exchange | Unsplit exchange | Physical recipe / documentation |
|---|---|---|---|
| Phosphate | `EX_pi_e_i` | `EX_pi_e` | KH2PO4 3 g/L and Na2HPO4·2H2O 8.5 g/L |
| Carbon dioxide | `EX_co2_e_i` | `EX_co2_e` | No quantified concentration or addition |
| Ferric iron | `EX_fe3_e_i` | `EX_fe3_e` | No separate ferric addition specified |
| Proton | `EX_h_e_i` | `EX_h_e` | Final culture-medium pH 7.4; no uptake measurement |
| Manganese(II) | `EX_mn2_e_i` | `EX_mn2_e` | MnCl2·4H2O 1 mg/L |
| Ferrous iron | `EX_fe2_e_i` | `EX_fe2_e` | FeSO4·7H2O 3 mg/L |
| Zinc(II) | `EX_zn2_e_i` | `EX_zn2_e` | ZnSO4·7H2O 4.5 mg/L |
| Magnesium(II) | `EX_mg2_e_i` | `EX_mg2_e` | MgSO4 2 mM |
| Calcium(II) | `EX_ca2_e_i` | `EX_ca2_e` | CaCl2 100 µM |
| Nickel(II) | `EX_ni2_e_i` | `EX_ni2_e` | No quantified addition specified |
| Copper(II) | `EX_cu2_e_i` | `EX_cu2_e` | CuSO4·5H2O 0.3 mg/L |
| Selenate | `EX_sel_e_i` | `EX_sel_e` | No quantified addition specified |
| Cobalt(II) | `EX_cobalt2_e_i` | `EX_cobalt2_e` | CoCl2·6H2O 0.3 mg/L |
| Water | `EX_h2o_e_i` | `EX_h2o_e` | Aqueous solvent |
| Molybdate | `EX_mobd_e_i` | `EX_mobd_e` | Na2MoO4·2H2O 0.4 mg/L |
| Sulfate | `EX_so4_e_i` | `EX_so4_e` | MgSO4 and the trace-element sulfate salts above |
| Ammonium | `EX_nh4_e_i` | `EX_nh4_e` | NH4Cl 1 g/L |
| Potassium | `EX_k_e_i` | `EX_k_e` | KH2PO4 3 g/L |
| Sodium | `EX_na1_e_i` | `EX_na1_e` | NaCl 0.5 g/L, phosphate, molybdate, and EDTA salts |
| Chloride | `EX_cl_e_i` | `EX_cl_e` | NaCl 0.5 g/L, NH4Cl 1 g/L, CaCl2, and trace chlorides |
| Oxygen | `EX_o2_e_i` | `EX_o2_e` | Aerobic shaking; no dissolved-O2 measurement or concentration supplied |
| Tungstate | `EX_tungs_e_i` | `EX_tungs_e` | No quantified addition specified |
| Selenite | `EX_slnt_e_i` | `EX_slnt_e` | No quantified addition specified |

Evidence: AMN `Dataset_input/iML1515.csv` first 23 medium columns and `iML1515_EXP.csv`; physical recipe from paper p. 12, “Culture conditions”. Other documented **physical** additions are thiamine-HCl 1 mg/L, Na2EDTA·2H2O 15 mg/L, and H3BO3 1 mg/L. They have **no dedicated input among these 38 exchanges**; the workflow does not explicitly model their supplied concentrations. Thiamine's full-model inward exchange is `EX_thm_e_i` (unsplit `EX_thm_e`), but it is excluded from the input list and reduced models. No dedicated EDTA or boric-acid input ID is supplied by this workflow. The trace-element stock is adjusted to pH 4 and stored at 4 °C; the final medium is adjusted to pH 7.4 and filtered through 0.22 µm.

### Exclusions, implicit exchanges, and execution trace

- **D-glucose** (`EX_glc__D_e_i`, unsplit `EX_glc__D_e`) is not an experimental variable or fixed input. It is absent from both reduced models. In the full duplicated input XML it is initially open at 10, but `run_cobra` zeros the current medium before applying `IN`; generation therefore closes it. Do not inherit the original iML1515 glucose medium.
- Every unselected member of the ten-source list is excluded in simulation; direct experimental nominal caps are zero. The pFBA experimental comparison is the explicit epsilon exception in the bound table.
- Other carbon sources, including L-lactate (`EX_lac__L_e_i`), sucrose (`EX_sucr_e_i`), and lactose (`EX_lcts_e_i`), are not supplied as inputs; they are distinct from D-lactate and the listed disaccharides. The reduced XMLs retain **exactly 38 inward exchange reactions**, all represented in the input header. There are no additional unlisted inward exchanges in those reduced models.
- The full duplicated model has 331 positive-upper-bound inward `EX_…_i` exchanges; many have the tiny placeholder cap `1e-300`, and D-glucose is the only unlisted one with a materially positive initial cap. `run_cobra` resets all existing medium entries, including those placeholders, before setting the 38 specified inputs. An input is applied only if its reaction is already a key in `model.medium`; the tiny positive bounds keep otherwise closed listed organic reactions addressable. Secretion reactions remain governed by the XML; they are not input nutrients. [AMN: `Library/Build_Dataset.py`, `run_cobra`, lines 427–434; `Dataset_input/iML1515_duplicated.xml`; `Dataset_model/iML1515_UB.xml` and `iML1515_EXP_UB.xml`.]

The observed experimental file has 10 source indicators, eight replicate growth columns, and mean/SD. `Dataset_input/iML1515_EXP.csv` contains the same source indicators in the same row order, prepended by 28 all-one columns, and only `GR_AVG` as the target. Target rounding changes values by at most `4.997e-10 h^-1`. The inspected experimental-building notebook writes `EXP110.csv`; an explicit script that constructs this exact 38-column handoff CSV from it was **not found in the inspected committed path**. Content alignment is verified independently.

`Build_Dataset.ipynb` cell 17 calls `TrainingSet(method='EXP', mediumsize=38, mediumbound='UB')` on the reduced `Dataset_input/iML1515_EXP.xml`. `TrainingSet.__init__` directly slices the CSV into X and Y; it does not convert g/L concentrations, simulate growth, or use replicate SD. It saves `Dataset_model/iML1515_EXP_UB.npz`. Direct `input_AMN` applies a global maximum scaler before splitting; here the maximum is 1, so inputs are unchanged. The target is not normalized. [AMN: `Library/Build_Dataset.py` lines 606–666, 680–723; `Library/Build_Model.py` lines 288–342.]

Simulation is a separate path. `Build_Dataset.ipynb` cell 13 takes each experimental **presence pattern**, requests 100 pFBA samples per pattern, and reduces the full split model. It does not use measured growth rates as simulated targets. `iML1515.csv` sets 28 fixed levels to 1, ten variable levels to 100, every nominal maximum to 2.2, and `ratio_drawing=0`. Supplying an explicit nonempty `varmed` list overrides random source selection: all k sources in that condition are used. `create_random_medium_cobra` then multiplies each variable sampled cap by **k+1**. The saved dataset verifies positive cap ranges 0.044444–4.4 (k=1), 0.066667–6.6 (k=2), 0.088889–8.8 (k=3), and 0.111111–11 (k=4); the sampling is a 99-level discrete grid, not a continuous uniform draw. Fixed caps remain 2.2. [AMN: `Library/Build_Dataset.py` lines 482–568, especially 543; saved `Dataset_model/iML1515_UB.npz`, arrays `X`, `levmed`, `valmed`.]

For downstream FBA, cell 20 uses `run_cobra(method='pFBA')`, maximizing `BIOMASS_Ec_iML1515_core_75p37M` and then minimizing total flux through COBRA's pFBA. GLPK configuration is timeout 5 seconds, presolve `auto`, simplex; flux magnitudes below `1e-8` are zeroed in returned flux dictionaries. A default second-objective fraction of 0.75 is irrelevant here because there is only one objective. The reduced XML retains ATP maintenance lower bound **6.86** and upper bound 1000. These settings are separate from the unrolled neural LP/QP layers. [AMN: `Library/Build_Dataset.py` lines 408–476; reduced XMLs.]

## 2. Experimental data and condition counts

### Biological experiment and measurement processing

The strain is **E. coli DH5-alpha**, not unmodified K-12 MG1655. iML1515 is the mechanistic reconstruction used to model these measurements; it does not establish that the assayed strain is MG1655. The target is the **maximum specific growth rate in h^-1**, not final OD, growth yield, or an intracellular-flux vector. Cultures proceed from −80 °C glycerol stocks to LB for 7 hours, then 5 µL inoculum into 200 µL supplemented M9 for 14 hours overnight, then a 5 µL transfer to a replicate plate for growth monitoring. The assay uses 96-well U-bottom plates, 37 °C, maximum continuous orbital shaking, and OD600 measured every 10 minutes over 24 hours. [Paper p. 12, “Culture conditions”; supplement p. 25, Fig. S11.]

The committed processing assigns ten conditions per plate to columns 2–11, with rows A–H as eight technical replicates. Eleven plates provide 880 intended condition-replicate measurements; peripheral unused columns are not additional training conditions. `dict_all` selects slices of shuffled design tables for the dates in the plate table below. Design tables contain all binary combinations of a specified cardinality; the first 10/20/40/40 selected conditions are tested, using seed 3 for design shuffling. [AMN: `Build_Experimental.ipynb` cells 11, 13, 15, 34, 36–37; `Library/Build_Experimental.py` lines 47–96, 141–157.]

Actual growth processing is:

1. Read raw `…_data.csv` and per-well `…_start_stop.csv`; exclude hard-coded per-plate wells from `outliers_dic`.
2. Smooth OD four times. Each interior smoothing point uses `y[i-2:i+2]`, a **four-point** slice, sets its minimum and maximum to NaN, and averages the remainder; boundary points are retained. Blank subtraction exists only as a commented line.
3. Convert the specified start/stop times to indices using `int(time × 6)`. Missing times default to the start/end of the record, limiting the maximal-rate search to the curated growth phase where specified.
4. Fit a line to **natural log OD versus time** in each six-sample sliding window and retain the maximum slope. Six samples at ten-minute spacing span 50 minutes between the first and last observation, although the paper calls this a one-hour window. The loop `range(len(xdata)-6)` also omits the final possible window.
5. Compute each condition's mean and **sample SD (`ddof=1`)** across retained replicates, skipping missing values. Concatenate the eleven per-plate result tables; neither replicates nor SD become independent training rows or training-loss weights.

Evidence: AMN `Library/Build_Experimental.py` lines 160–227; `Build_Experimental.ipynb` cells 30, 32, 34, 36–37; paper p. 12, “Growth rates determination”. Recomputing these functions in memory for **all eleven plates** reproduced each saved nonmissing replicate value to within `8.1e-15 h^-1` and reproduced every missing-value mask. Their concatenation equals `EXP110.csv` exactly. This validates the released processing, including its outlier-list typo discussed in section 4.

### Verified counts

`EXP110.csv` contains **110 rows, 110 distinct 10-bit source patterns, and no duplicated media**. The model-input CSV has shape `(110,39)`; its first 38 columns are features and its final column is `GR_AVG`. The experimental NPZ has `X=(110,38)`, `Y=(110,1)`, both float64 on disk, and matches this CSV. Growth targets range from **0.070366424 to 0.420538476 h^-1**. All 110 means and SDs match recalculation from replicate columns within floating-point precision.

| Number of variable sources | Distinct conditions / model rows | Total variable-source memberships |
|---:|---:|---:|
| 1 | 10 | 10 |
| 2 | 20 | 40 |
| 3 | 40 | 120 |
| 4 | 40 | 160 |
| **Total** | **110** | **330** |

There is no zero-variable-source condition. All five fixed organic **model inputs** cover 110/110 conditions; the four amino acids also have publication evidence of physical supplementation. Physical glycerol supplementation is not verified. Counting the five fixed organic model inputs would produce 6–9 organic inputs per condition, but that is not the experimental variable-source mixture size.

| Variable source | Conditions containing it | Sole **variable** source | In a variable-source mixture |
|---|---:|---:|---:|
| D-ribose | 34 | 1 | 33 |
| Maltose | 28 | 1 | 27 |
| Melibiose | 35 | 1 | 34 |
| Trehalose | 35 | 1 | 34 |
| D-fructose | 34 | 1 | 33 |
| D-galactose | 35 | 1 | 34 |
| Acetate | 31 | 1 | 30 |
| D-lactate | 31 | 1 | 30 |
| Succinate | 31 | 1 | 30 |
| Pyruvate | 36 | 1 | 35 |
| **Sum of overlapping memberships** | **330** | **10** | **320** |

Counts use `k = sum(first ten indicators)` and, for source j, `sum(x_j>0)`, `sum((x_j>0)&(k==1))`, and `sum((x_j>0)&(k>1))`. A mixture contributes to every source it contains, so 330 memberships are not 330 independent conditions. The 10/20/40/40 cardinality counts agree with the paper's design. [AMN: `Dataset_experimental/EXP110.csv`; paper pp. 11–12.]

| Retained technical replicates per condition | Number of conditions |
|---:|---:|
| 2 | 7 |
| 3 | 16 |
| 4 | 26 |
| 5 | 14 |
| 6 | 20 |
| 7 | 15 |
| 8 | 12 |

There are **557 retained replicates**, 323 excluded slots, mean **5.063636** per condition, population SD **1.759531**, sample SD **1.767584**. The paper's Methods report 2–8 retained replicates averaging 4.6 ± 1.6; Fig. 3's caption says three measured replicates. Neither summary matches the released file.

| Plate date / filename prefix | Retained replicates |
|---|---:|
| 20220504 | 48 |
| 20220429 | 36 |
| 20220506 | 49 |
| 20220507 | 51 |
| 20220512 | 53 |
| 20220513 | 54 |
| 20220514 | 65 |
| 20220823 | 53 |
| 20220824 | 50 |
| 20220825 | 48 |
| 20220826 | 50 |

### Separate simulated and external datasets

`Dataset_model/iML1515_UB.npz` is a **simulated pretraining/benchmark dataset**, with 11,000 pFBA rows and 550 flux targets per row. It contains precisely the 110 experimental source patterns, with 100 simulated cap draws per pattern. The cardinality groups have 1,000/2,000/4,000/4,000 rows. Its `S` is `(1083,550)`. “550,000 labels” in comparisons means 1,000 rows × 550 flux targets, not 550,000 distinct media. The experimental NPZ instead uses `S=(1080,543)` and measured growth targets; it is not a concatenation with the simulated labels.

The other E. coli datasets are **separate supervised tasks**:

- `Dataset_input/biolog_iML1515_EXP.csv`: `(17400,431)` table for 145 metabolic-gene knockout mutants × 120 media, from ASAP/Biolog K-12 measurements. Paper Fig. 4 trains an AMN with medium and reaction-KO features on this dataset; it is not pretraining for the 110-media DH5-alpha experiment or an external held-out test of that experiment. Filtering removes unsupported substrates/genes and duplicates. The growth/no-growth threshold is 0.165 h^-1, and the baseline's optimum scalar is 11, both specific to that dataset. [Paper pp. 6–8, 12; AMN `Build_Model_AMN_KO.ipynb`, `Library/Build_Model_KO.py`; `Figures.ipynb` cells 19–25 identify its saved prediction artifacts.]
- `Dataset_input/rijs_iML1515_EXP2.csv`: `(128,120)` table for 64 regulator-gene knockout mutants × two media, 31 measured fluxes as targets after 89 input columns. It is a separate multi-flux AMN benchmark in Fig. S8, not extra training data for EXP110. [Paper p. 8; supplement pp. 21–22; AMN `Figures.ipynb` cell 42.]
- P. putida/iJN1463 assays belong to a different organism/task and do not augment this E. coli dataset.

## 3. Exact training and evaluation procedure

### Solver scope across the inspected author workflows

**The release does not use pFBA in every experiment.** Its E. coli core simulation uses pFBA (`Build_Dataset.ipynb` cell 11), as does the 110-pattern iML1515 simulated reservoir (cell 13). The saved core and iML1515 simulation NPZs also declare `method='pFBA'`. The actual E. coli comparator call in cell 20 uses pFBA even though that cell loads experimental inputs with `method='EXP'`. The Biolog knockout comparator likewise calls pFBA (`Build_Dataset_KO.ipynb` cell 14).

In contrast, **P. putida/iJN1463 uses plain FBA**, both for its simulated dataset (`Build_Dataset.ipynb` cell 14; `Dataset_model/IJN1463_10_UB.npz`, 4,860 rows, `method='FBA'`) and its mechanistic comparator (cell 21). `Library/Build_Dataset.py` lines 441--442 dispatch to `cobra.flux_analysis.pfba(model)` only when requested, otherwise to `model.optimize()`. The pFBA call supplies no `fraction_of_optimum` override (COBRA default 1.0); the helper's default `method='FBA'` is not evidence that every caller uses FBA. Direct AMN-Wt/LP/QP predictions use their learned mechanistic layers, and experimental targets are measured labels; neither should be described as a dataset of pFBA-generated growth targets. These statements concern the inspected release paths rather than asserting the solver used in every historical run behind the publication.

### Direct experimental AMNs: effective settings

`Build_Model_AMN.ipynb` cells **24 (QP), 26 (LP), and 28 (Wt)** each instantiate a new `Neural_Model` from `Dataset_model/iML1515_EXP_UB`. They **do not load simulated pretrained weights**. Experimental growth supervision and simulated reservoir pretraining are distinct workflows.

Shared effective settings: 38 binary medium inputs; target raw mean growth rate; three outer runs with NumPy seeds **1, 2, 3**; **1000 epochs**, batch size **5**, **10 shuffled folds**, `niter=0`, `scaler=True` (effective scaler 1), no early stopping. Adam is supplied as the string `'adam'`, so its effective learning rate is **0.001**, irrespective of a `train_rate` attribute. No TensorFlow/Keras initialization seed is set. `learn_rate` is the mechanistic update step, not the Adam rate. [AMN: notebook cells above; `Library/Build_Model.py` lines 469–490, 618–638, 732–750, 1244–1290.]

| Variant | Actual architecture and trainable components | Mechanistic settings |
|---|---|---|
| AMN-QP | Dense 38 → 500 ReLU, dropout 0.25, Dense 500 → **543** ReLU; both Dense layers trainable. Uptake components of V0 clipped to nominal input caps. Stoichiometric/projection matrices frozen. | **4** unrolled constraint updates; step **0.001**, momentum/decay **0.9**. No experimental target supplied to the inner solver. |
| AMN-LP | Independent Dense networks initialize V0 (38 → 500 → **543**, final ReLU) and M0 (38 → 500 → **2703**, final **linear**), each with dropout 0.25. Both networks trainable; mechanistic matrices frozen. | **4** primal/dual updates; step **0.001**; objective vector multiplied by **100**. M0 is learned, not initialized to zero. |
| AMN-Wt | Custom RNN cell with factorized learned input mapping 38 → 500 → 38, learned biases, and a **543 × 1080** recurrent weight array multiplied by the fixed M2V mask. V2M and normalized M2V specify biological connectivity. | Same input repeated for **4** recurrent steps. The actual recurrence contains **no ReLU activation and no dropout operation**; `dropout=0.25` is stored but unused in this cell. Both input and recurrent weights are learned. |

Dense weights use random-normal initialization and zero biases; Wt uses `add_weight` kernels and random-normal biases. No L1/L2 weight regularizer, weight decay, uncertainty weighting, or target standardization is specified. [AMN: `Library/Build_Model.py`, `Dense_layers` lines 240–263; `get_V0`/`QP_layers` lines 403–467; `get_M0`/`LP_layers` lines 532–616; `RNNCell` lines 651–730.]

### Actual optimized loss and inner objectives

For direct AMNs, `input_AMN` extends the target from `[growth]` to `[growth,0,0,0]`. The corresponding output prefix is `[predicted_growth, a_S, a_in, a_pos]`, followed by diagnostic V and, for LP/QP, V0. `my_mse` optimizes **only the first four columns**; diagnostic flux columns are not separately supervised. Per example the effective loss is

```text
a_S   = ||S V||_2 / m
a_in  = ||ReLU(Pin V - x)||_2 / nin       (UB mode)
a_pos = ||ReLU(-V)||_2 / n

L_direct = ( (predicted_growth - measured_growth)^2
             + ||S V||_2^2 / m^2
             + ||ReLU(Pin V - x)||_2^2 / nin^2
             + ||ReLU(-V)||_2^2 / n^2 ) / 4
```

The training batch aggregates the per-example loss in Keras. For current experimental files, `m=1080`, `n=543`, `nin=38`. The paper's Eq. (2) uses **1/m, 1/nin, 1/n**, rather than the implemented squared denominators, and writes an unaveraged sum. The change in *relative* mechanistic weighting is substantive; a global division by four alone would not explain it. The code's `my_mae` function also returns MSE despite its name. [AMN: `Library/Build_Model.py` lines 68–84, 128–226, 344–366; paper pp. 10–11; supplement pp. 13–15.]

QP training invokes the inner solver **without target values**, so its unrolled updates reduce constraint residuals; growth-target error is propagated through the outer loss. Let column-vector fluxes be V, `r=ReLU(Pin V-x)`, and `p=ReLU(-V)`. The actual gradient-like update direction is

```text
g(V) = S.T(SV)/(2 m^2) + Pin.T r/nin^2 - p/n^2
d_t = 0.9 d_(t-1) - learn_rate × g(V_t),   d_0 = 0
V_(t+1) = V_t + d_t
```

UB uses an all-one update mask. This is the executed normalization, including the extra stoichiometric factor 1/2; it is not an exact gradient of the paper's stated Eq. (2), or a converged quadratic-program solution. When targets are provided to standalone MM-QP, `Loss_all` also supplies the target-residual direction. That MM path should not be substituted for AMN-QP training. [AMN: `Loss_SV`, `Loss_Vin`, `Loss_Vpos`, `Gradient_Descent`, `get_V0`, lines 144–189, 368–443.]

LP uses the negative biomass objective vector `c` constructed from the metabolic objective, multiplied by 100 inside `LP`. Its state updates are `V += learn_rate × dV` and `M += learn_rate × dM`; four iterations refine learned initialization. `b_ext` encodes medium availability, positivity, and ATPM lower bound, with size `2m+n=2703` for the current experimental model; `b_int` is zero with length 478. `input_AMN` appends these bounds to the 38 medium features, yielding **3219 actual LP input columns**. Dense initializers consume the first 38. The LP's biomass maximization principle, the QP's residual minimization, and the **outer measured-growth loss** are different objectives. [AMN: `Library/Build_Dataset.py` lines 239–377; `Library/Build_Model.py` lines 309–327, 500–616.]

Reported constraint statistics use `Loss_constraint = (a_S^2+a_in^2+a_pos^2)/3`, averaged over samples. They are **not** the optimized four-term training loss. `evaluate_model` computes growth R2 separately with sklearn; Keras `my_r2` during fit operates on all four supervised columns, including the three zero columns, and is not the reported growth-only Q2. [AMN: `Library/Build_Model.py` lines 78–84, 192–226, 1006–1066.]

### Verified cross-validation, initialization, and possible leakage

`train_evaluate_model` uses **`KFold(n_splits=10, shuffle=True)`**, with **no stratification and no explicit `random_state`**. NumPy's current RNG state controls the split. Each outer run produces ten folds of **99 training conditions and 11 held-out conditions**. The held-out fold is also passed to Keras as `validation_data`: there is no third validation set and no independent experimental test set. Each retained condition mean appears once in held-out predictions per run. Thus each direct variant trains 30 models across the three outer runs. [AMN: `Library/Build_Model.py` lines 1102–1240.]

Weights are built afresh for each fold: `model_type` reconstructs Dense/LP/QP networks, and Wt explicitly reconstructs a `Neural_Model` and RNN cell. No trained fold weights are passed into the next fold. RC rebuilds its prior for each fold but copies the same pretrained frozen reservoir weights, intentionally. NumPy seeds alone do not guarantee repeatable neural initialization: the code does not set a TensorFlow seed or deterministic-execution controls.

The split unit is the **condition mean**, not a raw replicate. There are no duplicate ten-source compositions in the 110 rows, and no evidence of the same condition's technical replicates being split across folds. Replicate averaging and manual exclusions occur before CV. A plate can contribute distinct media to both train and held-out folds; this is not a plate-grouped assessment.

Preprocessing calls `MaxScaler` on the full feature array **before splitting**. In EXP110 it only recovers the already known constant maximum 1, so this does not expose held-out growth labels. Simulated pretraining and model reduction use source patterns from **all 110 experimental media**; a hypothetical RC CV therefore uses knowledge of held-out compositions, but not their measured growth labels, in the fixed simulation prior. This is composition-informed pretraining, not evidence of replicate-label leakage. Hyperparameter justification comes from simulated E. coli core data; no nested experimental CV/tuning protocol is present. The common-scaler FBA optimum is selected on all experimental target values, so its score is a tuned fit score.

Early stopping defaults to **False** in all relevant direct/RC cells. If explicitly enabled, the callback has patience 10, monitors held-out `val_loss`, and **does not set `restore_best_weights=True`**. The nominal `epochs = 0.9 * model.epochs` local variable is unused: fit executes `epochs=model.epochs`. There is one fitting attempt (`Niter=1`), even when training R2 is poor. Keras fit uses its default training-data shuffling. [AMN: `Library/Build_Model.py` lines 1119–1164, 1244–1258, 1466–1484.]

The routine compares held-out-fold R2 to choose a “best” model for return and evaluates it on all rows. With default `niter=0`, this whole-data prediction **does not replace** the assembled held-out predictions. Moreover, QP/LP/ANN/RC set `model=parameter`, so `Netmax` aliases the same parameter object whose `.model` is overwritten in later folds; the returned “best” network may actually be the **last fold's network**. Wt reconstructs independent parameter objects and avoids that particular alias. This affects returned/saved-network provenance, not the already stored per-fold predictions. A selected fold network is not a refit on all conditions. [AMN: `train_model` lines 1123–1139; `train_evaluate_model` lines 1224–1239.]

### Simulated pretraining and the shipped reservoir

The simulation generator uses NumPy seed **10**, 100 draws for each of the 110 patterns, pFBA targets, and reduction of reactions with zero simulated activity while preserving medium reactions. This gives the committed 550-reaction/1083-metabolite reservoir model. [AMN: `Build_Dataset.ipynb` cell 13; `Library/Build_Dataset.py` `reduce_model`, `reduce_and_run`, lines 379–405, 668–678.]

The **current simulation-training cell** and the **checkpoint actually loaded by RC** differ:

| Setting | `Build_Model_AMN.ipynb` cell 14 | Shipped `Reservoir/iML1515_UB_AMN_QP_param.csv` / H5 |
|---|---|---|
| Input/output | 38 inputs; 550-flux latent state; biomass supervision | Same dimensions; total diagnostic output 1104 |
| Hidden layer | 1 × 500, ReLU, dropout 0.25 | Same, confirmed from H5 layer configuration |
| Input scaler | MaxScaler over training X | **11.0** |
| Mechanistic QP steps / step size | **4 / 0.01**, decay 0.9 | **0 / 0.01**, decay 0.9 |
| Training epochs | **25** | **500** in parameter CSV; epoch history not embedded |
| Adam rate / batch | Default **0.001 / 5** | H5 optimizer confirms **0.001 / 5** from parameter record |
| CV / additional test | **5-fold** on 90% of rows, separately held-out 10% | Parameter record **xfold=0**, `niter=0`; historical split not reconstructible |
| Early stopping | False | False |

The current cell first removes **1100** random rows for an independent simulation test and uses 9900 remaining rows in five folds: **7920 train / 1980 held-out** in each fold. Table S1 reports 100 epochs for the corresponding iML1515-QP benchmark, whereas the cell requests 25 and the shipped record says 500. These are three distinct evidence layers, not interchangeable defaults. The independent-test block of QP cell 14 also assigns `reservoir.model.b_ext = btest`, although `btest` is not defined in that cell; execution depends on prior notebook state or fails there. [Supplement pp. 18–20; AMN notebook cell 14.]

H5 inspection was read-only through h5py in the existing AMN environment. The file contains 28 graph layers, a 38→500→550 Dense network and diagnostic/projection operations, consistent with **zero unrolled QP updates**, and records **Keras 2.9.0**. Its serialized fit metric is `my_mae` rather than the current constructor's `my_r2`; `my_mae` in the current library is MSE. The environment file requests Python 3.7, TensorFlow/Keras 2.6, NumPy 1.20.3, COBRA, sklearn, pandas, and pyDOE2, with several dependencies unpinned. Thus README's exact-environment claim does not establish exact checkpoint-environment reproduction.

### Experimental reservoir training and downstream pFBA

Committed `Build_Model_RC.ipynb` **cell 9** loads that shipped QP checkpoint and reads `Dataset_input/iML1515_EXP.csv` via `read_XY(...,38)` without target scaling. Effective settings: NumPy seeds **0,1,2**, one trainable prior with **10 variable inputs → 500 ReLU → 10 ReLU outputs**, dropout 0.25, random-normal kernels/zero biases, no post-network, **1000 epochs**, batch size **5**, Adam rate **0.0001**, **`xfold=0`**, `niter=0`, no early stopping.

The first 28 inputs are replaced with reservoir `valmed=2.2`; the prior sees only the last ten binary variables. Its ten outputs are concatenated with the fixed inputs and multiplied by the 38-element presence mask. The combined caps are divided by **11**, passed through a reconstructed frozen `QP_layers` graph, and its Dense weights are copied from the H5. With the shipped parameter `timestep=0`, this graph performs no mechanistic refinement beyond initialization/clipping and diagnostic residuals. Dropout in the reconstructed frozen Dense path is still present; freezing weights does not itself remove dropout during outer training. [AMN: `Library/Build_Model.py` lines 851–957, 1466–1550.]

RC targets remain one column, unlike `input_AMN`. Consequently `my_mse` minimizes **only growth MSE**; residual output columns are diagnostic, with **no additive mechanistic penalty in the RC outer training loss**. Both the pretrained growth surrogate and the learned cap prior contribute to predictions, but only the prior has trainable weights in this E. coli configuration. [AMN: `my_mse` lines 68–71, `input_RC` lines 851–867, `RC` lines 935–957.]

For `xfold<2`, the same 110 rows are training, validation, and evaluation data. The notebook variable/file label `Q2` does not make this a held-out score. Paper p. 8 explicitly describes training the new layer on the whole dataset for Fig. 5, consistent with this code. It exports caps from **the first outer run**, not their three-run mean; multiplies internal caps back by scaler 11; appends measured Y; and writes the result for COBRA. Current output naming uses `…_RC_…`, whereas committed historical artifacts use `…_RC_AMN_…` and `_train`/`_pred` suffixes. No current dedicated E. coli RC cell sets `xfold=10` to regenerate the `_pred` artifact; its CV provenance is supported by the supplement and filename, not a logged execution of such a cell. [AMN: `Build_Model_RC.ipynb` cell 9; supplement p. 23, Fig. S9.]

`Dataset_input/iML1515_UB_AMN_QP_RC_AMN_solution_for_Cobra_train.csv` and `_pred.csv` are 110×39 headered handoff tables. Their caps match the corresponding raw solution exports rounded to **three significant digits**: maximum differences are **0.049482** and **0.049216**, respectively. They preserve zero caps for absent sources and fixed caps 2.2; present-source caps can also be zero. Maximum variable caps in these headered files are **17.9** (train) and **19.4** (pred), demonstrating that learned caps are not clamped to 2.2 or the simulation maximum 11. The full-precision raw exports and the rounded handoff files are distinct artifacts. `Build_Dataset.ipynb` cell 20 requires selecting the appropriate medium file and scaler 1 for these caps; its active default instead runs the common-scaler 2.5 baseline.

### ANN and other comparison procedures

The general ANN constructor uses Dense layers with dropout, regression MSE, Adam default 0.001, and no mechanistic-loss term. `Build_Model_ANN_Dense.ipynb` cell 7 currently trains **simulation flux targets**, not EXP110 growth rates: one 500-unit hidden layer, ReLU outputs for 550 fluxes, 100 epochs, batch 5, five shuffled folds, three runs, NumPy seed 2 set once. It draws **1000 indices with `replace=True`** from 11,000 simulation rows for each run. Duplicated sampled rows can appear in both training and held-out folds; CV is not grouped by original simulation row. Biomass R2 and constraint losses are calculated afterward from full-flux predictions. This is a separate duplication risk from the clean condition-level EXP110 split. [AMN: notebook cells 6–7; `Library/Build_Model.py` lines 232–280.]

The committed `Result/Raw_data/iML1515_EXP_UB_ANN_Dense_PRED.csv` and `_Q2.csv` contain three 110-condition experimental prediction runs and reproduce the score below, but **no matching experimental ANN execution cell** was found in the inspected ANN notebook. Its exact architecture, preprocessing, seeds, and split provenance cannot be inferred from its filename or the simulated ANN settings.

The experimental random-forest comparison is executable in `Figures.ipynb` cell 14: the ten variable indicators, `GR_AVG` target, `RandomForestRegressor(n_estimators=500,max_depth=None,random_state=i)`, five runs `i=0…4`, and `cross_val_predict` with `KFold(10,shuffle=True,random_state=i)`, `n_jobs=10`. No feature or target scaling is applied. The paper states 1000 estimators and the same CV scheme as the three-run AMNs; the current figure cell differs in both estimator and repetition counts. No forest was retrained during this audit.

Hyperparameter selection is justified by simulated E. coli core experiments: supplement p. 17 describes 1000 simulated samples, 100 epochs, and five-fold comparisons of depth, width, and learning rate. It motivates one hidden layer and learning rate 0.001, with larger widths for larger networks. The committed grid-search/history artifacts under `Result/Raw_data/` document these comparisons, but they do not provide a complete nested selection history for EXP110 or a seed manifest for all shipped models.

### Held-out prediction aggregation and checked metrics

For direct AMNs, fold predictions are written into the original row positions of `Ypred`, so each outer run yields a 110-element vector of held-out growth predictions. The notebooks calculate a **pooled** sklearn score for each vector,

```text
Q2_run = 1 - sum_i (growth_i - prediction_i)^2
              / sum_i (growth_i - mean(growth))^2
```

and save three prediction rows and three Q2 values. Their displayed mean/SD uses `np.mean` and `np.std` (**population SD, ddof=0**) across runs. Fold-level mean R2/SD in `ReturnStats` is a different statistic, because each fold has its own target mean and denominator. R2 denotes in-sample fit and Q2 denotes predictive use in the paper, but the calculation itself is the same sklearn function; naming alone cannot establish a held-out evaluation.

`Figures.ipynb` cell 16 plots the mean prediction for each condition, horizontal errors from `EXP110.GR_STD` (sample SD over retained technical replicates), and vertical errors from pandas `data.std(axis=0)` (**sample SD, ddof=1**, over three prediction runs). Those error bars are not standard errors or confidence intervals. The plotted point means have a different R2 from the mean of the three run-specific Q2s.

Recalculated against `EXP110.GR_AVG` without retraining:

| Artifact in `AMN:Result/Raw_data/` | Per-run pooled R2/Q2 | Mean ± population SD | R2 of mean prediction |
|---|---|---|---:|
| `iML1515_EXP_UB_AMN_QP_PRED.csv` | 0.817634, 0.747827, 0.768338 | **0.777933 ± 0.029295** | 0.808120 |
| `iML1515_EXP_UB_AMN_LP_PRED.csv` | 0.785962, 0.798795, 0.769069 | **0.784609 ± 0.012173** | 0.808042 |
| `iML1515_EXP_UB_AMN_Wt_PRED.csv` | 0.767517, 0.766808, 0.787296 | **0.773874 ± 0.009496** | 0.786290 |
| `iML1515_EXP_UB_ANN_Dense_PRED.csv` | 0.782540, 0.777594, 0.769708 | **0.776614 ± 0.005285** | 0.806474 |
| `iML1515_UB_AMN_QP_RC_AMN_PRED_train.csv` | 0.975219, 0.974122, 0.975561 | **0.974967 ± 0.000614** | 0.975363 |
| `iML1515_UB_AMN_QP_RC_AMN_PRED_pred.csv` | 0.767077, 0.727220, 0.786610 | **0.760302 ± 0.024715** | 0.776769 |

Every companion `_Q2.csv` agrees with its prediction-derived values to displayed precision. The first three rows support the Fig. 3 labels **QP 0.78±0.03, LP 0.78±0.01, Wt 0.77±0.01**. The ANN row is an artifact calculation, not a verified replication of its training protocol.

The actual plotting inputs for **Fig. 5c**, **Fig. 5d**, and **Fig. S9** are respectively `Cobra_train.csv`, `Cobra_alone.csv`, and `Cobra_pred.csv`, identified by `Figures.ipynb` cells 28, 30, and 44. Their recalculated scores are **0.968089**, **0.518178**, and **0.786445**. `Cobra_alone.csv` has a misleading extension: it is **110 binary float64 values**, read with `np.fromfile`, not text CSV. The scalar-search table `Cobra_scaler.csv` is semicolon-delimited; it lists 2.0–3.0 by 0.1 and a maximum recorded R2 of 0.5182 at **2.5**. Fig. 5's printed baseline 0.51 and Fig. S9's 0.78 are close to these values but do not use standard nearest-two-decimal rounding (which would give 0.52 and 0.79).

The source-data workbooks are **not identical prediction evidence** to those raw CSVs. The copies inside `Result/Source Data.zip` are byte-identical to the individual XLSX files, so the conflict exists within committed release artifacts:

| Source-data workbook / sheet | Recalculated score against its own target column | Difference from plotting CSV evidence |
|---|---|---|
| `Data_Fig3.xlsx`, `Fig3`, QP1–3 | **0.778581 ± 0.032819** | Different individual predictions; approximately the same summary as raw QP |
| Same sheet, LP1–3 | **0.782246 ± 0.024892** | SD differs from Fig. 3's 0.01 and raw LP's 0.012173 |
| Same sheet, Wt1–3 | **0.705032 ± 0.037019** | Does not reproduce Fig. 3's 0.77±0.01 |
| `Data_Fig5.xlsx`, `E. coli UB for cobra`, `GR Cobra` | **0.956005** | Does not equal `Cobra_train.csv` score 0.968089 |
| `Data_FigS9.xlsx`, `E. coli UB for cobra (preds)`, `GR_Cobra` | **0.771173** | Does not equal `Cobra_pred.csv` score 0.786445 |

Fig. 3 workbook targets preserve the same condition order and agree with the model CSV targets; the different prediction values are not a row-permutation explanation. The Fig. 5/S9 workbook targets are rounded to four decimals; that small target rounding does not account for the prediction differences. These workbooks lack enough run metadata to establish why a different artifact set was deposited.

The additional experimental-variability calculation in `Figures.ipynb` cell 11 draws independent normal samples with mean `GR_AVG` and scale `GR_STD`, calculates R2 against the means, and repeats 1000 times; no seed is set in that cell. Its saved output is 0.911 ± 0.019, close to the paper's 0.91±0.02. This is a noise-comparison simulation, **not a proven upper limit** on achievable prediction. Cell 12's interval-overlap helper contains incorrect disjunctions, including `max_pred < min_true` as a success case, so its “within variances” fractions should not be accepted without correcting/recomputing the logic. This does not enter the training or pooled Q2 calculations. [Paper p. 7; AMN `Figures.ipynb` cells 11–12.]

### Verified execution sequence for reproduction

1. Preserve the released plate-specific exclusion lists and search windows; process raw plates to retained replicate growth rates, condition means, and sample SD. Retaining the current typo is necessary to reproduce the committed measurements exactly.
2. Use the verified 110×38 binary input array and unscaled mean-growth targets. For direct Fig. 3 workflows, use the current **543×1080 experimental reduced model** and record that it differs from the paper's 550×1083 description. Train QP/LP/Wt independently from fresh initialization with the settings above, split 99/11 by shuffled **ordinary KFold**, repeat with NumPy seeds 1–3, and collect held-out predictions.
3. For reservoir work, distinguish recreating simulation training from loading the shipped checkpoint. The committed simulated dataset has 11,000 rows, fixed caps 2.2, mixture-scaled variable caps up to 11, and a 550-reaction model. The shipped checkpoint's effective settings include scaler 11 and **zero QP steps**; current cell 14 and Table S1 do not recreate its saved training history verbatim.
4. For Fig. 5's experimental fit, freeze that checkpoint, train the 10→500→10 cap prior on **all 110 means**, Adam 0.0001, 1000 epochs, batch 5, seeds 0–2. Extract the first run's caps, restore their scale, and run pFBA with the selected handoff table. A CV reproduction of Fig. S9 requires a separate explicit 10-fold RC run; the current committed E. coli cell does not select it.
5. Use the exact plotting CSV artifacts to reproduce the reported graph scores. Record whether full-precision raw caps, rounded headered handoff caps, or workbook caps were used for any new downstream solver evaluation; their provenance differs.

## 4. Material differences between publication and implementation

| Issue | Publication claim | Verified release evidence | Consequence / classification |
|---|---|---|---|
| Fixed simulated caps | Obligatory uptake bounds 10 (paper pp. 9–10) | `iML1515.csv` and saved simulation X use **2.2** for all 28 fixed inputs | **Contradiction** for iML1515 generation; changes organic supply, oxygen and mineral capacity. |
| Variable simulated caps | Random between 0 and 2.2, zero excluded (paper p. 10) | Line 543 multiplies by k+1; saved caps reach **4.4–11**, discrete 99-level grid | **Contradiction**; simulation coverage depends on mixture cardinality. |
| Physical glycerol | Minimal-model medium includes glycerol (paper p. 9); physical recipe lists amino acids (p. 12) | Glycerol is fixed in all model inputs; no physical supplement concentration found | **Missing experimental evidence**, not proof glycerol was supplied to cultures; materially affects carbon interpretation. |
| Amino acid recipe | Mix total 0.04 g/L, four constituents 5 mg/L each (p. 12) | Listed component sum is 0.020 g/L | **Internal publication inconsistency**; physical recipe needs clarification. |
| Replicate statistics | Methods: average 4.6±1.6; Fig. 3: three measured replicates | 557 retained values, average **5.06**, 2–8 per medium | **Contradiction**; use actual per-condition replicate counts and SD. |
| Growth-window details | One-hour regression window (p. 12) | Six samples span 50 minutes; four smoothing passes, blank subtraction disabled | **Implementation detail / timing discrepancy**; exact raw-data reproduction needs code, not just prose. |
| CV | Stratified 10-fold, three repeats (pp. 5–6) | Ordinary shuffled `KFold`, no stratification, three direct-AMN runs | **Contradiction**; fold composition can differ, though condition means are kept intact. |
| Experimental model dimensions | 550 reactions, 1083 metabolites (Fig. 3) | Experimental NPZ/XML **543/1080**; simulated reservoir **550/1083** | **Committed artifact mismatch**; seven removed reaction IDs listed below. |
| LP initialization | Shadow prices initialized to zero, 1083 values (Fig. 3 caption) | A learned linear-output M0 network with **2m+n=2703** dual values | **Architecture contradiction**; additional trainable initializer and different dual representation. |
| Mechanistic-loss weights | Eq. (2): normalization by m, nin, n | Squared residual norms divided by **m², nin², n²** before average | **Loss contradiction**; alters balance of growth fit and feasibility penalties. |
| Wt dropout | Fig. 3 specifies dropout 0.25 for all variants | Wt stores rate but never applies dropout in its cell | **Effective-setting discrepancy**; current recurrence does not match assumed Dense regularization. |
| Reservoir configuration | Table S1's QP has four mechanistic iterations, 100 epochs | Current simulation cell: four steps, 25 epochs; shipped parameters: **zero steps, 500 epochs, xfold 0** | **Code/checkpoint/publication mismatch**; do not claim the released checkpoint was obtained by the current cell or Table S1 protocol. |
| Reservoir score | Fig. 5 fit ≈0.97; Fig. S9 held-out ≈0.78 | Current RC cell fits all rows; raw downstream scores 0.968089 vs 0.786445 | **Agreement in evaluation intent**, with missing executable/logged provenance for the historical CV export; the 0.97 is not experimental held-out performance. |
| RC outer loss | General Methods discuss growth plus constraint losses for AMNs | RC has one target column; optimizes growth MSE only | **Variant-specific implementation distinction**; mechanistic diagnostics are not RC training penalties. |
| Random forest | 1000 estimators, same CV scheme as AMN (p. 6) | Figure cell has 500 estimators and five seeded repeats | **Comparison-setting mismatch**; deposited number cannot be regenerated by assuming the stated settings. |
| Source-data consistency | Workbooks deposited as source data for figures | Workbook Wt, LP uncertainty, and downstream FBA values differ from plotting CSVs | **Artifact provenance conflict**; report the artifact used, not just a figure label. |
| Reproducibility environment | README claims matching original package versions | Environment requests Keras 2.6; H5 says **2.9.0**; several packages unpinned | **Environment mismatch / missing version evidence**; exact replay is not established. |

The 543-reaction experimental model omits the following seven reactions present in the 550-reaction simulated model: **`EX_h2o_e_o`, `G6PDH2r_for`, `GND`, `H2Otex_o`, `PGL`, `TRPAS2_rev`, `TRPS3`**. This includes oxidative pentose-phosphate reactions, so the difference is not solely cosmetic bookkeeping. Provenance of that additional reduction is not recorded in the inspected experimental-building path. Both XMLs and NPZs are committed, unchanged artifacts.

There is also a reproducible exclusion-list error: `Library/Build_Experimental.py` line 115 contains adjacent literals **`"C5" "F7"`** without a comma in `outliers_20220514`, producing the invalid well name **`C5F7`**. Therefore C5 and F7 remain active; recomputing the committed function exactly matches their saved inclusion. Correcting the typo would change two replicate measurements and their condition means/SDs, but would not account for the full publication/release difference in average replicate counts. No correction was made in this audit.

## Important limitations and unresolved questions

- This is an audit of the committed post-publication release and its local working copy, not a reconstruction of the exact manuscript-submission checkout. The local source edits do not explain the principal contradictions above: those occur in unchanged committed files. Saved results can reproduce graph numbers without proving which code revision or checkpoint generated them.
- Missing evidence includes the physical glycerol dose (if any), resolution of the amino-acid mixture total, provenance of the 543-reaction experimental model, exact experimental ANN training settings, historical RC CV fold assignments/cap exports, and the release's inconsistent source-data workbook runs.
- NumPy seeds are available in notebooks, but TensorFlow seeds, full dependency locks, fold index manifests, optimizer histories, and original training logs are insufficient for bit-for-bit model reproduction. No held-out experimental labels are demonstrably reused as fold training targets in the direct AMN workflow; stronger claims about hidden historical tuning cannot be established from the available evidence.
- Neural feasibility is soft and requires independent checking. The shipped zero-step reservoir is especially important to distinguish from a converged mechanistic solver. High growth R2 alone does not establish internal mass balance, realistic uptake rates, or physiological equivalence between modeled fixed supplements and the measured cultures.
- The scalar baseline uses all 110 labels for cap selection; Fig. 5 is an in-sample learned-cap result. Fig. 3 and the separate historical Fig. S9 artifacts are the relevant predictive evidence. Plate-grouped generalization and transfer to strains/media outside this 110-condition design are not tested by the inspected workflow.
- Verification included all plate replicate values/missing masks, concatenation, experimental CSV/NPZ alignment, source membership/cardinality counts, simulated cap ranges/pattern blocks, XML reaction/bound inventory, saved H5 configuration, stored Q2 versus predictions, workbook comparisons, and read-only Git provenance checks. No models were retrained and no downstream FBA scores were regenerated from caps; the latter scores were recalculated from **saved growth predictions**.
- Artifact identity (SHA-256): `EXP110.csv` = `3d7c024aae9b7a66a2bfbeb129254a9ff8f7043caa60c148e57d5df8dbae6ff9`; `iML1515_EXP_UB.npz` = `ee0ee1441b18b9c8e856bed9b2f6fb1ee986b19b73637c0b6eb2411db2421706`; `iML1515_UB.npz` = `f5edf679c7d21b7e3b2f4158e4098defa69fd73835ad96eb6eeb024572c298ac`; `iML1515_UB_AMN_QP_model.h5` = `6f0f8569d81a246f8ef5eae58f324911b6566670f6e77fcea1e2a1e794a04dc9` (paths under AMN folders identified above).
