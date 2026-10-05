# iML1515 Broad Organic-Source Pool Literature Review Prompt

We are designing two broad simulated pretraining distributions, D and E, for an iML1515-based flux-prediction model of **unmodified *E. coli* K-12 MG1655**. This strain is the physiological target throughout the literature search and source-pool selection. We have already screened iML1515 for organic exchange reactions that permit growth and saved the candidate list, with exchange IDs, in:

```text
data/reference/iML1515_growth_carbon_sources.csv
```

Your task is to determine (1) the final carbon/organic-source pool and its exact size, and (2) a defensible maximum K for randomly drawing 1–K distinct sources in one simulated medium. Treat the pool beyond the mandatory AMN/MINN sources below, and the choice of K within the computational constraints below, as open research questions. Do not anchor on a previously proposed pool size or maximum.

## Mandatory AMN and MINN source coverage

**Every AMN and MINN source listed below must be included in both the D and E source pools.** These are fixed inclusion requirements, not candidates for exclusion based on the literature review. Investigate and document their evidence and physiological qualifications alongside the other candidates. If support is incomplete or condition dependent, report that limitation while retaining the required source; do not invent evidence. Verify the exact exchange IDs and metabolite names against the candidate data and model.

### AMN: 15 required sources

| Compound | Exchange ID |
|---|---|
| D-Ribose | `EX_rib__D_e` |
| Maltose | `EX_malt_e` |
| Melibiose | `EX_melib_e` |
| Trehalose | `EX_tre_e` |
| D-Fructose | `EX_fru_e` |
| D-Galactose | `EX_gal_e` |
| Acetate | `EX_ac_e` |
| D-Lactate | `EX_lac__D_e` |
| Succinate | `EX_succ_e` |
| Pyruvate | `EX_pyr_e` |
| Glycerol | `EX_glyc_e` |
| L-Alanine | `EX_ala__L_e` |
| L-Proline | `EX_pro__L_e` |
| L-Threonine | `EX_thr__L_e` |
| Glycine | `EX_gly_e` |

### MINN: 1 required source

| Compound | Exchange ID |
|---|---|
| D-Glucose | `EX_glc__D_e` |

These 16 distinct sources must belong to the selectable pool. They do not all need to occur in each sampled medium, and their pool membership does not impose K ≥ 16.

## Literature and pool selection

Conduct a deep, systematic search of primary scientific literature. Investigate measured compositions of diverse industrial and agricultural side streams and other media used to grow microorganisms: lignocellulosic and crop hydrolysates, sugar-industry streams, brewery residues, dairy streams, fruit-processing residues, crude glycerol, algal biomass, food waste, fermented wastewaters, and other relevant feedstocks you discover. Search beyond these examples. Prioritize growth and substrate-uptake studies of unmodified *E. coli* K-12 MG1655. Evidence from other K-12 strains or other microorganisms may support composition or qualified physiological inference, but identify the strain explicitly and do not present adaptation or engineering results as native MG1655 capability.

For each candidate exchange, distinguish three different claims:

1. The compound was measured in a real feedstock or growth medium.
2. A microorganism was shown to consume it.
3. Unmodified *E. coli* K-12 MG1655 can consume it or grow on it under the intended oxygen conditions.

Presence alone must not be treated as evidence of *E. coli* uptake. Check known model-versus-physiology discrepancies, strain dependence, adaptation or engineering requirements, stereochemistry, and whether a compound was measured before processing, after hydrolysis, or only after fermentation. Prefer papers with quantified composition and actual microbial cultivation. Do not infer compound identity from an unresolved analytical peak.

Choose the additional pool members using explicit inclusion and exclusion criteria while retaining all mandatory AMN/MINN sources. Review every compound in the saved candidate list, and identify important compounds absent from that list that the model cannot represent; distinguish a missing model exchange from an exchange merely absent from the growth-positive candidate list. For the selected pool, provide a table with exchange ID, compound, feedstock or medium, direct evidence, physiological qualification, decision, and supporting paper. Mark mandatory AMN/MINN membership separately from evidence-based discretionary inclusion. Provide a shorter table explaining every exclusion, and explicitly verify that no mandatory source was excluded. The final integer must follow from these decisions and the mandatory-source union; do not select a target integer first.

## Choosing K under the sample budget

The training budget is **1,000,000 samples**. The sampled condition space must remain broad enough to represent useful biological diversity without spreading this finite budget so thinly that training coverage becomes sparse. **Recommend an integer K ≤ 20; K > 20 is outside the acceptable design space.** This ceiling is a computational design constraint, not a biological claim, and K = 20 should not be treated as the default. A smaller K may be preferable.

Investigate the number of distinct, available organic substrates that coexist in individual measured streams. Define what “present” means using an analytical detection or meaningful-concentration threshold, and show how the count changes if that threshold changes. Count resolved, accessible substrates eligible for the selected pool separately from the total measured organic inventory. Keep naturally occurring compositions separate from deliberately formulated laboratory media.

Base K primarily on **representative, practically relevant mixture richness across feedstock families**, with comparable thresholds and availability assumptions. Do not set K from the richest isolated report, a rare unusually complex side stream, a union across different samples, or an extensive trace-metabolite inventory. Report unusually rich cases as exceptions and explain their exclusion from the default distribution. Do not claim that heterogeneous or insufficient data establish a typical upper limit or percentile.

Combine this literature assessment with an explicit coverage argument for the one-million-sample budget. For final pool size N, compare several plausible K values satisfying K ≤ min(20, N) using the discrete subset count `sum(binomial(N, k), k=1..K)`. State the assumed sampling distribution over mixture cardinality and source identities, and discuss how sampling continuous uptake bounds further expands the condition space. Use these counts as a coverage diagnostic, not as proof that every subset must be observed or that cardinality alone determines learnability. Explain why the recommended K offers a sensible balance between representative mixtures and useful sample density; merely satisfying K ≤ 20 is insufficient justification.

Recommend one numerical K within this constraint, explain whether it is a chosen coverage cap, an observed representative bound, or a supported percentile-based choice, and state what realistic mixtures it excludes. If evidence does not identify a unique K, say so and justify the design value from both literature and sampling considerations without calling it a literature-established maximum. Make clear when computational limits require truncating even realistic richer mixtures.

## Source verification and citation format

Search until additional feedstock families and targeted searches for weakly supported candidates stop materially changing the decisions. Report your search strategy and the main evidence gaps. Verify every cited paper against the publisher, PubMed, or the paper itself; give DOI, full bibliographic details, and the specific table, figure, or passage supporting each claim. Return correct, copy-ready BibTeX for every paper used. **BibTeX citation keys must have the form `1stauthoryear`: the first author's surname immediately followed by the four-digit publication year**, for example `Schwalbach2012` and `Keating2014`. Use ASCII-normalized surnames without spaces or punctuation; add a lowercase suffix only to distinguish otherwise identical keys, such as `Smith2020a` and `Smith2020b`. Use these keys consistently when referring to papers in the source record and review; do not use generic numbered source IDs as BibTeX keys. Do not invent references or use a review article as the sole support for a composition or growth claim.

## Output files

Create the final curated pool as:

```text
data/reference/iML1515_broad_organic_source_pool.csv
```

The CSV must contain exactly two columns:

```text
exchange_id,metabolite_name
```

Include only the final selected pool, using the exact iML1515 exchange IDs and metabolite names from the candidate data where available.

Also save a complete source record as:

```text
data/reference/iML1515_broad_pool_sources.md
```

This file should contain all primary sources used to support inclusion, exclusion, physiological qualification, or the choice of K, together with their copy-ready BibTeX entries using the required `1stauthoryear` keys. For each source, include the citation key, full citation, DOI, URL, the relevant compound or feedstock, and the specific table, figure, supplementary item, or passage that supports the decision.

Save the full literature and design review as:

```text
docs/working_notes/iML1515_broad_pool_literature_review.md
```

Finish with these clearly labeled outputs:

- Final pool size: [integer]
- Final pool: [complete list of exchange IDs and compounds]
- Mandatory source coverage: [confirm all 15 AMN sources and MINN D-glucose are included]
- Recommended maximum K: [integer ≤ 20]
- What that maximum represents: [one precise sentence]
- Sample-budget justification: [brief explanation of coverage with 1,000,000 samples]
- Remaining biological limitations: [brief list]

This is a literature and design review. Do not edit model code, generator code, or existing datasets. Only create the three output files specified above.
