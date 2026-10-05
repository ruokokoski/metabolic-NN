# iML1515 Broad Organic-Source Pool Literature Review Prompt

We are designing two broad simulated pretraining distributions, D and E, for an iML1515-based *E. coli* flux-prediction model. We have already screened iML1515 for organic exchange reactions that permit growth and saved the candidate list, with exchange IDs, in:

```text
data/reference/iML1515_growth_carbon_sources.csv
```

Your task is to determine (1) the final carbon/organic-source pool and its exact size, and (2) a defensible maximum K for randomly drawing 1–K sources in one simulated medium. Treat both as open research questions. Do not anchor on a previously proposed pool size or maximum.

Conduct a deep, systematic search of primary scientific literature. Investigate measured compositions of diverse industrial and agricultural side streams and other media used to grow microorganisms: lignocellulosic and crop hydrolysates, sugar-industry streams, brewery residues, dairy streams, fruit-processing residues, crude glycerol, algal biomass, food waste, fermented wastewaters, and other relevant feedstocks you discover. Search beyond these examples. Include studies of *E. coli* growth and substrate uptake where available.

For each candidate exchange, distinguish three different claims:

1. The compound was measured in a real feedstock or growth medium.
2. A microorganism was shown to consume it.
3. The relevant *E. coli* strain can consume it or grow on it under the intended oxygen conditions.

Presence alone must not be treated as evidence of *E. coli* uptake. Check known model-versus-physiology discrepancies, strain dependence, adaptation or engineering requirements, stereochemistry, and whether a compound was measured before processing, after hydrolysis, or only after fermentation. Prefer papers with quantified composition and actual microbial cultivation. Do not infer compound identity from an unresolved analytical peak.

Choose the pool using explicit inclusion and exclusion criteria. Review every compound in the saved candidate list, and identify important compounds absent from that list that the model cannot represent. For the selected pool, provide a table with exchange ID, compound, feedstock or medium, direct evidence, physiological qualification, decision, and supporting paper. Provide a shorter table explaining every exclusion. The final integer must follow from these decisions; do not select a target integer first.

Investigate the number of distinct, available organic substrates that coexist in individual measured streams. Define what “present” means using an analytical detection or meaningful-concentration threshold, and show how the count changes if that threshold changes. Keep naturally occurring compositions separate from deliberately formulated laboratory media. Determine whether the literature supports a typical upper limit at all. Recommend one numerical K for the simulator, explain whether it is an observed maximum, a chosen coverage cap, or a percentile-based choice, and state what realistic mixtures it excludes. If evidence does not identify a unique K, say so and justify the recommended design value without calling it a literature-established maximum.

Search until additional feedstock families and targeted searches for weakly supported candidates stop materially changing the decisions. Report your search strategy and the main evidence gaps. Verify every cited paper against the publisher, PubMed, or the paper itself; give DOI, full bibliographic details, and the specific table, figure, or passage supporting each claim. Return correct, copy-ready BibTeX for every paper used. Do not invent references or use a review article as the sole support for a composition or growth claim.

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

This file should contain all primary sources used to support inclusion, exclusion, physiological qualification, or the choice of K. For each source, include the full citation, DOI, URL, the relevant compound or feedstock, and the specific table, figure, supplementary item, or passage that supports the decision.

Save the full literature and design review as:

```text
docs/working_notes/iML1515_broad_pool_literature_review.md
```

Finish with these clearly labeled outputs:

- Final pool size: [integer]
- Final pool: [complete list of exchange IDs and compounds]
- Recommended maximum K: [integer]
- What that maximum represents: [one precise sentence]
- Remaining biological limitations: [brief list]

This is a literature and design review. Do not edit model code, generator code, or existing datasets. Only create the three output files specified above.
