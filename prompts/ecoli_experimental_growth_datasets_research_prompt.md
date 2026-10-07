# Prompt: Find the best five experimental E. coli growth datasets for FluxTransformer

Conduct deep research to find new experimental growth datasets for unmodified
*Escherichia coli*. Select the **best five independent datasets** for testing
this repository's frozen FluxTransformer with a trainable MLP in front of it,
using the same approach as the AMN and MINN experiments.

Save the findings in **`docs/reference/ecoli_experimental_growth_datasets.md`**.
Write compactly. Use direct sentences. Explain what was searched, what was
found, why the five datasets were selected, and how each can be used. Do not
write a general review of bacterial growth or neural networks.

## Establish suitability for iML1515

Read repository instructions and the relevant sections of:

- `docs/experiment_notes/AMN_experiment_notes.md`
- `docs/experiment_notes/MINN_training_notes.md`
- `docs/experiment_notes/iML1515_sampling_study_notes.md`
- `ecoli_iML1515_AMN_model_testing.ipynb`
- `ecoli_iML1515_A_model_testing.ipynb`
- `ecoli_iML1515_B_model_testing.ipynb`
- Existing C/D/E evaluation notebooks and `iml1515_evaluation.py`, as needed.

Use the existing experiments to understand the general approach: train a
FluxTransformer on iML1515 simulations, then freeze it and train a front MLP
using experimental conditions and growth targets. Dataset discovery and
quality are the focus; do not prescribe MLP architecture or training details.

**Do not restrict selection to existing reservoirs, checkpoints, or nutrient
input lists.** Larger simulated nutrient sets and new iML1515 FluxTransformer
training are allowed when needed. Ordinary configuration and data preparation
may change. The dataset must support the same general workflow without
requiring a fundamentally different FluxTransformer architecture or GEM.

For each candidate, assess whether its nutrients and conditions can be
represented with **iML1515**, checking `models/iML1515.json` or
`models/iML1515.xml` for exchange identities. Identify any needed expansion of
the simulated nutrient set and any genuine model limitations. Distinguish
experimental uptake rates, medium concentrations, and nutrient presence.
Concentrations are not flux bounds. Do not treat an unmeasured nutrient as
absent. Assess fixed supplements, oxygen, and relevant simulation conditions.

## Search broadly and verify primary data

Search PubMed/Europe PMC, publisher articles and supplements, Google Scholar
or an accessible equivalent, Figshare, Zenodo, Dryad, institutional repositories,
and author code/data repositories. Search MediaDB, EcoCyc, PRECISE/iModulonDB,
and fluxomics collections such as CeCaFDB where accessible. Follow relevant
papers' references and citing studies. Record inaccessible sources explicitly.

Investigate the Aida–Ying defined-media growth collection, mixed-carbon growth
studies such as Hermsen et al., and experimental uptake/growth studies such as
Gerosa et al. These are leads, not predetermined winners. Search beyond them.
Check whether later data releases repeat earlier studies.

Use queries covering defined media, carbon-source mixtures, nutrient
concentrations, measured specific growth rates, growth curves, uptake/secretion
rates, chemostats, and strain names. Search across unmodified E. coli strains
without prioritizing one strain. Do not restrict discovery to the current
AMN/MINN nutrient lists.

Verify that each candidate uses unmodified E. coli and record the strain.
Assess documented strain-specific differences only where they affect iML1515
suitability. Exclude knockouts, engineered,
adaptively evolved, and selected mutant strains from eligible rows. A
mixed-strain dataset may qualify through a verified unmodified subset.
Do not infer that a strain is unmodified from a generic "wild type" label.

Open the actual data files, not just abstracts. Inspect their headers and
representative rows. Verify the link, filename, format, medium description,
growth target, units, replicate structure, and number of eligible conditions.
Downloading small files into a temporary directory is allowed. Do not import
datasets into this repository or run model training during this research.

## Eligibility and selection

An eligible dataset must have accessible experimental growth rates, or raw
growth curves from which specific growth rates can be estimated reproducibly;
known unmodified E. coli strains; sufficiently described conditions; and
suitability for iML1515 simulation and the FluxTransformer/front-MLP approach.
Needing a larger simulated nutrient set or a newly trained reservoir is not
a reason for exclusion.

Prefer quantitative specific growth rates over endpoint OD, yield, or
growth/no-growth. Biolog respiration alone is not a specific-growth-rate target.
For curves, specify blank correction, the exponential phase, time units, and
logarithm base. Flag diauxic phases, oxygen uncertainty, and missing metadata.
Do not silently interpret carrying capacity as growth rate.

Reject target leakage: inputs must not contain growth labels or uptake rates
calculated using those labels. Distinguish measured batch growth from a
chemostat growth rate imposed by dilution. Do not use dilution rate to predict
its equivalent growth target. Check overlap with the current AMN/MINN data;
repackaged benchmark rows do not constitute a new independent dataset.

Rank eligible candidates by: (1) quantitative growth-target and condition
quality, (2) suitability for iML1515, (3) independent nutrient diversity,
(4) usable condition count and replication, and (5) accessibility and
reproducibility. Explain trade-offs briefly. Count
unique conditions separately from wells, time points, replicates, and RNA-seq
samples. A large transcriptome collection is not automatically a large growth
dataset. For mixed collections, count only the verified eligible subset.

Seek five genuinely usable datasets. Continue targeted searches for gaps and
weak candidates before final selection. If fewer than five satisfy the hard
requirements, state the shortfall and list near-misses separately. Do not
present an incompatible or inaccessible dataset as eligible to fill the table.

## Required findings file

Keep the report concise, preferably within 1,200 words excluding bibliography
and URLs. Include:

1. **Scope and search:** search date, sources actually searched, representative
   queries, coverage, and access failures. State the iML1515 suitability criteria
   briefly. Keep model-workflow background to one or two sentences.
2. **Ranked selection table:** one row per selected dataset, up to five. Columns:
   rank and dataset; verified strain; eligible unique conditions and replicates;
   experimental inputs and growth target/units; iML1515 suitability and needed
   simulated nutrient-set expansion; main limitation; **verified dataset download link
   and exact filename/format**; accompanying article citation key.
3. **Use and selection rationale:** at most three short sentences per dataset.
   Give required preprocessing, exchange mapping or an exact mapping-file
   reference, and why it ranks above alternatives. State whether oxygen is
   measured or only described by aeration, and whether experimental inputs are
   composition or independently measured exchange rates.
4. **Rejected alternatives:** a compact table of the strongest rejected or
   overlapping candidates with concrete reasons. Distinguish inaccessible data,
   strain exclusions, insufficient metadata, and genuine iML1515 limitations.
   Do not reject a dataset solely because a current reservoir lacks its inputs.
5. **Exact article details:** for each selected dataset's associated article,
   give authors, year, exact title, journal, volume/issue, pages or article
   number, DOI, and the supplement/table establishing the data provenance.
   Use consistent `FirstauthorYear` keys. Verify details against primary sources.
   For standalone datasets, give creators, title, release/version, repository,
   and DOI/accession; state that no accompanying article was verified.

Prefer stable DOI/accession links plus direct file links. If a landing page is
needed, name the exact file and download step. Record authentication or license
restrictions. A paper link alone is not a verified dataset-loading link.
Label remaining uncertainty explicitly. Do not invent counts, measurements,
citations, or compatibility claims.

Write only the dedicated findings Markdown file. Preserve existing work. Do
not change notebooks, generators, models, datasets, or experiment notes.
At completion, link the report and state how many eligible datasets were
verified and any remaining iML1515 suitability limitations.
