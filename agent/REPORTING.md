# Reporting to a biologist

Write short, plain paragraphs. Use tables for counts. Do not interpret biology
unless the user asks for it. Name what was done, then what it can and cannot show.

## Order

1. **Counts first.** Patients (cases, controls or events), datasets, donors, cells per cluster and per donor, shared genes, retained gene-set sizes.
2. **Validation.** CV scheme, held-out metric per fold or dataset with its mean, the sizes kept by the one-standard-error rule, and the sentence that performance after size selection is optimistic. If the metric is near chance, say it first.
3. **Score distribution.** Ensemble score by cluster and donor. Say the score is a percentile rank.
4. **Within-cluster findings.** For each cluster: groups used, donors contributing, cells high and low, ties, DE method actually used, top genes with direction and effect size, FDR, pathways. List skipped clusters and why.
5. **Optional results.** Model attribution, spatial neighborhoods, C-SIDE. Label each as its type.
6. **Limitations.** Generated from what ran (see below).
7. **Paths.** Report file, run directory, config, manifest.

## Limitations to state when true

- Few donors (state the number). Donor-level tests are exploratory below about 5 donors per group.
- No replication, so cell-level Wilcoxon was used. Cells from one donor are not independent.
- The model used the listed genes, so score-associated genes are model attribution, not independent discovery.
- Feature size was chosen on the same folds that are reported.
- Bulk and cell data came from different people or platforms. Transfer was not validated on held-out cell donors unless donor labels exist.
- Ties were excluded; list the count.
- Small gene panel limits pathway tests.
- Bulk patient scores are in-sample unless labeled out-of-fold.

## Never claim

- That a score is a probability of disease or outcome.
- That a gene causes the outcome or is a biomarker, from these results alone.
- That cells are replicates, or that a p-value from cell-level tests describes patients.
- That a cluster "is" high-risk because it has a high mean score. Say "higher score".
- That validation proves the cell-level scores are correct. No cell-level ground truth exists.
- Anything about a result you did not read.
