# Methods

This describes what DEGASv3 implements, following `docs/SPEC.md`. Each section
says what the step is and what it is not. Default values are config keys.

## 1. Preprocessing

Bulk and cell genes are intersected (version suffixes stripped). Counts are
transformed per sample with `1.5 ** log2(x+1)`, then z-scored and min-max
scaled across the selected genes. This happens after gene selection, separately
for bulk and cells. Log-transformed bulk input is flagged and handled.
*It is: a deterministic rescaling that puts bulk and cell vectors on a common
range. It is not batch correction.*

## 2. Model

DEGAS uses a shared feature extractor (3 sigmoid layers, 25 features, dropout
0.5) with an outcome head, a Wasserstein discriminator with gradient penalty
that aligns bulk and cell features, and an optional cell-label head. Binary
outcomes use a 2-class softmax; survival uses a Cox partial-likelihood loss.
Training: Adam, learning rate 0.01, 300 updates, class-balanced batches of 200
patients and 200 cells. `model: auto` uses the label head when clusters exist.
Scoring a cell uses the feature extractor and outcome head only.
*It is: a way to give each cell a number reflecting similarity to the patient
outcome. It is not a classifier of cells with known ground truth.*
Citations: Johnson et al. 2022; DEGASv2 preprint (Liu et al. 2025).

## 3. Fold-internal feature ranking

Within each training fold only, genes are ranked by residual variance of
log-CPM bulk expression (after removing dataset means for ranking), ties by
mean. An optional `mixed` mode alternates variance genes and cluster marker
genes. Smaller sets are prefixes of larger ones.
*It is: outcome-blind gene selection inside each fold. It is not a test of which
genes differ between cases and controls.*

## 4. Cross-validation and one-standard-error selection

Default: 5 patient-grouped folds, stratified by outcome, 3 seeds per fit.
Alternative: leave one dataset out. Metrics per held-out unit: AUROC (primary),
average precision, log loss, Brier; for survival, Harrell C-index. Metrics are
averaged with equal weight per fold. A bootstrap of patients (2,000 resamples,
fitted models fixed) gives the standard error of the best size. Every size
within one standard error of the best is retained.
*It is: a way to avoid trusting one noisy best size. It is not an unbiased
estimate of performance, because the same folds choose the size. The standard
error is not a confidence interval and omits retraining variance.*

## 5. Rank ensemble

Final models are fitted on all bulk samples for each retained size with 5
seeds. For each size, seed outputs are averaged and ranked across all scored
cells (average ties, divided by N). The ensemble score is the equal mean of
those percentiles, and is not ranked again. Raw scores are kept.
*It is: a stable ordering of cells. It is not a probability, and it is not
normalized within donor unless `rank_scope: sample` is set.*

## 6. Clustering and stratification

If no cluster labels are given, scanpy computes normalized log counts, 2,000
highly variable genes, PCA (50), neighbors (15), Leiden (resolution 1.0), and
UMAP, without outcome or scores. Within each cluster, cells are split at the
median or at quantiles (0.25, 0.75) of the ensemble score. Cells equal to a cutoff
are ties and excluded.
*It is: a within-cluster contrast that separates score effects from cell-type
effects. It is not a comparison of clinically labeled groups.*

## 7. Differential expression

| Method | When | What it does |
|---|---|---|
| Pseudobulk DESeq2 | at least 2 donors with both groups and at least 10 cells each | raw counts summed per donor and group; PyDESeq2 with `~ donor + group`; BH FDR |
| Cell-level Wilcoxon | no replication, or requested | Wilcoxon rank-sum, tie-corrected, on log-normalized counts; log2FC, AUC, BH FDR |
| edgeR (optional) | `--engine edger`, R available | same pseudobulk and designs; filterByExpr, TMM, quasi-likelihood F-test |

*Pseudobulk is: a donor-aware comparison where each donor contributes one
high and one low profile. Cell-level Wilcoxon is not donor-aware: its p-values
describe cells, not patients. Neither shows a gene causes the outcome.*
A clinical contrast (donor-level labels, `~ dataset + label`) is added when
labels exist, with a scatter against the high/low log2 fold changes. Genes used
by the model are flagged: their agreement is model attribution.

## 8. Pathways

Gene-set enrichment (GSEA preranked) on the DE statistic, using MSigDB Hallmark
by default (15 to 500 genes per set, 1,000 permutations, seed 42), plus
hypergeometric over-representation of significant up and down genes against
the tested background.
*It is: a summary of which annotated programs sit at either end of a ranked
list. It is not independent confirmation, since the ranking comes from a model
score.*

## 9. SHAP attribution (optional)

Gradient SHAP on each final model with a reference of low-score cells (below
the cluster's 25th percentile) from the same cluster, donor-balanced. Targets
are at most one cell per donor and quartile above Q25. Reports signed and
absolute means, pooled and donor-equal, with donor bootstraps and additivity
checks.
*It is: model attribution relative to low-score same-cluster cells. It is not
an effect of perturbing a gene, and independent substitutions can create
unrealistic expression combinations.*

## 10. Spatial neighborhoods (single-cell spatial)

Per section, a Delaunay graph on centroids (long edges dropped at the 99th
percentile length). Graph-geodesic radii are calibrated so median neighborhoods
hold target sizes (5, 10, 20, 40 cells). Raw counts in each neighborhood around
sampled seed cells are pooled, scored with the final models, and ranked within
the section.
*It is: a way to give sparse small-panel data a count depth closer to bulk.
Neighborhoods overlap and are not independent.*

## 11. RCTD and C-SIDE (spot-based)

RCTD decomposes spots into reference cell types; C-SIDE tests, within a cell
type, how expression relates to a spot-level covariate, here the scaled
ensemble score, across replicate sections, then summarizes at population level.
*It is: a cell-type-aware spatial association. It is not a test of cell-level
causality.*

## Citations

| Method | Reference |
|---|---|
| DEGAS | Johnson TS et al. Genome Med 2022;14:11. doi:10.1186/s13073-022-01012-2 |
| DEGASv2 | Liu Z, Li S, Couetil J, Zhang J, Huang K, Johnson TS. bioRxiv 2025. doi:10.1101/2025.02.03.636036 |
| PyDESeq2 | Muzellec B et al. Bioinformatics 2023;39(9):btad547. doi:10.1093/bioinformatics/btad547 |
| edgeR | Robinson MD, McCarthy DJ, Smyth GK. Bioinformatics 2010;26:139-140. doi:10.1093/bioinformatics/btp616 |
| GSEA | Subramanian A et al. PNAS 2005;102:15545-15550. doi:10.1073/pnas.0506580102 |
| GSEApy | Fang Z, Liu X, Peltz G. Bioinformatics 2023;39(1):btac757. doi:10.1093/bioinformatics/btac757 |
| MSigDB Hallmark | Liberzon A et al. Cell Syst 2015;1:417-425. doi:10.1016/j.cels.2015.12.004 |
| SHAP | Lundberg SM, Lee SI. NeurIPS 2017 (arXiv:1705.07874) |
| RCTD | Cable DM et al. Nat Biotechnol 2022;40:517-526. doi:10.1038/s41587-021-00830-w |
| C-SIDE | Cable DM et al. Nat Methods 2022;19:1076-1087. doi:10.1038/s41592-022-01575-3 |
| scanpy | Wolf FA, Angerer P, Theis FJ. Genome Biol 2018;19:15. doi:10.1186/s13059-017-1382-0 |
| Leiden | Traag VA, Waltman L, van Eck NJ. Sci Rep 2019;9:5233. doi:10.1038/s41598-019-41695-z |
