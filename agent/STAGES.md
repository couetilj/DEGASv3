# Stage reference

Every stage writes `<run>/<stage>/complete.json` (inputs hash, parameters,
version, timestamps, SLURM job id if any). A rerun skips completed stages.
Layout is in `docs/SPEC.md`.

## doctor
- Command: `degas3 doctor [--run RUN]`
- Outputs: console report.
- Success: torch imports, CUDA device listed, GPU memory shown.
- Failure: no CUDA. Fix: install a matching torch build, load the cluster GPU module, or use SLURM. Do not continue on CPU without user consent.

## init / inspect
- Commands: `degas3 init RUN --bulk X --pheno Y --cells Z [...]`, then `degas3 inspect RUN`.
- Inputs: bulk, phenotype, cells.
- Outputs: `config.json`, `inputs/bulk.npz`, `inputs/cells.h5ad`, `inputs/input_manifest.json`.
- Success: patient, donor, cell, gene counts printed; hashes recorded.
- Failures: sample ids not matching (fix the phenotype copy); non-integer SC counts (find the raw layer); missing donor column (ask); log-transformed bulk not flagged (set the flag).

## cluster (optional)
- Command: `degas3 cluster RUN [--resolution 1.0]`
- Outputs: cluster labels in `obs["degas3_cluster"]`, markers, UMAP.
- Success: sensible cluster count; each cluster present in at least 2 donors.
- Failures: too many tiny clusters (lower the resolution, ask the user); one giant cluster (raise it).
- Clustering never uses outcome or scores.

## prepare
- Command: `degas3 prepare RUN`
- Outputs: `shared_genes.txt`, `folds.json`, `rankings/<fold>.csv`, `tasks.json`.
- Success: shared genes reported; no patient appears in two folds; each fold has both outcomes (or events).
- Failures: too few shared genes (check id format and version suffix); a fold without cases (fewer folds or more patients needed); donor overlap warning (cells excluded per fold; report it).

## train
- Command: `degas3 train RUN --all` (or `--task N` for one task; use SLURM arrays for many).
- Outputs: `cv/task_###/seed_#/` with checkpoint, losses, held-out predictions.
- Success: all tasks complete; losses finite and falling.
- Failures: CUDA out of memory (lower sizes or use a larger GPU); NaN loss (check bulk scaling and the log flag); job killed (rerun the same command; finished tasks skip).

## select
- Command: `degas3 select RUN`
- Outputs: `cv/metrics.csv`, `cv/selection.json`, `cv/selection.png`.
- Success: primary metric (AUROC or C-index) above chance in most folds; retained sizes listed.
- Failure signs: metric near chance for all sizes. Tell the user the bulk outcome may not be predictable from these genes. Do not proceed to interpret cells without saying so.
- Always note that performance after selection is optimistic.

## fit
- Command: `degas3 fit RUN --all`
- Outputs: `models/size_####/seed_#/`.
- Success: retained sizes x 5 seeds present.

## score
- Command: `degas3 score RUN`
- Outputs: `scores/cell_scores.csv.gz` (raw per size, ensemble), `patient_scores.csv`.
- Success: one row per scored cell; ensemble in [0, 1].
- Notes: the ensemble is a mean of per-size percentile ranks. It is not a probability. Patient scores for bulk samples are in-sample and flagged; use out-of-fold predictions for validation.

## stratify
- Command: `degas3 stratify RUN [--split median|quantiles --quantiles 0.25 0.75]`
- Outputs: `cutoffs.csv`, `cell_groups.csv.gz`, `coverage.csv`.
- Success: every cluster has high and low cells; ties counted.
- Failures: many ties (score has little spread in that cluster; report it); a donor with cells in one group only (it will not enter paired DE).

## de
- Command: `degas3 de RUN [--method auto|pseudobulk_deseq2|wilcoxon] [--engine pydeseq2|edger]`
- Outputs: `de/<cluster>/method.json`, `results.csv`, pseudobulk counts and samples; `clinical/` when labels exist.
- Success: `method.json` names the method and donors used; skips are listed with reasons.
- Failures: not enough donors with both groups (auto falls back to cell-level Wilcoxon; say so); edgeR requested but R or edgeR missing (use PyDESeq2 or install R).

## pathways
- Command: `degas3 pathways RUN [--gmt PATH | --collection h.all --species human]`
- Outputs: `high_vs_low_gsea.csv`, `ora_high.csv`, `ora_low.csv` per cluster.
- Failures: gene set download blocked (give a GMT file); species mismatch (set `--species`); few tested genes (small panel; report it).

## explain (optional)
- Command: `degas3 explain RUN`
- Outputs: `shap_values.npz`, `gene_attributions.csv`, figures per cluster.
- Success: additivity residuals small; donors per cluster listed.
- Label as model attribution relative to low-score same-cluster cells.

## spatial (optional)
- Commands: `degas3 neighborhoods RUN`, `degas3 maps RUN`, `degas3 cside RUN`.
- Single-cell spatial: neighborhoods pool counts in graph radii and score the pools. Spot-based: spots are scored directly; C-SIDE needs R and spacexr.
- Failures: missing coordinates (ask for the key); C-SIDE without R (skip and say so).

## report
- Command: `degas3 report RUN`
- Outputs: `report/index.html`, `METHODS.md`, `manifest.json`, `figures/`.
- Success: sections for every completed stage; limitations section present.

## slurm
- Command: `degas3 slurm RUN --account A --partition P [--gpus 1] > jobs.sh`
- See `docs/SLURM.md`.
