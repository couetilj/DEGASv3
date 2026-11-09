# DEGASv3 agent operating rules

## Purpose

You run DEGASv3 for a user. DEGASv3 transfers patient outcomes from bulk RNA-seq
(binary or time-to-event) to single-cell (SC) or spatial (ST) data. It then
compares high-score and low-score cells within each cluster.
You interview the user, check inputs and compute, run stages in order, and report
honestly. You do not edit package code. You adapt inputs or ask.

Detail files: `agent/INTAKE.md`, `agent/STAGES.md`, `agent/REPORTING.md`,
`agent/GUARDRAILS.md`, `docs/DATA_FORMATS.md`, `docs/SPEC.md`.

## 1. Intake interview

Ask before running anything. Use the exact questions in `agent/INTAKE.md`.
Ask in small groups. Do not guess answers.

Bulk data:
- Path to the bulk matrix. Raw counts or log-transformed?
- Path to the phenotype table. Which column holds `sample_id`?
- Outcome type: `binary` (which value is case, which is control) or `survival` (time and event columns).
- Is there a `patient_id` column (several samples per person)? Is there a `dataset` column (batch or study)?

Single-cell or spatial data:
- Path and format: h5ad, 10x (h5 or mtx), Xenium outs, Visium.
- Modality: `sc`; single-cell spatial (Xenium, CosMx, MERSCOPE); or spot-based (Visium).
- Donor column. Required for donor-level inference.
- Existing cluster or cell-type labels? If none, offer to cluster (PCA, neighbors, Leiden, UMAP).
- Optional donor-level clinical labels (0/1) for a clinical-versus-score comparison.
- For ST: section or acquisition column and coordinate key.

Compute:
- Run `degas3 doctor`. Confirm a CUDA GPU.
- No GPU: ask whether to use a cluster (`degas3 slurm`, see `docs/SLURM.md`). Never fall back to CPU silently. Use `--allow-cpu` only if the user agrees and the data are tiny.

## 2. Data contract

Follow `docs/DATA_FORMATS.md`. Check raw counts, gene id format, sample id
overlap between bulk matrix and phenotype, and donor values. Fix problems by
converting copies of the inputs. Never edit the user's originals.

## 3. Stage order

Run one stage at a time. After each, read `<stage>/complete.json` and the printed
counts and warnings. Stop on any error. Do not retry blindly.

| Stage | Command | Verify |
|---|---|---|
| Doctor | `degas3 doctor` | GPU found, versions printed |
| Init | `degas3 init RUN --bulk X --pheno Y --cells Z ...` | `config.json` matches the interview |
| Inspect | `degas3 inspect RUN` | patients, donors, cells, shared genes; case/control counts; warnings |
| Cluster (optional) | `degas3 cluster RUN [--resolution 1.0]` | cluster count; no tiny clusters; cluster sizes by donor |
| Prepare | `degas3 prepare RUN` | shared gene count; folds have no patient in two folds; task count |
| Train | `degas3 train RUN --all` | every `cv/task_###/seed_#/complete.json` exists; losses finite |
| Select | `degas3 select RUN` | `cv/metrics.csv`, `cv/selection.json`; retained sizes; metric above chance |
| Fit | `degas3 fit RUN --all` | one model per retained size and seed |
| Score | `degas3 score RUN` | `scores/cell_scores.csv.gz` row count equals scored cells; score range 0 to 1 |
| Stratify | `degas3 stratify RUN [--split median\|quantiles]` | `strata/coverage.csv`: high, low, tie per cluster and donor |
| DE | `degas3 de RUN [--method ...]` | `de/<cluster>/method.json`; donors used; skipped clusters and reasons |
| Pathways | `degas3 pathways RUN` | per-cluster tables; tested-gene counts |
| Explain (optional) | `degas3 explain RUN` | additivity residuals small |
| Spatial (optional) | `degas3 neighborhoods RUN`, `degas3 maps RUN`, `degas3 cside RUN` | per-sample outputs; R present for cside |
| Report | `degas3 report RUN` | `report/index.html`, `manifest.json`, limitations section |

`degas3 run-all RUN` chains the default stages. Prefer it only after the
decisions below are settled and written into `config.json`.

## 4. Decision points to surface

Ask the user. Give a recommendation and the reason. Record the answer.

| Decision | Options | Default and advice |
|---|---|---|
| Clustering | use supplied labels, or compute | Use supplied labels if they are trusted. Otherwise cluster. Ask for the Leiden resolution (default 1.0). |
| CV scheme | `patient_kfold` or `leave_one_dataset_out` | k-fold by patient. Offer leave-one-dataset-out only when `dataset` has at least 2 levels. |
| Model class | `auto`, `blank`, `class` | `auto`: class model when clusters exist. Explain the override. |
| Split | `median` or `quantiles` | `median` is the default. Quantiles (0.25, 0.75) drop the middle cells. |
| DE method | `auto`, `pseudobulk_deseq2`, `wilcoxon` | `auto` picks pseudobulk when at least 2 donors have both groups with enough cells. Otherwise cell-level Wilcoxon. Tell the user which was used. |
| Explain | yes or no | Optional and slower. Needs GPU time. |
| Neighborhoods | yes or no | Single-cell ST only. |
| C-SIDE | yes or no | Spot-based ST. Needs R and spacexr. |

## 5. Scientific guardrails

Full text and rationale: `agent/GUARDRAILS.md`.

- The patient is the unit for outcome inference.
- Donors are the unit for cell-level differential expression. Cells are not replicates.
- Rank genes only inside training folds.
- Never choose the feature size on the final test. Cross-validation after size selection is optimistic. Say so.
- Percentile scores are not probabilities.
- Genes the model used are model attribution, not independent discovery.
- Without replication, run cell-level Wilcoxon and name it that way.
- Report ties, coverage, and skipped clusters.
- Keep raw inputs read-only.
- Use a new run directory for each attempt. Never overwrite a completed run.
- Keep disease claims out of the report unless the user supplied them. Describe the data.

## 6. Reporting

Run `degas3 report RUN`. Follow `agent/REPORTING.md`.
- Give counts first: patients, donors, cells, clusters, shared genes.
- Then validation: held-out metric, retained sizes, one-standard-error rule, optimism note.
- Then within-cluster findings, with donors used and skips.
- State limitations from what actually ran: donor count, missing replication, panel size, tied cells.
- Give the report path and the run directory path.

## 7. Failure handling

- Read the error. Most errors carry a remedy.
- Fix inputs, not code. If code looks wrong, stop and tell the user.
- Resume with the same command. Completed stages skip.
- If a run is damaged or parameters change, make a new run directory.
- Do not delete `complete.json` files to force reruns unless the user agrees.

## 8. Conduct

- Use plain language. Short sentences.
- Do not claim results before the stage finishes and you have read the outputs.
- Do not invent flags. Run `degas3 STAGE --help` when unsure.
- Do not commit data, checkpoints, or run directories to git.
- Do not send expression data to external services.
