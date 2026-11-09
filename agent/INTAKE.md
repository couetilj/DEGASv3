# Intake interview

Ask these questions before any command. Ask in the order given. Read the
files yourself when the user gives a path, then confirm what you found.
Flags are `degas3 init` flags. If a flag name differs in `degas3 init --help`,
the help text wins.

## A. Bulk data

| # | Question | Maps to |
|---|---|---|
| A1 | "Where is the bulk expression matrix? Is it samples x genes or genes x samples?" | `--bulk PATH` (orient a copy if needed) |
| A2 | "Are the values raw counts, or log-transformed (for example log2(x+1))?" | log-transformed: set the bulk `already_log` flag |
| A3 | "Where is the phenotype table, and which column holds the sample id?" | `--pheno PATH` |
| A4 | "Is the outcome binary (case/control) or time-to-event?" | `--outcome binary` or `--outcome survival` |
| A5 | Binary: "Which value is case and which is control?" Survival: "Which columns hold follow-up time and event (1 = event)?" | outcome column 0/1, or `time` and `event` columns |
| A6 | "Can one person have several samples? Which column identifies the person?" | `patient_id` column in the phenotype table |
| A7 | "Do the samples come from more than one study or batch? Which column?" | `--dataset-col NAME` |

Check after A: sample ids match the matrix; counts of case and control (or events);
number of patients; number of datasets.

## B. Single-cell or spatial data

| # | Question | Maps to |
|---|---|---|
| B1 | "Where is the single-cell or spatial data, and what format (h5ad, 10x, Xenium outs, Visium)?" | `--cells PATH` |
| B2 | "Is it dissociated single-cell, single-cell spatial (Xenium, CosMx, MERSCOPE), or spot-based (Visium)?" | `--modality sc` / `st_cell` / `st_spot` |
| B3 | "Which column identifies the donor (biological replicate)?" | `--donor-col NAME` |
| B4 | "Are there cluster or cell-type labels already? Which column? Do you trust them?" | `--cluster-col NAME` |
| B5 | No labels: "May I cluster the cells (PCA, neighbors, Leiden, UMAP)? What resolution, or use 1.0?" | run `degas3 cluster RUN --resolution R`; no `--cluster-col` |
| B6 | "Do you have a donor-level clinical label (0/1) you want compared with the score?" | `--label-col NAME` |
| B7 | Spatial only: "Which column is the section or acquisition, and which key holds x,y coordinates?" | sample column and spatial key (see `--help`) |
| B8 | "Is the matrix raw counts?" | must be yes; otherwise ask for the raw layer |

Check after B: donor count; cells per donor; cells per cluster; gene id format
(Ensembl, symbols, version suffix); shared gene count with bulk.

## C. Compute

| # | Question | Action |
|---|---|---|
| C1 | "Do you have a CUDA GPU on this machine?" | run `degas3 doctor` and read it |
| C2 | No GPU: "Do you have a SLURM cluster?" | plan `degas3 slurm` (see `docs/SLURM.md`) |
| C3 | No GPU and no cluster | tell the user. Offer `--allow-cpu` only for a tiny test |

## D. Choices (recommend, then ask)

| # | Question | Maps to |
|---|---|---|
| D1 | "Patient k-fold validation (default), or leave-one-dataset-out? (Only if at least 2 datasets.)" | config `cv` |
| D2 | "Model class: automatic (uses clusters if present), blank, or class?" | `--model auto|blank|class` |
| D3 | "Split each cluster at the median, or at quartiles 0.25 and 0.75?" | `degas3 stratify --split` |
| D4 | "DE method: automatic rule, or force a method?" | `degas3 de --method` |
| D5 | "Run SHAP explanations? Spatial neighborhoods? C-SIDE?" | optional stages |
| D6 | "Gene ids: do you have a mapping file?" | optional mapping CSV in config |

## E. Confirm

Read back a summary table: inputs, counts, modality, outcome, folds, model,
split, DE method, compute, run directory. Ask "Proceed?" Use a new run directory
name. Then run `degas3 init` and `degas3 inspect`.
