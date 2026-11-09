# Tutorial on simulated data

This walk-through uses the built-in simulator. It runs in minutes. A real
dataset needs a GPU. On a laptop, add `--allow-cpu` and keep the data small.

## 1. Install and check

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
degas3 doctor
```

`doctor` lists torch, CUDA, GPU memory, and optional R tools. Without CUDA,
training stops unless you pass `--allow-cpu`.

## 2. Simulate

```bash
degas3 simulate demo_data --seed 0
```

The folder contains a bulk count matrix, a phenotype table (outcome, patient id,
dataset), and cell data with donors and clusters, plus spatial examples. List
the folder to see exact file names.

## 3. Initialize and inspect

```bash
degas3 init runs/demo --bulk demo_data/<bulk file> --pheno demo_data/<pheno file> \
    --cells demo_data/<cells file> --outcome binary --donor-col donor
degas3 inspect runs/demo
```

Read the summary: patients, cases and controls, donors, cells per cluster,
shared genes. `inputs/input_manifest.json` stores paths, sha256 hashes, shapes.

## 4. Run the chain

```bash
degas3 run-all runs/demo --allow-cpu
```

Or run the stages one by one (`prepare`, `train --all`, `select`, `fit --all`,
`score`, `stratify`, `de`, `pathways`) and read `complete.json` after each.

## 5. Report

```bash
degas3 report runs/demo
```

Open `runs/demo/report/index.html`.

## What each directory holds

| Directory | Contents |
|---|---|
| `config.json` | every parameter used |
| `inputs/` | hashed copies and manifest |
| `prepared/` | shared genes, folds, fold-internal rankings, task list |
| `cv/` | per-task checkpoints, held-out predictions, `metrics.csv`, `selection.json`, `selection.png` |
| `models/` | final models per size and seed |
| `scores/` | `cell_scores.csv.gz`, `patient_scores.csv` |
| `strata/` | cutoffs, high/low/tie assignment, coverage per cluster and donor |
| `de/<cluster>/` | `method.json`, `results.csv`, pseudobulk inputs |
| `pathways/<cluster>/` | enrichment tables |
| `explain/`, `spatial/`, `cside/` | optional stages |
| `report/` | `index.html`, `METHODS.md`, `figures/`, `manifest.json` |

## How to read the report

1. **Counts.** Confirm the numbers match your design.
2. **Cross-validation curve.** Each point is a gene-set size. Retained sizes lie within one standard error of the best. Values near 0.5 AUROC mean no outcome signal was found. After selection the numbers are optimistic.
3. **Score maps and violins.** Scores are percentile ranks (0 to 1). Donor points show donor-level spread. Look for donors that disagree.
4. **Coverage table.** High, low, and tied cells per cluster and donor. Missing groups explain skipped tests.
5. **DE tables.** Check the method line (pseudobulk or cell-level Wilcoxon) and the donors used. Effect sizes are log2 fold changes. Genes the model used are flagged.
6. **Pathways.** Positive enrichment means toward the high-score side.
7. **Limitations.** Read this section last but do not skip it.

In the simulator the outcome signal is built in, so a good metric is expected.
Do not read the simulated genes as biology.
