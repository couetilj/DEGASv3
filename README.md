# DEGASv3

DEGASv3 moves patient outcomes from bulk RNA-seq onto single cells or spatial
spots. It then asks what differs between high-score and low-score cells inside
each cluster.

In short: **patient bulk outcome -> per-cell or per-spot DEGAS score -> within-cluster high-vs-low comparison.**

DEGAS (Diagnostic Evidence GAuge of Single cells) is a deep transfer-learning
model. It learns from bulk samples with a known outcome (case/control, or
time-to-event) and scores each cell by how much its expression resembles the
high-risk side of that outcome. DEGASv3 wraps the model in a checked, resumable
workflow with patient-level validation, clustering, differential expression,
pathways, optional explanations, and a report.

## Who it is for

- Biologists with a bulk cohort that has an outcome and a single-cell or spatial dataset from the same tissue type.
- Analysts who want a reproducible run directory with provenance for every stage.
- People who want a coding agent to do the work while they answer questions.

## Two ways to use it

1. **Drive it with a coding agent.** Open this repository in Claude Code, Codex, or a similar agent. The agent reads `AGENTS.md`, interviews you (`agent/INTAKE.md`), runs the stages, and writes the report. It asks before every scientific choice.
2. **Run the CLI yourself.** Follow `docs/TUTORIAL.md`. Each stage is one `degas3` command.

## Install

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
degas3 doctor
```

GPU: training needs a CUDA GPU. Install a PyTorch build that matches your CUDA
driver. `degas3 doctor` reports torch, CUDA, GPU memory, and whether R, edgeR,
and spacexr are present (R is optional). A CPU run is possible with `--allow-cpu`
for small tests only. It is slow and never chosen silently. On a cluster, see
`docs/SLURM.md`.

## Quickstart on simulated data

```bash
degas3 simulate demo_data --seed 0
degas3 init runs/demo --bulk demo_data/bulk_counts.csv --pheno demo_data/bulk_pheno.csv \
    --cells demo_data/cells.h5ad --outcome binary --donor-col donor \
    --cluster-col cell_type --label-col label
degas3 inspect runs/demo
degas3 run-all runs/demo --gmt tests/fixtures/toy.gmt --allow-cpu   # small test only; use a GPU for real data
degas3 report runs/demo
# open runs/demo/report/index.html
```

The simulated genes are synthetic, so they do not overlap MSigDB Hallmark;
`--gmt tests/fixtures/toy.gmt` supplies a matching toy gene-set file (without it,
pathways records `no_eligible_gene_sets` and the chain continues; `--skip-pathways` omits the stage).

`simulate` writes `bulk_counts.csv`, `bulk_pheno.csv`, `cells.h5ad`,
`spatial_cells.h5ad`, `visium_spots.h5ad`, and `simulate_manifest.json` (the
planted signal, for checking results).

## Stages

| # | Command | What it does | Main output |
|---|---|---|---|
| 0 | `degas3 doctor` | Check Python, torch, CUDA, optional R | console summary |
| 1 | `degas3 init`, `degas3 inspect` | Record inputs, hash them, print the intake summary | `config.json`, `inputs/` |
| 2 | `degas3 cluster` | Optional PCA, neighbors, Leiden, UMAP (outcome-blind) | cluster labels, markers |
| 3 | `degas3 prepare` | Shared genes, patient folds, fold-internal gene rankings | `prepared/` |
| 4 | `degas3 train` | Cross-validation fits | `cv/task_###/` |
| 5 | `degas3 select` | Held-out metrics and one-standard-error size choice | `cv/metrics.csv`, `cv/selection.json` |
| 6 | `degas3 fit` | Final models on all patients | `models/` |
| 7 | `degas3 score` | Rank-based ensemble score per cell | `scores/` |
| 8 | `degas3 stratify` | High/low groups within each cluster | `strata/` |
| 9 | `degas3 de` | High vs low differential expression | `de/<cluster>/` |
| 10 | `degas3 pathways` | Gene-set enrichment | `pathways/<cluster>/` |
| 11 | `degas3 explain` (optional) | Model attribution with SHAP | `explain/` |
| 12 | `degas3 neighborhoods`, `maps`, `cside` (optional) | Spatial pooling, maps, cell-type DE | `spatial/`, `cside/` |
| 13 | `degas3 report` | Self-contained HTML report | `report/` |

`degas3 run-all` runs the default chain (stages 1 to 10 and the report). Every
stage writes `complete.json` and skips if it already exists.

## Input requirements (summary)

| Input | Needs |
|---|---|
| Bulk counts | samples x genes, raw counts preferred; flag log-transformed data |
| Phenotype | `sample_id`; outcome (0/1) or `time` + `event`; optional `patient_id`, `dataset` |
| Cells | raw counts as h5ad, 10x, Xenium outs, or Visium; a donor column; optional cluster and label columns |

Details and conversion recipes are in `docs/DATA_FORMATS.md`. Use at least two
donors for any donor-level inference. The patient, not the cell, is the unit for
outcome claims.

## What DEGASv3 does not do

- It does not turn scores into probabilities. Scores are percentile ranks.
- It does not prove that a gene drives the outcome. Genes the model uses are model attribution, not independent discovery.
- It does not make cells independent replicates. Donors are the unit for cell-level comparisons.

See `docs/METHODS.md` and `agent/GUARDRAILS.md`.

## Documentation

| File | Content |
|---|---|
| `AGENTS.md` | Operating rules for a coding agent |
| `agent/` | Intake script, stage reference, reporting, guardrails |
| `docs/DATA_FORMATS.md` | Accepted inputs |
| `docs/METHODS.md` | Methods and their limits |
| `docs/TUTORIAL.md` | Walk-through on simulated data |
| `docs/SLURM.md` | Cluster use |
| `docs/SPEC.md` | Binding specification |

## Citation

Cite the software (`CITATION.cff`) and the DEGAS paper:

Johnson TS, Yu CY, Huang Z, et al. Diagnostic Evidence GAuge of Single cells
(DEGAS): a flexible deep transfer learning framework for prioritizing cells in
relation to disease. Genome Medicine. 2022;14:11. doi:10.1186/s13073-022-01012-2

Preprint: doi:10.1101/2020.06.16.142984. DEGASv2 preprint (Wasserstein
alignment): Liu Z, Li S, Couetil J, Zhang J, Huang K, Johnson TS.
doi:10.1101/2025.02.03.636036.

## Credits

The model code in `degas3/core` is a cleaned port of the DEGASv2 Python package
(https://github.com/ElliotLiu1997/DEGASv2), developed by Ziyu Liu, Travis S.
Johnson, Sihong Li, and Jiahui Liu. The DEGASv2 method preprint is by Liu, Li,
Couetil, Zhang, Huang, and Johnson. The original DEGAS was created by Travis S.
Johnson and colleagues (https://github.com/tsteelejohnson91/DEGAS). The workflow
conventions in DEGASv3 (fold-internal feature ranking, one-standard-error size
ensembles, rank-based scores, within-cluster stratification, donor-aware DE) come
from applied DEGAS studies by the DEGAS team.

## License

MIT. See `LICENSE`.
