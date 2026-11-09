# DEGASv3 specification (binding for implementation agents)

DEGASv3 is a disease-agnostic, agent-driven workflow that transfers patient-level
outcome information (bulk RNA-seq with a binary or time-to-event outcome) onto
single-cell (SC) or spatial transcriptomic (ST) data using the DEGAS deep
transfer-learning model, then asks what distinguishes high-score from low-score
cells **within each cluster**.

Two deliverables live in one repository:

1. `degas3`: a deterministic Python package and CLI. Every stage reads and writes a
   run directory, is idempotent (skips when `complete.json` exists), and records
   provenance.
2. The **agent playbook** (`AGENTS.md`, `CLAUDE.md`, `agent/*.md`,
   `.claude/skills/degas3/SKILL.md`): instructions that let a coding agent
   (Claude Code, Codex, or similar) interview the user, verify inputs and GPU
   access, choose options, run stages, and report honestly.

Nothing in the package or docs may reference a specific disease, tissue, or
cell type except inside the simulator and clearly marked examples. Use the
generic vocabulary below.

## Vocabulary

| Term | Meaning |
|---|---|
| patient / sample | one bulk RNA-seq profile (`sample_id`); `patient_id` groups multiple samples from one person |
| outcome | `binary` (0 = control/reference, 1 = case) or `survival` (time + event) |
| dataset | optional bulk batch/study column; enables leave-one-dataset-out |
| cell | one SC cell, ST cell (Xenium/CosMx/MERSCOPE), or ST spot (Visium) |
| donor | SC/ST biological replicate (`donor` column); the inference unit for DE |
| cluster | SC/ST grouping used for stratification (supplied label or computed Leiden cluster) |
| score | DEGAS output: raw model output in [0,1] (binary) or hazard (survival); **ensemble score** is the rank-based ensemble in [0,1] |
| high / low | cells above / below the within-cluster cutoff of the ensemble score |

## Package layout and file ownership

```
DEGASv3/
  pyproject.toml            # name degas3, python>=3.10, console script `degas3`
  README.md  LICENSE (MIT)  CITATION.cff  AGENTS.md  CLAUDE.md
  agent/                    # playbook: INTAKE.md STAGES.md REPORTING.md GUARDRAILS.md
  .claude/skills/degas3/SKILL.md
  docs/                     # SPEC.md (this), METHODS.md, TUTORIAL.md, DATA_FORMATS.md, SLURM.md
  degas3/
    __init__.py  __main__.py  cli.py            # argparse CLI dispatching to stages
    contracts.py              # dataclasses + run-directory layout (see below)
    config.py                 # RunConfig: load/validate/merge JSON, defaults
    runs.py                   # RunDir, complete.json, manifest, hashing, versions
    io/  (loaders)            # bulk.py sc.py spatial.py (h5ad, csv/tsv, 10x h5/mtx, Xenium outs, Visium)
    core/                     # vendored DEGAS model code (from DEGASv2 DEGAS_python), cleaned
      __init__.py options.py preprocess.py models.py losses.py sampling.py
    features.py               # fold-internal gene ranking: variance (default), mixed
    folds.py                  # patient-grouped k-fold, leave-one-dataset-out, donor exclusion
    train.py                  # fit one model; predict; checkpoint save/load (CPU-loadable)
    validate.py               # CV driver, metrics (AUROC, AP, log loss, Brier, C-index), one-SE selection
    score.py                  # final ensemble fits + rank-based ensemble score
    cluster.py                # scanpy: normalize, HVG, PCA, neighbors, Leiden, UMAP
    stratify.py               # within-cluster median/quantile split; ties; coverage tables
    de.py                     # pseudobulk + PyDESeq2 (paired), cell-level Wilcoxon fallback, clinical contrast, edgeR export
    pathways.py               # gseapy prerank (MSigDB Hallmark default) + hypergeometric ORA
    explain.py                # gradient SHAP vs low-quartile same-cluster reference
    spatial/                  # neighborhoods.py (Delaunay graph radius pooling), maps.py, cside.py (RCTD->CSIDE export/run/parse)
    report.py                 # self-contained HTML + METHODS.md + manifest
    doctor.py                 # environment checks (torch, CUDA, GPU memory, R, edgeR, spacexr)
    simulate.py               # toy bulk + SC + ST (single-cell spatial + Visium-like) generator
    slurm.py                  # sbatch script templates
    plotting/                 # style.py (palette, square panels), figures per stage
  r/                          # R script templates: edger_pseudobulk.R, rctd_cside.R (run only if R present)
  tests/                      # pytest, CPU only, tiny data from simulate
```

## Shared contracts (`degas3/contracts.py`)

```python
@dataclass
class BulkData:
    counts: np.ndarray          # samples x genes, raw counts (float64 ok); log-transformed input flagged by config
    genes: list[str]
    sample_ids: list[str]
    pheno: pd.DataFrame         # index = sample_ids; columns: outcome (0/1) or time,event; optional patient_id, dataset
    outcome_type: Literal["binary", "survival"]
    already_log: bool = False

@dataclass
class CellData:                 # thin wrapper over AnnData
    adata: anndata.AnnData      # .X or .layers["counts"] raw counts (cells x genes), var_names = genes
    donor_col: str | None
    cluster_col: str | None
    label_col: str | None       # optional per-donor clinical outcome label (0/1) for clinical-vs-score comparison
    modality: Literal["sc", "st_cell", "st_spot"]
    spatial_key: str | None     # obsm key with x,y coordinates for ST
    sample_col: str | None      # acquisition / section id for ST
```

Run directory layout (all stages use `RunDir(root)` helpers; every stage writes
`<stage>/complete.json` with timestamps, inputs hashes, parameters, package
version, git commit if available, SLURM job id if present):

```
<run>/config.json
<run>/inputs/            bulk.npz  cells.h5ad  input_manifest.json (paths + sha256 + shapes)
<run>/prepared/          shared_genes.txt  folds.json  rankings/<fold>.csv  tasks.json
<run>/cv/task_###/seed_#/   checkpoint.pt losses.csv heldout_predictions.csv complete.json
<run>/cv/metrics.csv  cv/selection.json  cv/selection.png
<run>/models/size_####/seed_#/  checkpoint.pt losses.csv complete.json
<run>/scores/            cell_scores.csv.gz (cell_id, donor, cluster, raw_size_####, ensemble_score)  patient_scores.csv
<run>/strata/            cutoffs.csv  cell_groups.csv.gz  coverage.csv
<run>/de/<cluster>/      method.json  results.csv  pseudobulk_counts.mtx samples.csv  (clinical/ subdir when label_col)
<run>/pathways/<cluster>/ high_vs_low_gsea.csv  ora_high.csv ora_low.csv
<run>/explain/<cluster>/ shap_values.npz  gene_attributions.csv  figures
<run>/spatial/<sample>/  neighborhoods.csv  pooled_counts.npz  pooled_scores.csv  maps/
<run>/cside/             export/  rctd/  population_de/  (only when requested and R available)
<run>/report/            index.html  METHODS.md  figures/  manifest.json
```

Checkpoints must be loadable without CUDA (`torch.load(..., map_location="cpu")`)
and contain: state dicts for feature extractor + outcome head, `genes` (ordered),
`opt`, `outcome_type`, `model_type`, `seed`, `size`, `fold`.

## Scientific methods (defaults; every default is a config key)

### Inputs and gene universe
- Intersect bulk genes with SC/ST genes (and with every ST sample panel when
  modality is ST). Report counts. Gene identifiers are matched exactly after
  stripping version suffixes (`ENSG...\.\d+`); optional user-provided mapping CSV.
- Raw counts required for SC/ST (integers; check) and preferred for bulk.
  Bulk `already_log` flag (log2(x+1) inputs) is passed to `preprocess_counts`.
- Preprocessing is DEGAS `preprocess_counts`: `normalize_scale(1.5 ** log2(x+1))`
  per sample (z-score then min-max across the selected genes). Applied **after**
  gene selection, separately for bulk and cells.

### Model
- Vendored DEGAS architecture: 3-layer sigmoid feature extractor (feature_dim 25,
  dropout 0.5), Wasserstein discriminator with gradient penalty (lambda3 3.0,
  gp 10, n_critic 2), outcome head (softmax 2-class for binary, sigmoid hazard
  with negative-log partial-likelihood Cox loss for survival), optional cell-label
  head (ClassClass / ClassCox, lambda1 1.0). Adam lr 0.01, 300 updates, class-
  balanced batches of 200 patients and 200 cells.
- `model: auto` → Class* when `cluster_col` exists, Blank* otherwise. `--model blank|class` overrides.
- Training requires a CUDA device unless `--allow-cpu` (warn: slow; identical math).
- Seeding: python, numpy, torch, cudnn deterministic. No global seeding at import time.
- Predict for cells uses feature extractor → outcome head only (no discriminator).

### Fold-internal feature ranking
- `variance` (default): log-CPM on training bulk samples; subtract dataset means
  if `dataset` exists (ranking only); rank genes by residual variance, then mean.
- `mixed`: alternate one bulk-variance gene and one SC cluster marker (one-vs-rest
  tie-corrected Wilcoxon on log-normalized counts, positive log2FC, BH in log
  space); marker slots rotate across clusters, least-represented cluster first;
  up to `marker_cells_per_cluster` (500) cells per cluster per donor-balanced
  sample. Requires cluster labels. Never pad silently; error when exhausted.
- Rankings use **only** the training fold's bulk samples (and, for `mixed`, only
  SC cells not excluded for that fold). Nested lists: smaller sizes are prefixes.

### Validation and size selection
- Candidate sizes default `[10,20,40,60,120,240,500,1000,2500,5000,10000]` clipped
  to the shared gene count (and the full shared universe appended when smaller than
  the max).
- `cv: patient_kfold` (default): 5 folds grouped by `patient_id` (falls back to
  `sample_id`), stratified by outcome (binary) or event (survival), seed 42.
  `cv: leave_one_dataset_out` when `dataset` has ≥ 2 levels and the agent/user chooses it.
- Seeds per CV fit: 3 (`cv_seeds`). Final seeds: 5 (`final_seeds`).
- Donor leakage: if SC/ST `donor` values match bulk `patient_id` values, exclude
  those cells from that fold's cell set; report matches. Config `paired_datasets`
  maps bulk dataset → SC dataset to exclude.
- Metrics per held-out fold/dataset: binary AUROC (primary), average precision,
  log loss, Brier; survival Harrell C-index (primary). Averaged across folds
  with equal weight.
- One-SE rule: 2,000 patient bootstraps (stratified by outcome within fold/dataset),
  fixed models; retain every size whose mean metric ≥ best − SE(best). Record
  table and figure. State in outputs that performance after selection is optimistic.

### Scoring
- Final models: all bulk samples, all eligible cells, each retained size × 5 seeds.
- Raw score per cell per size = mean of seeds. Percentile = average-tie rank across
  all scored cells / N (default `rank_scope: global`; `rank_scope: sample` ranks within
  `sample_col`/`donor`). Ensemble score = equal mean of per-size percentiles.
  Never re-rank the mean. Also keep raw per-size scores and the raw mean.
- Patient scores: same ensemble of raw outputs for bulk samples (in-sample, flagged)
  plus the out-of-fold CV predictions.

### Clustering (when no cluster label)
- scanpy: `normalize_total(1e4)` → `log1p` → HVG 2000 (seurat flavor, batch_key =
  donor when ≥ 2 donors) → scale (clip 10) → PCA 50 → neighbors 15 → Leiden
  (resolution 1.0, igraph flavor) → UMAP. Writes labels into `obs["degas3_cluster"]`,
  marker table (Wilcoxon top 20 per cluster), and UMAP coordinates. Never uses
  outcome or scores. If UMAP exists in `obsm["X_umap"]` reuse it for plots.

### Stratification
- Within each cluster, cutoff on the ensemble score: `split: median` (default) or
  `split: quantiles [0.25, 0.75]`. Cells equal to a cutoff are `tie` and excluded.
  Output per cluster × donor: n_high, n_low, n_tie, fractions.

### Differential expression (high vs low)
- `pseudobulk_deseq2` (default when ≥ 2 donors have both groups with ≥ `min_cells`
  (default 10) cells): sum raw counts per donor × group; PyDESeq2 with design
  `~ donor + group`; results: log2FC, stat, pvalue, BH padj; filter genes with
  ≥ 10 total counts. Record donors used and skips.
- `wilcoxon` (when replication is absent, or `--de-method wilcoxon`): scanpy
  `rank_genes_groups(method="wilcoxon", tie_correct=True)` high vs low on
  log-normalized counts within the cluster; report log2FC, AUC, p, BH padj.
  Call it "Cell-level Wilcoxon" in outputs (no extra qualifiers).
- Clinical contrast (only when `label_col` exists and ≥ 2 donors per label):
  pseudobulk per donor × cluster, PyDESeq2 `~ dataset + label` (dataset only if
  ≥ 2 levels); scatter of clinical vs high/low log2FC on shared genes with
  model-input genes flagged.
- edgeR path: always write pseudobulk `counts.mtx` + `samples.csv`; when
  `--de-engine edger` and `Rscript` + edgeR exist, run `r/edger_pseudobulk.R`
  (filterByExpr, TMM, QL F-test, same designs) and parse results.

### Pathways
- gseapy `prerank` on ranking = stat (DESeq2 Wald stat or signed −log10 p for
  Wilcoxon), MSigDB Hallmark (`msigdb_collection: h.all`, `species: human|mouse`,
  version pinned in config, downloaded via gseapy and cached; or user GMT path),
  min 15 / max 500, 1000 permutations, seed 42. Also hypergeometric ORA for
  padj < 0.05 up/down lists against the tested-gene background.

### Explanations (optional stage)
- `shap.GradientExplainer` on each final model's preprocessed inputs; reference =
  donor-balanced sample (≤ 100) of same-cluster cells below the cluster's 25th
  percentile; targets = ≤ 1 cell per donor × quartile (Q2–Q4). Attribute the mean
  raw score across retained models (not the percentile). Report signed and
  absolute means, pooled and equal-donor, with 2,000 donor bootstraps; verify
  additivity residuals. Label outputs "model attribution relative to low-score
  same-cluster cells".

### Spatial
- `st_spot` (Visium): spots are scored directly. Optional `degas3 cside`: export
  SC reference (cluster labels, ≤ 1000 cells/cluster) and each section to MTX/CSV,
  write `r/rctd_cside.R` (spacexr `Reference`, `SpatialRNA`, `create.RCTD.replicates`
  doublet mode, `run.CSIDE.replicates` with the per-section 0–1 min-max scaled
  ensemble score as explanatory variable, `CSIDE.population.inference`), run when
  Rscript + spacexr exist, parse `population_de` tables, volcano per cell type.
- `st_cell` (Xenium/CosMx/MERSCOPE): cells scored directly. Optional
  `degas3 neighborhoods`: per sample, Delaunay triangulation on centroids, drop
  edges above the 99th percentile length, geodesic (shortest-path) radii
  calibrated so the median neighborhood holds `targets` [5,10,20,40] cells
  (1,024 calibration seeds, 1,024 evaluation seeds); pool raw counts of all cells
  within radius around `n_seeds` (2,048) sampled seed cells; score pools with the
  final models (same ensemble convention, ranked within sample); maps of seeds
  colored by score; cell-type fractions per pool.
- Maps: cells/spots colored by ensemble score (blue–gray–red, 0–1, midpoint 0.5)
  and by cluster.

### Report
- `report/index.html`: self-contained (base64 PNGs), sections per completed
  stage, every table of counts (patients, donors, cells per cluster, coverage),
  CV curve + selection, score UMAP/violins, high/low DE tables, pathways, SHAP,
  spatial maps, and a limitations section generated from what actually ran.
- `report/METHODS.md`: parameter-filled methods text with citations.
- `manifest.json`: input hashes, config hash, package version, git commit,
  python/torch/CUDA versions, GPU name, seeds, timings, SLURM ids.

## Plot style (`degas3/plotting/style.py`)
- Square panels 6.4 × 6.4 in, 300 dpi PNG + PDF; `bbox_inches=None`; DejaVu Sans;
  title 12 pt, labels 10 pt, ticks 9 pt; no top/right spines.
- Score colormap: blue `#2166AC` → `gray90 (#E5E5E5)` → red `#B2182B`, limits 0–1,
  midpoint 0.5, label "Ensemble DEGAS percentile".
- Categorical: Okabe–Ito palette, cycling with lighter variants beyond 8.
- Case/control: `#D55E00` / `#0072B2`. Donor points over violins; inference on donors.

## CLI

```
degas3 doctor [--run RUN]
degas3 simulate OUT_DIR [--seed 0] [--n-patients 120 ...]
degas3 init RUN --bulk X --pheno Y --cells Z [--outcome binary|survival] [--donor-col] [--cluster-col] [--label-col] [--modality] [--dataset-col] ...
degas3 inspect RUN                   # writes inputs/input_manifest.json and prints the intake summary
degas3 cluster RUN [--resolution 1.0]
degas3 prepare RUN                   # gene universe, folds, rankings, tasks
degas3 train RUN [--task N | --all] [--allow-cpu]      # CV tasks
degas3 select RUN                    # metrics + one-SE
degas3 fit RUN [--all]               # final models
degas3 score RUN
degas3 stratify RUN [--split median|quantiles --quantiles 0.25 0.75]
degas3 de RUN [--method auto|pseudobulk_deseq2|wilcoxon] [--engine pydeseq2|edger]
degas3 pathways RUN [--gmt PATH | --collection h.all --species human]
degas3 explain RUN
degas3 neighborhoods RUN ; degas3 maps RUN ; degas3 cside RUN
degas3 report RUN
degas3 slurm RUN --account A --partition P [--gpus 1] > jobs.sh
degas3 run-all RUN                   # the default chain for the minimal scenario
```

## Quality bar
- Every stage: `complete.json`, idempotent, clear `ValueError` with remedy text.
- Tests on CPU with `simulate` output (small sizes, 20 updates) covering: loaders,
  preprocess parity with the reference formula, folds (no patient leakage),
  rankings fold-internal, one-SE selection, rank ensemble, stratify ties, DE both
  methods, pathways on a toy GMT, neighborhoods geometry, report build.
- No hard-coded paths. No disease-specific names outside `simulate` and examples.
- Docstrings state what each statistic is and is not (e.g. percentile ≠ probability).

## Foundation interfaces (implemented in Wave 1; build on these, do not re-implement)

- `degas3.config.RunConfig.load(run)` → config with all defaults; `update_config_file(run, **kv)`.
- `degas3.runs.RunDir(run)`: `.bulk_npz`, `.cells_h5ad`, `.input_manifest`, `.stage_dir(stage)`,
  `.cv_task_dir(task, seed)`, `.model_dir(size, seed)`, `.complete_path(stage, *subdir)`,
  `.stage_complete(stage, *subdir)`, `.write_complete(stage, payload, *subdir)`,
  `environment()`, `sha256_file/sha256_path/sha256_json`.
- `degas3.io.bulk.load_bulk_npz(path)` → `BulkData`; `degas3.io.sc.get_counts(adata)`;
  `degas3.io.sc.intersect_genes(bulk, cells, extra_panels=None, out_path=None)` → `GeneIntersection(genes, counts)`;
  `degas3.io.bulk.strip_ensembl_version`.
- `degas3.core.options.make_options(outcome_type, with_cell_labels, **overrides)` (set
  `input_shape`, `high_reso_output_shape`, `seed`, `tot_iters`, `batch_size`, `pat_batch_size`, `feature_dim`).
- `degas3.core.models`: `resolve_device("auto", allow_cpu)`, `build_model(opt, device)`,
  `fit(model, bulk_X, bulk_y, cell_X, cell_y=None, iterations=None, seed=None)` → loss DataFrame
  (caller writes `losses.csv`), `predict_scores(model, X)` → np.ndarray in [0,1],
  `save_checkpoint(model, path, genes=, seed=, size=, fold=, extra=)`, `load_checkpoint(path, device)` → (model, ckpt).
  `bulk_y`: 1-D 0/1 for binary; (n,2) time,event for survival. `cell_y`: integer cluster codes.
  Inputs must already be `preprocess_counts`-transformed over the SAME ordered gene list.
- `degas3.core.preprocess.preprocess_counts(X, already_log=False)`.
- `degas3.plotting.style`: `apply_style()`, `square_panel(title, xlabel, ylabel)` → (fig, ax),
  `colorbar(fig, ax, mappable, label)`, `save_panel(fig, ax, stem)`, `score_cmap()`, `score_norm()`,
  `categorical_palette(n)`, constants `CASE`, `CONTROL`.
- CLI: a stage module exposes `cli_<stage>(args: argparse.Namespace) -> int`; `args.run` is the run
  directory; see `STAGE_MODULES` in `degas3/cli.py` for module names and `build_parser` for flags.
  `run-all` chains: cluster (if no cluster_col) → prepare → train --all → select → fit --all → score →
  stratify → de → pathways → report.
- Config keys of note: `sizes, model, feature_dim, feature_selection, marker_cells_per_cluster,
  iterations, batch_size, allow_cpu, cv, folds, cv_seeds, final_seeds, selection_seed, bootstraps,
  paired_datasets, rank_scope, split, quantiles, de_method, de_engine, min_cells, msigdb_collection,
  msigdb_version, species, gmt_path, pathway_*, explain_*, neighborhood_targets, n_seeds,
  calibration_seeds, evaluation_seeds, donor_col, cluster_col, label_col, sample_col, dataset_col,
  patient_col, spatial_key, outcome, modality, already_log`.
- Simulated files: `bulk_counts.csv` (genes x samples), `bulk_pheno.csv` (sample_id, patient_id,
  dataset, outcome, time, event), `cells.h5ad` (obs donor, cell_type, label), `spatial_cells.h5ad`
  (obs sample, obsm spatial), `visium_spots.h5ad`, `simulate_manifest.json`.
