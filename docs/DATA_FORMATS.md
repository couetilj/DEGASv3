# Data formats

DEGASv3 accepts three inputs: a bulk count matrix, a phenotype table, and
single-cell or spatial data. Where this page and `docs/SPEC.md` differ, SPEC.md
wins. Run `degas3 init --help` for current flag names.

## Bulk counts

| Item | Requirement |
|---|---|
| Format | CSV or TSV (optionally gzipped) |
| Layout | first column = gene or sample ids. Either orientation works: the loader compares the ids with the phenotype `sample_id` values and transposes when the samples are in the header. At least one id must match. |
| Values | raw integer counts preferred. Non-negative. |
| Log-transformed | allowed if it is log2(x+1). Set the bulk `already_log` flag. Do not mix. |
| Gene ids | symbols or Ensembl ids, consistent with the cell data |
| Duplicates | gene names must be unique. Sum or drop duplicates on a copy. |

## Phenotype table

CSV or TSV, one row per bulk sample.

| Column | Required | Notes |
|---|---|---|
| `sample_id` | yes | matches the bulk matrix sample ids exactly |
| outcome (binary) | yes for binary | 0 = control/reference, 1 = case |
| `time`, `event` | yes for survival | time > 0; event 1 = event, 0 = censored |
| `patient_id` | optional | groups several samples from one person; folds are grouped by it |
| `dataset` | optional | bulk batch or study; enables leave-one-dataset-out and ranking adjustment |

Rows with missing outcome are an error. Remove them on a copy and tell the user.

## Single-cell data (modality `sc`)

| Format | How to give it |
|---|---|
| h5ad | `--cells file.h5ad`. Raw integer counts in `.X` or `.layers["counts"]`. `var_names` = genes. |
| 10x HDF5 or MTX | `--cells path` to the `.h5` file or the directory with `matrix.mtx`, `features.tsv`, `barcodes.tsv`. Donor must come from a separate metadata CSV or be one donor per file. |

Needed `obs` columns:

| Column | Required | Notes |
|---|---|---|
| donor | yes for inference | `--donor-col`. One value per biological replicate. |
| cluster / cell type | optional | `--cluster-col`. If absent, run `degas3 cluster`. |
| clinical label | optional | `--label-col`. Donor-level 0/1. Constant within a donor. |

## Spatial data

| Modality | Source | Notes |
|---|---|---|
| `st_cell` | Xenium outs directory, or CosMx/MERSCOPE exported to h5ad | cell x gene counts plus x,y centroids in `obsm` (the spatial key). Section or acquisition id in `obs`. Gene panels differ by section; DEGASv3 intersects them all. |
| `st_spot` | Visium (Space Ranger `outs`, or h5ad) | spot x gene counts plus spot coordinates. |

Xenium outs: point at the output directory holding `cell_feature_matrix.h5`
and `cells.csv.gz` (columns `cell_id`, `x_centroid`, `y_centroid`). `cells.parquet`
is not read; only "Gene Expression" features are kept. One directory per donor section; provide a donor
and section for each.

## Convert a Seurat object to h5ad

Use the raw counts assay, not normalized data. Export counts, metadata, and
embeddings. Pick one route.

SeuratDisk:

```r
library(Seurat); library(SeuratDisk)
DefaultAssay(obj) <- "RNA"; SaveH5Seurat(obj, "obj.h5Seurat", overwrite = TRUE); Convert("obj.h5Seurat", dest = "h5ad", overwrite = TRUE)
```

zellkonverter:

```r
library(zellkonverter); library(Seurat); writeH5AD(as.SingleCellExperiment(obj, assay = "RNA"), "obj.h5ad")
```

After conversion, open the file in Python and check that `adata.X` (or
`layers["counts"]`) holds integers and that `obs` has the donor column. For
Seurat v5 objects, join layers first (`obj <- JoinLayers(obj)`). If the counts
sit in a layer, say so in the intake.

## Raw counts

SC and ST counts must be raw integers. DEGASv3 checks this and stops
otherwise. Normalized or scaled matrices cannot be repaired. Ask the user for
the raw counts. Bulk raw counts are preferred; log-transformed bulk is accepted
with the flag.

## Gene identifiers

- Bulk and cell genes are intersected by exact match.
- Ensembl version suffixes are stripped before matching (`ENSG00000141510.12` becomes `ENSG00000141510`).
- Case matters for symbols. Do not mix symbols and Ensembl ids; provide a mapping CSV (source id, target id) in the config.
- For ST, genes are also intersected with every section panel.
- `degas3 inspect` prints the shared gene count. If it is far below expectation, check id type first.

## Checklist before `degas3 init`

- Sample ids match between matrix and phenotype.
- Outcome has both classes (or events), with several patients each.
- Counts are non-negative; SC/ST are integer.
- Donor column exists and has at least 2 values.
- Gene id types agree.
