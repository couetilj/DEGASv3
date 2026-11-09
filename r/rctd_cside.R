#!/usr/bin/env Rscript
# RCTD (doublet mode, replicates) followed by C-SIDE with the DEGAS ensemble score as the
# explanatory variable. Generated and called by `degas3 cside`; see cside/HOW_TO_RUN.md in a run.
#
# Usage: Rscript rctd_cside.R <export_dir> <out_dir> <max_cores> <fdr> <cell_type_threshold>
#
# <export_dir> layout (written by degas3):
#   reference/{matrix.mtx (genes x cells, integer), genes.csv (gene), barcodes.csv (barcode),
#              cell_types.csv (barcode, cell_type)}
#   sections/<section>/{matrix.mtx (genes x spots), genes.csv, coords.csv (barcode,x,y)}
#   explanatory/<section>.csv (barcode, value)   # ensemble score min-max scaled to [0,1] per section
#
# Outputs in <out_dir>: weights/<section>_weights.csv (RCTD weights, rows normalised to sum to 1),
# population_de/<cell_type>_pop_DE.csv (significant genes), population_de/<cell_type>_pop_DE_all.csv
# (all tested genes, when spacexr exposes them), all_cells_DE.csv, and a final line
# "DEGAS3_CSIDE_COMPLETE {json}".
#
# C-SIDE statistics are spacexr's, estimated on spots that mix several cell types; the explanatory
# variable is a model-derived percentile, so results are associations with that score, not effects
# of the outcome.

suppressPackageStartupMessages({
  library(Matrix)
  library(spacexr)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5L) {
  stop("usage: Rscript rctd_cside.R <export_dir> <out_dir> <max_cores> <fdr> <cell_type_threshold>")
}
export_dir <- args[[1]]
out_dir <- args[[2]]
max_cores <- as.integer(args[[3]])
fdr <- as.numeric(args[[4]])
cell_type_threshold <- as.integer(args[[5]])
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

read_mtx <- function(dir, id_col) {
  mat <- as(readMM(file.path(dir, "matrix.mtx")), "CsparseMatrix")
  genes <- read.csv(file.path(dir, "genes.csv"), stringsAsFactors = FALSE)$gene
  stopifnot(nrow(mat) == length(genes))
  rownames(mat) <- genes
  mat
}

# -- reference
ref_dir <- file.path(export_dir, "reference")
ref_mat <- read_mtx(ref_dir)
barcodes <- read.csv(file.path(ref_dir, "barcodes.csv"), stringsAsFactors = FALSE)$barcode
cells_df <- read.csv(file.path(ref_dir, "cell_types.csv"), stringsAsFactors = FALSE)
stopifnot(ncol(ref_mat) == length(barcodes), all(cells_df$barcode == barcodes))
colnames(ref_mat) <- barcodes
cell_types <- factor(cells_df$cell_type)
names(cell_types) <- barcodes
reference <- Reference(counts = ref_mat, cell_types = cell_types, require_int = TRUE)
cat(sprintf("[rctd_cside] reference: %d genes x %d cells, %d cell types\n",
            nrow(ref_mat), ncol(ref_mat), nlevels(cell_types)))

# -- sections
sections <- sort(list.dirs(file.path(export_dir, "sections"), recursive = FALSE, full.names = FALSE))
if (length(sections) == 0L) stop("no sections found under ", file.path(export_dir, "sections"))
replicates <- list()
for (s in sections) {
  sdir <- file.path(export_dir, "sections", s)
  mat <- read_mtx(sdir)
  coords <- read.csv(file.path(sdir, "coords.csv"), stringsAsFactors = FALSE)
  stopifnot(ncol(mat) == nrow(coords))
  colnames(mat) <- coords$barcode
  coord_df <- data.frame(x = coords$x, y = coords$y, row.names = coords$barcode)
  replicates[[s]] <- SpatialRNA(coords = coord_df, counts = mat)
  cat(sprintf("[rctd_cside]   %s: %d spots\n", s, ncol(mat)))
}

# -- RCTD, doublet mode
myRCTD <- create.RCTD.replicates(
  spatialRNA.replicates = replicates,
  reference = reference,
  replicate_names = sections,
  group_ids = rep(1L, length(sections)),
  max_cores = max_cores,
  CELL_MIN_INSTANCE = 25
)
myRCTD <- run.RCTD.replicates(myRCTD, doublet_mode = "doublet")

wdir <- file.path(out_dir, "weights")
dir.create(wdir, showWarnings = FALSE)
for (s in sections) {
  w <- as.matrix(myRCTD@RCTD.reps[[s]]@results$weights)
  rs <- rowSums(w)
  rs[rs == 0] <- 1
  wn <- w / rs
  write.csv(data.frame(barcode = rownames(wn), wn, check.names = FALSE, stringsAsFactors = FALSE),
            file.path(wdir, paste0(s, "_weights.csv")), row.names = FALSE)
}

# -- C-SIDE with the scaled score as explanatory variable
exvar <- list()
for (s in sections) {
  ev <- read.csv(file.path(export_dir, "explanatory", paste0(s, ".csv")), stringsAsFactors = FALSE)
  v <- ev$value
  names(v) <- ev$barcode
  keep <- intersect(names(v), rownames(myRCTD@RCTD.reps[[s]]@results$results_df))
  exvar[[s]] <- v[keep]
  cat(sprintf("[rctd_cside]   %s: %d spots with an explanatory value kept by RCTD\n", s, length(keep)))
}
ct_levels <- levels(myRCTD@reference@cell_types)
myRCTD <- run.CSIDE.replicates(
  myRCTD,
  cell_types = ct_levels,
  explanatory.variable.replicates = exvar,
  cell_type_threshold = cell_type_threshold,
  fdr = fdr
)
myRCTD <- CSIDE.population.inference(myRCTD, fdr = fdr)
saveRDS(myRCTD, file.path(out_dir, "myRCTD_reps_cside_population.rds"))

# -- tables
pdir <- file.path(out_dir, "population_de")
dir.create(pdir, showWarnings = FALSE)
all_de <- list()
for (ct in names(myRCTD@population_sig_gene_df)) {
  df <- myRCTD@population_sig_gene_df[[ct]]
  if (is.null(df) || nrow(df) == 0L) next
  df <- cbind(gene = rownames(df), df, cell_type = ct, stringsAsFactors = FALSE)
  write.csv(df, file.path(pdir, paste0(make.names(ct), "_pop_DE.csv")), row.names = FALSE)
  all_de[[ct]] <- df
}
full <- tryCatch(myRCTD@population_de_results, error = function(e) NULL)
if (!is.null(full)) {
  for (ct in names(full)) {
    df <- full[[ct]]
    if (is.null(df) || !is.data.frame(df) || nrow(df) == 0L) next
    df <- cbind(gene = rownames(df), df, cell_type = ct, stringsAsFactors = FALSE)
    write.csv(df, file.path(pdir, paste0(make.names(ct), "_pop_DE_all.csv")), row.names = FALSE)
  }
}
n_entries <- 0L
if (length(all_de) > 0L) {
  combined <- do.call(rbind, lapply(all_de, function(d) d[, intersect(names(all_de[[1]]), names(d)), drop = FALSE]))
  write.csv(combined, file.path(out_dir, "all_cells_DE.csv"), row.names = FALSE)
  n_entries <- nrow(combined)
} else {
  cat("[rctd_cside] no significant genes at the requested FDR\n")
}
cat(sprintf('DEGAS3_CSIDE_COMPLETE {"sections": %d, "cell_types_tested": %d, "cell_types_with_sig_genes": %d, "n_sig_entries": %d}\n',
            length(sections), length(ct_levels), length(all_de), n_entries))
