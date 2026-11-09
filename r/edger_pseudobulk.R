#!/usr/bin/env Rscript
# edgeR quasi-likelihood F-test on donor pseudobulk counts.
#
# Usage: Rscript edger_pseudobulk.R <counts.mtx> <samples.csv> <paired|clinical> <out.csv>
#
#   counts.mtx  genes x samples MatrixMarket matrix of raw counts; gene names are read from
#               genes.txt in the same directory (row order), else rows are named by index.
#   samples.csv one row per column of counts.mtx.
#               paired:   columns donor, group (levels low, high)         design ~ donor + group
#               clinical: columns label and optionally dataset            design ~ dataset + label
#                         (dataset used only when it has >= 2 levels)     or     ~ label
#   out.csv     columns gene, logFC, logCPM, F, PValue, FDR for the LAST design coefficient
#               (group high vs low; or the second label level vs the first, levels sorted).
#
# Donors are the replicates. filterByExpr(design), TMM, estimateDisp(robust), glmQLFit(robust).
# Prints one JSON line on completion.
suppressPackageStartupMessages({library(Matrix); library(edgeR)})
a <- commandArgs(TRUE)
if (length(a) != 4) stop("usage: edger_pseudobulk.R counts.mtx samples.csv paired|clinical out.csv")
mtx <- a[1]; samples_csv <- a[2]; mode <- a[3]; out_csv <- a[4]
if (!mode %in% c("paired", "clinical")) stop("design must be 'paired' or 'clinical'")

counts <- as.matrix(readMM(mtx))
genes_file <- file.path(dirname(mtx), "genes.txt")
genes <- if (file.exists(genes_file)) readLines(genes_file) else paste0("gene", seq_len(nrow(counts)))
if (length(genes) != nrow(counts)) stop("genes.txt length does not match counts.mtx rows")
rownames(counts) <- genes
m <- read.csv(samples_csv, stringsAsFactors = FALSE, colClasses = "character")
if (ncol(counts) != nrow(m)) stop("samples.csv rows do not match counts.mtx columns")

if (mode == "paired") {
  m$donor <- factor(m$donor)
  m$group <- factor(m$group, levels = c("low", "high"))
  design <- model.matrix(~ donor + group, m)
} else {
  m$label <- factor(m$label, levels = sort(unique(m$label)))
  if ("dataset" %in% names(m) && length(unique(m$dataset)) > 1) {
    m$dataset <- factor(m$dataset)
    design <- model.matrix(~ dataset + label, m)
  } else {
    design <- model.matrix(~ label, m)
  }
}
if (qr(design)$rank < ncol(design) || nrow(design) <= ncol(design)) stop("design is not estimable")
coef <- ncol(design)

y <- DGEList(counts = counts)
keep <- filterByExpr(y, design = design)
if (sum(keep) <= 10) stop("too few genes pass filterByExpr")
y <- calcNormFactors(y[keep, , keep.lib.sizes = FALSE], method = "TMM")
y <- estimateDisp(y, design, robust = TRUE)
fit <- glmQLFit(y, design, robust = TRUE)
test <- glmQLFTest(fit, coef = coef)
tab <- topTags(test, n = Inf, sort.by = "none")$table
out <- data.frame(gene = rownames(tab), logFC = tab$logFC, logCPM = tab$logCPM, F = tab$F,
                  PValue = tab$PValue, FDR = tab$FDR)
write.csv(out, out_csv, row.names = FALSE)
cat(sprintf('{"status":"complete","mode":"%s","samples":%d,"genes_tested":%d,"coefficient":"%s","edgeR":"%s"}\n',
            mode, nrow(m), nrow(out), colnames(design)[coef], as.character(packageVersion("edgeR"))))
