# Implementation plan (waves)

Decisions confirmed with the user on 2026-11-09: vendor a clean copy of the
DEGASv2 Python backend; agent playbook + deterministic CLI; Class* models when
labels exist; global percentile ensemble; PyDESeq2 pseudobulk default with edgeR
optional; cell-level Wilcoxon when replication is absent; public MIT repo with
DEGAS citations; Cox outcomes included; 5 patient-grouped folds × 3 seeds, sizes
10–10000, one-SE; leave-one-dataset-out option; SHAP; neighborhood pooling for
single-cell ST only; RCTD→CSIDE for Visium; simulated example; GPU smoke test on
Quartz before pushing.

## Wave 1 — foundation (one agent)
pyproject, contracts, config, runs, io loaders, vendored core, doctor, simulate,
plotting style, cluster, CLI skeleton with all subcommands registered, tests for
loaders/simulate/preprocess.

## Wave 2 — parallel modules (agents own disjoint files)
A. features.py, folds.py, train.py, validate.py, score.py (+ tests)
B. stratify.py, de.py, pathways.py, r/edger_pseudobulk.R (+ tests)
C. spatial/ (neighborhoods, maps, cside), r/rctd_cside.R (+ tests)
D. explain.py, report.py, slurm.py (+ tests)
E. docs + playbook: README, AGENTS.md, CLAUDE.md, agent/*.md, skill, docs/*.md, CITATION.cff

## Wave 3 — integration
run-all on simulated data (CPU locally, GPU on Quartz), fix defects found in
review, standardize figures and directory outputs, final review, push.
