---
name: degas3
description: Use for any request to run DEGAS, or to transfer patient outcomes (bulk RNA-seq with a binary or survival outcome) to single-cell or spatial transcriptomic data and compare high-score versus low-score cells within clusters.
---

# DEGASv3 skill

1. Read `AGENTS.md` in the repository root. Follow it.
2. Run the intake interview in [agent/INTAKE.md](../../../agent/INTAKE.md). Check inputs against [docs/DATA_FORMATS.md](../../../docs/DATA_FORMATS.md). Confirm a GPU with `degas3 doctor` or plan SLURM ([docs/SLURM.md](../../../docs/SLURM.md)).
3. Run the stages in order using [agent/STAGES.md](../../../agent/STAGES.md). Verify `complete.json`, counts, and warnings after each stage. Surface the decision points to the user.
4. Apply [agent/GUARDRAILS.md](../../../agent/GUARDRAILS.md) throughout.
5. Run `degas3 report RUN` and summarize with [agent/REPORTING.md](../../../agent/REPORTING.md).

Do not edit package code to fit a dataset. Adapt copies of the inputs or ask the user.
