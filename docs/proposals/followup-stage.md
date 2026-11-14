# Proposal: a `followup` stage for narrowing DEGAS candidates

Status: **draft for discussion**. No package code changes in this pull request. `docs/SPEC.md`
is binding, so the spec would be amended first once the open decisions below are settled.

## Why

`degas3 de` produces, per cluster, a high-versus-low score contrast and (when donor labels exist)
a clinical contrast. Reading the two side by side leaves several explanations open for any gene:

- the high-score state is simply more common in cases (a frequency shift);
- every case cell differs, inside any score group (a within-state shift);
- the gene marks a state that cases and controls share;
- the effect comes from shallow cells, ambient RNA or mixed-identity cells;
- one study or platform carries it;
- the outcome label is standing in for a donor covariate such as age;
- the gene is a model input, so the association is partly attribution.

These predict different validation readouts (for example, fraction of positive cells versus
intensity in every cell), so they should be separated before a candidate is chosen. The stage
proposed here runs the checks that separate them, cheapest first, and reports the evidence per gene.

## Proposed commands

A stage after `de`, with steps that can be run separately. Each writes
`<run>/followup/<step>/complete.json` and skips when complete.

```
degas3 followup RUN --step donors       # per-donor paired effects, sign fraction, per-study means,
                                        # leave-one-study-out, donor bootstrap, state fraction by outcome
degas3 followup RUN --step strata       # outcome contrast within score strata; adjusted for state fraction
degas3 followup RUN --step matched      # paired contrast on cells matched within donor for depth
                                        # and non-self transcript fraction
degas3 followup RUN --step covariates --donor-table donors.csv --covariates age,bmi
degas3 followup RUN --step explain      # joins the above into one table per cluster
```

Optional, separate commands because they need other data or a GPU:

```
degas3 followup-retrain RUN NEWRUN --exclude genes.txt     # refit without a gene program (new run directory)
degas3 followup-external RUN --bulk X --groups Y --order A,B,C   # ordered outcome groups in an independent cohort
degas3 followup-marker RUN --marker G --identity I --rivals J,K  # panel marker inside a population defined by other genes
```

## How it maps onto the current package

| Need | Today | Gap |
|---|---|---|
| Score groups | `strata/cell_groups.csv.gz`, `coverage.csv`; default split is median | within-strata analysis has a middle group only with `--split quantiles`; with a median split, two strata are used and reported as such |
| Donor × group pseudobulk | `de/<cluster>/pseudobulk_counts.mtx`, `samples.csv`, `clinical/` | none; reuse |
| Minimum cells per donor and group | 10 in pseudobulk DE | reuse the same threshold |
| DE engine | PyDESeq2 default, edgeR optional (`de._fit`) | stratified and adjusted contrasts call `_fit` with a different design |
| Model-input flags | `de.model_input_genes` | reuse |
| Refit without genes | `prepare` → `train` → `select` → `fit` → `score` | an exclusion list applied at ranking time; always a new run directory |
| Spatial | `spatial/neighborhoods.py`, `maps.py` | marker-in-population and structure-level heterogeneity are new |
| Report | `report.py` | a follow-up section with its own limitations |

Suggested layout: `degas3/followup/{donors,strata,matched,covariates,explain}.py`, with spatial
additions under `degas3/spatial/`.

## Guardrails this stage adds

- A paired effect is reported with the number of donors, the fraction agreeing in sign, and the
  weakest leave-one-study-out estimate. A large gene count is not treated as a finding.
- Donor covariates are joined only on an identifier supplied with the data, never by similarity.
- The score is examined among controls against each covariate; a score that varies with a
  covariate in controls is reported as such.
- Any new search (covariate scans, marker pairs, external cohorts) states its own test count.
- Model-input genes are reported with the refit result next to the original effect when the refit
  has been run, and flagged as attribution when it has not.
- Statistics computed by ordinary regression on log CPM are labelled as a screen; FDRs come from
  the configured DE engine.

## Decisions

| # | Question | Status |
|---|---|---|
| 1 | Where do "non-self" marker lists for the matched contrast come from? | **Decided: derived from cluster markers**, with a user override. For a target cluster, non-self genes are markers that are specific to *other* clusters. See the note below. |
| 2 | How does the within-strata outcome contrast generalise to survival outcomes? | Open. Options: tertiles of the donor-level outcome; a per-gene Cox model on donor pseudobulk; binary only at first. |
| 3 | Are explanation labels shown in the report by default? | Open. Options: label as a table column beside the underlying numbers; numbers only; opt-in flag. |
| 4 | What is required before the paired contrast is trusted in malignant cells? | Open. Clone identity (inferred copy number) and cell-cycle phase are not handled. Options: require them as stratifiers; warn only; restrict the stage to non-malignant clusters until implemented. |

### Note on decision 1

Deriving non-self genes from cluster markers assumes a marker is specific to its own cluster. That
is not always true: a gene can be a marker of one cluster and still be genuinely expressed at a
lower level in the target cluster. Counting such a gene as contamination would remove real signal.
Proposed handling, to be confirmed:

- use only markers with high specificity (large fold change against every other cluster and low
  detection rate in the target cluster's deepest cells);
- report, per target cluster, which genes were used and their detection rate in that cluster;
- keep two variants in the output, as in the prototype: cells matched on depth and non-self
  fraction, and the cleanest half of cells. Disagreement between them is itself informative.

## Acceptance

- `docs/SPEC.md`, `docs/METHODS.md`, `agent/STAGES.md`, `agent/GUARDRAILS.md`,
  `agent/REPORTING.md` updated before code.
- `degas3 simulate` data runs every `followup` step in CI without a GPU.
- A synthetic generator with planted behaviours (frequency shift, within-state shift, shared
  state, single study, depth artifact) is added as a test, and the planted behaviours are recovered.
- No change to existing stage outputs.

## Prototype

The checks were prototyped outside this repository on one case study. That code reads a different
directory layout and has study-specific constants; it is a reference for the logic, not something
to merge as is.
