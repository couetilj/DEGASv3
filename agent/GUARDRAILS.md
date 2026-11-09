# Scientific guardrails

| Rule | Why |
|---|---|
| The patient is the unit for outcome inference. | Outcomes are measured per person; cells inherit nothing independent. |
| Donors are the unit for cell-level DE. | Cells from one donor are correlated; treating them as replicates gives false precision. |
| Rank and select genes only inside training folds. | Ranking on all patients leaks the held-out outcome into the model. |
| Group folds by `patient_id`, never split a patient. | Repeated samples from one person would inflate validation. |
| Exclude cells from donors that match held-out bulk patients. | The same person on both sides leaks identity. |
| Do not choose the feature size on the final test. | Selection on the test set makes the reported metric optimistic. State the optimism anyway. |
| Use the one-standard-error rule, not the single best size. | The best size is partly noise. |
| Percentile scores are not probabilities. | Ranks depend on the reference cells, not on calibrated risk. |
| Do not re-rank the mean of percentiles. | The ensemble is defined as the mean of per-size percentiles. |
| Genes used by the model are model attribution, not independent discovery. | The score is built from those genes, so association is partly built in. |
| Clustering never sees outcome or scores. | Otherwise clusters would encode the answer. |
| With no replication, run cell-level Wilcoxon and call it that. | It is the honest label for a test whose units are cells. |
| Report ties, coverage, and skipped clusters. | Silent exclusions change what the result means. |
| Check donor counts before every contrast. | Tests with one donor per group cannot support donor-level claims. |
| Never pad, impute, or invent missing inputs. | Fabricated values hide data problems. |
| Keep raw inputs read-only; convert copies. | Preserves provenance. |
| Use a new run directory for each attempt. | Completed runs are records. |
| Never change package code to fit a dataset. | Code changes alter the method; adapt inputs or ask. |
| Never fall back to CPU silently. | CPU runs are slow and may hide a setup problem. |
| Do not send expression data, cell ids, or patient metadata to external services. | Data protection. |
| Describe the data and the method; do not interpret biology unless asked. | Results here are associations from a model. |
| Treat spatial neighborhoods and overlapping windows as dependent. | Overlap makes them non-independent; group by donor or section. |
