# Running on a SLURM cluster

## Set up the environment

Use a GPU Python module so the torch build matches the driver. Names differ by site.

```bash
module avail python        # find a Python 3.10+ GPU module
module load python/<version> cuda/<version>
uv venv .venv && source .venv/bin/activate
uv pip install -e .
python -c "import torch; print(torch.cuda.is_available())"   # run this on a GPU node
```

Do not run training on a login node. Use an interactive GPU job for `degas3 doctor`.

## Create the run (cheap steps)

Run `init`, `inspect`, `cluster`, and `prepare` in a short interactive job or
the first batch job. Keep raw inputs on read-only storage and write only to the
run directory.

## Generate the job script

```bash
degas3 slurm runs/mine --account ACCOUNT --partition PARTITION --gpus 1 > jobs.sh
less jobs.sh     # read before submitting
bash jobs.sh     # submits jobs with dependencies
```

Check the script for time limits, memory, and module lines. Edit paths only in
the generated script, not in package code. Submit only with the user's consent.

## Stage dependencies

| Job | Depends on |
|---|---|
| prepare | none |
| train (array over tasks) | prepare |
| select | all train tasks |
| fit (array over size x seed) | select |
| score | all fit tasks |
| stratify | score |
| de | stratify |
| pathways | de |
| explain (optional `--explain`, GPU) | pathways |
| report | pathways (and explain) |

Each job uses `afterok` on its predecessor. Array tasks call
`degas3 train RUN --task N` and `degas3 fit RUN --task N` (0-based) so a failed task
can be rerun alone. The fit array needs `cv/selection.json`; if you generate the script
before `select` has run, fit is a single `fit --all` job, so regenerate the script after
`select` to get the array. Other options: `--cpu-partition`, `--time stage=HH:MM:SS`
(repeatable), `--venv`, `--modules`, `--explain`.

## Resume after failure

1. Check `squeue` and the job logs (`sacct -j JOBID`).
2. Find the failing stage and read its error.
3. Rerun the same command (or the same array index). Finished tasks skip because `complete.json` exists.
4. If parameters change, make a new run directory. Do not edit `config.json` of a finished run.
5. Resubmit the downstream jobs.

## Keep inputs read-only

- Put raw data in a read-only directory. Pass those paths to `init`.
- `inputs/` in the run directory holds hashed copies and the manifest. Never edit them.
- Converted files (transposed matrices, h5ad exports) go in a separate working folder.

## Provenance

`complete.json` records the SLURM job id. `report/manifest.json` collects input
hashes, the config hash, package version, torch and CUDA versions, GPU name, seeds, and timings.
