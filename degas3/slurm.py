"""Generate a bash script that submits the DEGASv3 stage chain with SLURM dependencies.

``degas3 slurm RUN --account A --partition P`` prints ``jobs.sh`` to stdout and writes the same
text to ``<run>/slurm/jobs.sh`` plus ``<run>/slurm/README.md``. This module never calls
``sbatch``; the user (or agent, with the user's consent) runs the script.

Chain: prepare -> train (array) -> select -> fit (array over size x seed once cv/selection.json exists) -> score -> stratify -> de -> pathways -> report
(optionally explain after pathways). Every link is ``--dependency=afterok``. All stages skip
work whose ``complete.json`` exists, so a failed chain is resumed by re-running the script.

Options not exposed as CLI flags are read from the namespace if present, otherwise from
environment variables: ``DEGAS3_CPU_PARTITION`` (default ``general``), ``DEGAS3_SLURM_TIME``
(one value for all stages, or ``train=08:00:00,fit=12:00:00``), ``DEGAS3_VENV``,
``DEGAS3_MODULES`` (a ``module load`` argument string), ``DEGAS3_SLURM_EXPLAIN=1``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Optional

from degas3.contracts import LAYOUT
from degas3.runs import RunDir

log = logging.getLogger(__name__)

DEFAULT_TIMES = {
    "prepare": "01:00:00", "train": "04:00:00", "select": "01:00:00", "fit": "08:00:00", "score": "02:00:00",
    "stratify": "00:30:00", "de": "04:00:00", "pathways": "02:00:00", "explain": "04:00:00", "report": "00:30:00",
}
MEMORY = {"de": "64G", "pathways": "16G", "report": "8G", "stratify": "16G"}
DEFAULT_MEMORY = "32G"
_TOKEN = re.compile(r"^[A-Za-z0-9_.,:+-]+$")
_TIME = re.compile(r"^(\d+-)?\d{1,2}:\d{2}(:\d{2})?$")


def _token(value: str, what: str) -> str:
    if not value or not _TOKEN.match(value):
        raise ValueError(f"{what} {value!r} is empty or has unsupported characters; use letters, digits, '_', '-', '.', ',' or ':'.")
    return value


def parse_times(spec: Optional[str]) -> dict[str, str]:
    """Per-stage wall times: defaults, overridden by ``HH:MM:SS`` (all stages) or ``stage=HH:MM:SS,...``."""
    times = dict(DEFAULT_TIMES)
    if not spec:
        return times
    for part in spec.split(","):
        part = part.strip()
        stage, _, value = part.rpartition("=")
        if not _TIME.match(value):
            raise ValueError(f"bad time {value!r}; use HH:MM:SS or D-HH:MM:SS, e.g. --time train=08:00:00,fit=12:00:00.")
        if stage:
            if stage not in times:
                raise ValueError(f"unknown stage {stage!r} in --time; valid: {sorted(times)}.")
            times[stage] = value
        else:
            times = {k: value for k in times}
    return times


def default_venv() -> Optional[str]:
    """Environment containing the running interpreter, when it looks like a venv/conda prefix."""
    prefix = Path(sys.prefix)
    return str(prefix) if (prefix / "bin" / "activate").is_file() else None


def task_ids(rd: RunDir) -> Optional[list[int]]:
    """CV task ids from ``prepared/tasks.json`` (list, ``{"tasks": [...]}`` or ``{"n_tasks": N}``), else None."""
    path = rd.path(LAYOUT.PREPARED, "tasks.json")
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    tasks = data.get("tasks") if isinstance(data, dict) else data
    if isinstance(data, dict) and tasks is None and isinstance(data.get("n_tasks"), int):
        return list(range(data["n_tasks"]))
    if not isinstance(tasks, list) or not tasks:
        return None
    ids = [t.get("task", t.get("id", i)) if isinstance(t, dict) else i for i, t in enumerate(tasks)]
    return sorted(int(i) for i in ids)


def fit_task_count(rd: RunDir) -> int:
    """Number of (size, seed) final fits, or 0 when ``cv/selection.json`` does not exist yet."""
    from degas3.config import RunConfig
    from degas3.validate import read_selected_sizes

    sizes = read_selected_sizes(rd, required=False)
    return len(sizes) * RunConfig.load(rd.root).final_seeds


def _array_spec(ids: list[int]) -> str:
    """Compress ids into a SLURM array spec such as ``0-4,7``."""
    out, start, prev = [], ids[0], ids[0]
    for i in ids[1:] + [None]:
        if i is not None and i == prev + 1:
            prev = i
            continue
        out.append(str(start) if start == prev else f"{start}-{prev}")
        if i is not None:
            start = prev = i
    return ",".join(out)


def build_script(
    run: str | Path, *, account: str, partition: str, cpu_partition: str = "general", gpus: int = 1,
    times: Optional[dict[str, str]] = None, venv: Optional[str] = None, modules: Optional[str] = None,
    explain: bool = False,
) -> str:
    """The ``jobs.sh`` text for ``run``. Pure function of its inputs and the run's ``prepared/``, ``cv/`` files."""
    rd = RunDir(run)
    times = times or dict(DEFAULT_TIMES)
    account, partition, cpu_partition = _token(account, "account"), _token(partition, "partition"), _token(cpu_partition, "cpu partition")
    if gpus < 1:
        raise ValueError("--gpus must be >= 1.")
    run_q = shlex.quote(str(rd.root))
    setup = ["set -euo pipefail"]
    if modules:
        setup.append(f"module load {modules}")
    if venv:
        setup.append(f"source {shlex.quote(str(Path(venv) / 'bin' / 'activate'))}")
    setup_str = "; ".join(setup)
    ids = task_ids(rd)
    lines = [
        "#!/usr/bin/env bash",
        f"# DEGASv3 SLURM chain for {rd.root}",
        "# Generated by `degas3 slurm`; read it, then run `bash jobs.sh` to submit. Stages skip when complete.json exists,",
        "# so after a failure fix the cause and run this script again.",
        "set -euo pipefail",
        f"RUN={run_q}",
        'mkdir -p "$RUN/logs" "$RUN/slurm"',
        "",
        "submit() {  # submit NAME STAGE ARRAY DEPENDENCY COMMAND... -> prints the job id",
        '  local name="$1" stage="$2" array="$3" dep="$4"; shift 4',
        "  local args=(--parsable)",
        '  args+=(--job-name="degas3_$name" --account=' + account + ' --cpus-per-task=8)',
        '  case "$stage" in',
        "    gpu) args+=(--partition=" + partition + f" --gres=gpu:{gpus});;",
        "    *) args+=(--partition=" + cpu_partition + ");;",
        "  esac",
        '  if [ -n "$array" ]; then args+=(--array="$array" --output="$RUN/logs/${name}_%A_%a.out"); else args+=(--output="$RUN/logs/${name}_%j.out"); fi',
        '  if [ -n "$dep" ]; then args+=(--dependency="afterok:$dep" --kill-on-invalid-dep=yes); fi',
        '  sbatch "${args[@]}" "$@"',
        "}",
        "",
    ]

    def job(var: str, name: str, kind: str, command: str, dep: Optional[str], array: str = "") -> None:
        mem = MEMORY.get(name, DEFAULT_MEMORY)
        wrap = shlex.quote(f"{setup_str}; {command}")
        dep_arg = f'"${dep}"' if dep else '""'
        lines.append(
            f'{var}=$(submit {name} {kind} "{array}" {dep_arg} --time={times[name]} --mem={mem} --wrap={wrap})'
        )
        lines.append(f'echo "{name}: job $' + var + '"')

    py = "python -m degas3"
    job("J_PREPARE", "prepare", "cpu", f"{py} prepare {run_q}", None)
    if ids:
        job("J_TRAIN", "train", "gpu", f'{py} train {run_q} --task "$SLURM_ARRAY_TASK_ID"', "J_PREPARE", _array_spec(ids))
    else:
        job("J_TRAIN", "train", "gpu", f"{py} train {run_q} --all", "J_PREPARE")
    job("J_SELECT", "select", "cpu", f"{py} select {run_q}", "J_TRAIN")
    n_fit = fit_task_count(rd)
    if n_fit:  # selection exists: one array task per (size, seed), 0-based like `train --task`
        job("J_FIT", "fit", "gpu", f'{py} fit {run_q} --task "$SLURM_ARRAY_TASK_ID"', "J_SELECT", _array_spec(list(range(n_fit))))
    else:  # sizes unknown until `select` has run
        job("J_FIT", "fit", "gpu", f"{py} fit {run_q} --all", "J_SELECT")
    job("J_SCORE", "score", "cpu", f"{py} score {run_q}", "J_FIT")
    job("J_STRATIFY", "stratify", "cpu", f"{py} stratify {run_q}", "J_SCORE")
    job("J_DE", "de", "cpu", f"{py} de {run_q}", "J_STRATIFY")
    job("J_PATHWAYS", "pathways", "cpu", f"{py} pathways {run_q}", "J_DE")
    last = "J_PATHWAYS"
    if explain:
        job("J_EXPLAIN", "explain", "gpu", f"{py} explain {run_q}", "J_PATHWAYS")
        last = "J_PATHWAYS:$J_EXPLAIN"
    job("J_REPORT", "report", "cpu", f"{py} report {run_q}", last)
    lines += ['echo "submitted; follow with: squeue -u $USER   (logs in $RUN/logs)"', ""]
    return "\n".join(lines)


def slurm_readme(run: str | Path) -> str:
    """Resume instructions written next to ``jobs.sh``."""
    return f"""# SLURM chain for this run

`jobs.sh` submits: prepare, train (array over CV tasks), select, fit, score, stratify, de, pathways,
report (explain optional), each with `--dependency=afterok` on its predecessor. It was generated by
`degas3 slurm` and never submitted by degas3 itself. Logs go to `logs/`.

Submit: `bash slurm/jobs.sh` (from the run directory's parent or anywhere; paths are absolute).

## Resume after a failure
1. `squeue -u $USER` and `sacct -j JOBID --format=JobID,State,ExitCode` find the failed stage; read its log in `logs/`.
2. Fix the cause (time limit, memory, missing optional dependency).
3. Run `bash slurm/jobs.sh` again. Every stage skips work whose `complete.json` exists, so finished
   tasks cost seconds. A failed array task can be run alone with `degas3 train RUN --task N`.
4. If parameters must change, start a new run directory; do not edit `config.json` of a finished run.

Regenerate with other limits: `degas3 slurm RUN --account A --partition P --gpus 1 > slurm/jobs.sh`.
Environment overrides: DEGAS3_CPU_PARTITION, DEGAS3_SLURM_TIME (`train=08:00:00,fit=12:00:00`), DEGAS3_VENV, DEGAS3_MODULES, DEGAS3_SLURM_EXPLAIN=1.
"""


def cli_slurm(args: argparse.Namespace) -> int:
    """CLI entry: print the script to stdout, save it and a README under ``<run>/slurm/``."""
    rd = RunDir(args.run)
    if not rd.config_path.is_file():
        raise ValueError(f"{rd.root} is not a run directory (no config.json); run `degas3 init` first.")

    def opt(name: str, env: str, default=None):
        value = getattr(args, name, None)
        if isinstance(value, (list, tuple)):
            value = ",".join(value)
        return value if value not in (None, "") else os.environ.get(env, default)

    venv = opt("venv", "DEGAS3_VENV", default_venv())
    script = build_script(
        rd.root, account=args.account, partition=args.partition,
        cpu_partition=opt("cpu_partition", "DEGAS3_CPU_PARTITION", "general"),
        gpus=int(getattr(args, "gpus", 1) or 1), times=parse_times(opt("time", "DEGAS3_SLURM_TIME")),
        venv=venv, modules=opt("modules", "DEGAS3_MODULES"),
        explain=bool(getattr(args, "explain", False) or os.environ.get("DEGAS3_SLURM_EXPLAIN") == "1"),
    )
    out = rd.ensure("slurm")
    (out / "jobs.sh").write_text(script)
    (out / "README.md").write_text(slurm_readme(rd.root))
    log.info("wrote %s and %s", out / "jobs.sh", out / "README.md")
    sys.stdout.write(script)
    return 0
