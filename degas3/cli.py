"""Command-line interface: ``degas3 <stage> RUN [options]``.

Working stages live in this package (doctor, simulate, init, inspect, cluster). Every other
stage is imported lazily from the module named in ``STAGE_MODULES``; the module must expose
``cli_<stage>(args: argparse.Namespace) -> int | None`` (``run_<stage>`` and ``main_<stage>`` are
also accepted). A missing module prints "stage '<name>' is not implemented yet" and exits 2.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from typing import Callable, Optional, Sequence

from degas3 import __version__

log = logging.getLogger("degas3")

#: Candidate modules per lazily imported stage, tried in order.
STAGE_MODULES: dict[str, list[str]] = {
    "prepare": ["degas3.prepare", "degas3.validate", "degas3.features"],
    "train": ["degas3.train"],
    "select": ["degas3.validate"],
    "fit": ["degas3.score", "degas3.train"],
    "score": ["degas3.score"],
    "stratify": ["degas3.stratify"],
    "de": ["degas3.de"],
    "pathways": ["degas3.pathways"],
    "explain": ["degas3.explain"],
    "neighborhoods": ["degas3.spatial.neighborhoods"],
    "maps": ["degas3.spatial.maps"],
    "cside": ["degas3.spatial.cside"],
    "report": ["degas3.report"],
    "slurm": ["degas3.slurm"],
}
#: Stages whose completion marker ``--force`` deletes here (spatial modules honor ``args.force`` themselves).
FORCE_MARKER_STAGES = {"stratify", "de", "pathways", "explain"}
_ENTRY_PREFIXES = ("cli_", "run_", "main_")

#: Order used by ``run-all`` (the minimal scenario).
RUN_ALL_CHAIN = ["prepare", "train", "select", "fit", "score", "stratify", "de", "pathways", "report"]


def _out(text: str) -> None:
    sys.stdout.write(text + "\n")


def _not_implemented(name: str) -> int:
    sys.stderr.write(f"stage '{name}' is not implemented yet\n")
    return 2


def _lazy_stage(name: str) -> Callable[[argparse.Namespace], int]:
    def handler(args: argparse.Namespace) -> int:
        if getattr(args, "force", False) and name in FORCE_MARKER_STAGES:
            from degas3.runs import RunDir

            RunDir(args.run).complete_path(name).unlink(missing_ok=True)
        for modname in STAGE_MODULES[name]:
            try:
                mod = importlib.import_module(modname)
            except ModuleNotFoundError as exc:
                if exc.name and (modname == exc.name or modname.startswith(exc.name + ".")):
                    continue  # this module (or a parent package) does not exist yet
                raise
            for prefix in _ENTRY_PREFIXES:
                fn = getattr(mod, prefix + name, None)
                if callable(fn):
                    return int(fn(args) or 0)
        return _not_implemented(name)

    return handler


# -- implemented stages ------------------------------------------------------------
def _cmd_doctor(args: argparse.Namespace) -> int:
    from degas3.doctor import exit_code, format_table, run_checks

    result = run_checks(args.run)
    _out(format_table(result))
    return exit_code(result)


def _cmd_simulate(args: argparse.Namespace) -> int:
    from degas3.simulate import simulate

    m = simulate(args.out_dir, seed=args.seed, n_patients=args.n_patients, n_cells=args.n_cells,
                 n_genes=args.n_genes, n_donors=args.n_donors)
    _out(f"wrote {len(m['files'])} files to {args.out_dir}")
    return 0


def _cmd_init(args: argparse.Namespace) -> int:
    from degas3.intake import format_summary, init_run

    m = init_run(
        args.run, bulk=args.bulk, pheno=args.pheno, cells=args.cells, outcome=args.outcome,
        modality=args.modality, donor_col=args.donor_col, cluster_col=args.cluster_col,
        label_col=args.label_col, sample_col=args.sample_col, dataset_col=args.dataset_col,
        patient_col=args.patient_col, pheno_id_col=args.pheno_id_col, spatial_key=args.spatial_key,
        already_log=args.already_log, force=args.force,
        **{k: v for k, v in (("model", args.model), ("cv", args.cv), ("feature_selection", args.feature_selection),
                             ("rank_scope", args.rank_scope)) if v is not None},
    )
    _out(format_summary(m))
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    from degas3.intake import inspect_run

    _, text = inspect_run(args.run)
    _out(text)
    return 0


def _cmd_cluster(args: argparse.Namespace) -> int:
    from degas3.cluster import run_cluster

    info = run_cluster(args.run, resolution=args.resolution, force=args.force)
    _out(json.dumps({k: info.get(k) for k in ("resolution", "n_clusters", "cluster_sizes")}))
    return 0


def _cmd_run_all(args: argparse.Namespace) -> int:
    from degas3.config import RunConfig

    cfg = RunConfig.load(args.run)
    chain = list(RUN_ALL_CHAIN)
    if not cfg.cluster_col:
        chain.insert(0, "cluster")
    if args.skip_pathways:
        chain.remove("pathways")
    for stage in chain:
        argv = [stage, args.run]
        if stage == "train":
            argv.append("--all")
            if args.allow_cpu:
                argv.append("--allow-cpu")
        elif stage == "fit":
            argv.append("--all")
            if args.allow_cpu:
                argv.append("--allow-cpu")
        elif stage == "pathways" and args.gmt:
            argv += ["--gmt", args.gmt]
        log.info("run-all: %s", " ".join(argv))
        code = main(argv)
        if code != 0:
            sys.stderr.write(f"run-all stopped at stage '{stage}' (exit {code})\n")
            return code
    return 0


# -- parser -------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="degas3", description="DEGASv3: transfer bulk patient outcomes to single-cell / spatial data.")
    p.add_argument("--version", action="version", version=f"degas3 {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="command", metavar="<stage>", required=True)

    def add(name: str, handler: Callable, help_: str, run: bool = True) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_, description=help_)
        if run:
            sp.add_argument("run", help="run directory")
        sp.set_defaults(handler=handler, command=name)
        return sp

    sp = add("doctor", _cmd_doctor, "check the environment (torch, CUDA, R, disk)", run=False)
    sp.add_argument("--run", default=None, help="write doctor.json into this run directory")

    sp = add("simulate", _cmd_simulate, "write toy bulk, single-cell and spatial data with a planted signal", run=False)
    sp.add_argument("out_dir")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--n-patients", type=int, default=120)
    sp.add_argument("--n-cells", type=int, default=3000)
    sp.add_argument("--n-genes", type=int, default=2000)
    sp.add_argument("--n-donors", type=int, default=8)

    sp = add("init", _cmd_init, "create a run directory from user inputs")
    sp.add_argument("--bulk", required=True, help="bulk expression (csv/tsv/h5ad/npz)")
    sp.add_argument("--pheno", required=True, help="phenotype csv: sample_id + outcome, or time,event")
    sp.add_argument("--cells", required=True, help="cells: h5ad, 10x h5/mtx dir, Xenium outs dir, Visium")
    sp.add_argument("--outcome", choices=["binary", "survival"], default=None)
    sp.add_argument("--donor-col", default=None)
    sp.add_argument("--cluster-col", default=None)
    sp.add_argument("--label-col", default=None, help="donor-level clinical label (0/1) in cell metadata")
    sp.add_argument("--modality", choices=["sc", "st_cell", "st_spot"], default=None)
    sp.add_argument("--dataset-col", default=None, help="phenotype column with the bulk dataset/batch")
    sp.add_argument("--patient-col", default=None, help="phenotype column grouping samples from one person")
    sp.add_argument("--sample-col", default=None, help="cell metadata column with the section/acquisition id")
    sp.add_argument("--pheno-id-col", default="sample_id")
    sp.add_argument("--spatial-key", default=None)
    sp.add_argument("--already-log", action="store_true", help="bulk values are log2(count+1)")
    sp.add_argument("--force", action="store_true", help="overwrite an existing run's inputs")
    sp.add_argument("--model", choices=["auto", "blank", "class"], default=None,
                    help="DEGAS variant: blank (outcome only), class (adds cluster classification), auto = class when a cluster column exists")
    sp.add_argument("--cv", choices=["patient_kfold", "leave_one_dataset_out"], default=None,
                    help="cross-validation: patient_kfold (grouped by patient) or leave_one_dataset_out (needs --dataset-col)")
    sp.add_argument("--feature-selection", choices=["variance", "mixed"], default=None,
                    help="gene ranking inside training folds: variance, or mixed (variance + cluster markers)")
    sp.add_argument("--rank-scope", choices=["global", "sample"], default=None,
                    help="percentile ranking of cell scores across all cells or within each sample/donor")

    add("inspect", _cmd_inspect, "print the intake summary and warnings (refreshes input_manifest.json)")

    sp = add("cluster", _cmd_cluster, "outcome-blind Leiden clustering of the run's cells")
    sp.add_argument("--resolution", type=float, default=None)
    sp.add_argument("--force", action="store_true")

    add("prepare", _lazy_stage("prepare"), "gene universe, folds, rankings, tasks")

    sp = add("train", _lazy_stage("train"), "train cross-validation tasks")
    g = sp.add_mutually_exclusive_group()
    g.add_argument("--task", type=int, default=None)
    g.add_argument("--all", action="store_true")
    sp.add_argument("--allow-cpu", action="store_true")

    add("select", _lazy_stage("select"), "cross-validation metrics and one-SE size selection")

    sp = add("fit", _lazy_stage("fit"), "fit final models")
    g = sp.add_mutually_exclusive_group()
    g.add_argument("--task", type=int, default=None,
                   help="fit only the N-th (size, seed) pair, 0-based, sizes ascending x seeds ascending (SLURM array index)")
    g.add_argument("--all", action="store_true", help="fit every (size, seed) pair (default)")
    sp.add_argument("--allow-cpu", action="store_true")

    add("score", _lazy_stage("score"), "score cells and patients with the rank-based ensemble")

    def force(sp, what="redo the stage even if it is marked complete"):
        sp.add_argument("--force", action="store_true", help=what)

    sp = add("stratify", _lazy_stage("stratify"), "within-cluster high/low split")
    force(sp)
    sp.add_argument("--split", choices=["median", "quantiles"], default=None)
    sp.add_argument("--quantiles", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"))

    sp = add("de", _lazy_stage("de"), "high-vs-low differential expression")
    sp.add_argument("--method", choices=["auto", "pseudobulk_deseq2", "wilcoxon"], default=None)
    sp.add_argument("--engine", choices=["pydeseq2", "edger"], default=None)
    force(sp)

    sp = add("pathways", _lazy_stage("pathways"), "pathway enrichment")
    sp.add_argument("--gmt", default=None, help="local GMT gene-set file (default: download MSigDB via gseapy)")
    sp.add_argument("--collection", default=None, help="MSigDB collection, e.g. h.all")
    sp.add_argument("--species", choices=["human", "mouse"], default=None)
    force(sp)

    sp = add("explain", _lazy_stage("explain"), "gradient SHAP model attributions")
    force(sp)
    sp = add("neighborhoods", _lazy_stage("neighborhoods"), "single-cell spatial neighborhood pooling")
    force(sp)
    sp = add("maps", _lazy_stage("maps"), "score and cluster maps")
    force(sp)
    sp = add("cside", _lazy_stage("cside"), "RCTD -> C-SIDE export/run/parse")
    force(sp)
    sp.add_argument("--reference", default=None, help="single-cell h5ad with cluster labels used as the RCTD reference")
    sp.add_argument("--reference-label", default=None, help="obs column of --reference with the cell-type labels")
    sp.add_argument("--max-cores", type=int, default=None, help="cores for RCTD / C-SIDE")
    sp.add_argument("--fdr", type=float, default=None, help="FDR threshold for C-SIDE significance")
    sp.add_argument("--cell-type-threshold", type=int, default=None, help="minimum cells of a type for C-SIDE testing")
    sp = add("report", _lazy_stage("report"), "self-contained HTML report and METHODS.md")
    force(sp, "accepted for consistency; the report is always rebuilt")

    sp = add("slurm", _lazy_stage("slurm"), "print sbatch scripts for the run")
    sp.add_argument("--account", required=True)
    sp.add_argument("--partition", required=True, help="GPU partition (train, fit, explain)")
    sp.add_argument("--cpu-partition", default=None, help="CPU partition for the other stages (default general)")
    sp.add_argument("--time", action="append", default=None, metavar="[STAGE=]HH:MM:SS",
                    help="wall time; repeat as stage=HH:MM:SS, or give one value for all stages")
    sp.add_argument("--venv", default=None, help="virtual environment to source in every job")
    sp.add_argument("--modules", default=None, help="argument string for `module load`")
    sp.add_argument("--explain", action="store_true", help="add the explain stage after pathways")
    sp.add_argument("--gpus", type=int, default=1, help="GPUs per GPU job")

    sp = add("run-all", _cmd_run_all, "default chain for the minimal scenario")
    sp.add_argument("--allow-cpu", action="store_true")
    sp.add_argument("--gmt", default=None, help="local GMT file passed to the pathways stage")
    sp.add_argument("--skip-pathways", action="store_true", help="omit the pathways stage")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    for noisy in ("fontTools", "matplotlib", "numba", "h5py", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        return int(args.handler(args) or 0)
    except ValueError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
