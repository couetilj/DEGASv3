"""``train`` stage: fit cross-validation models (one task = one fold x one gene-list size).

Each task is trained ``cv_seeds`` times. Preprocessing (``preprocess_counts``) is applied after
gene selection, to the training bulk rows and to the fold's eligible cells only. Held-out bulk
rows never influence the gene list, the preprocessing, or the weights.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from degas3.config import RunConfig
from degas3.core.models import build_model, fit, predict_scores, resolve_device, save_checkpoint
from degas3.core.options import make_options
from degas3.core.preprocess import preprocess_counts
from degas3.io.bulk import load_bulk_npz
from degas3.prepare import load_prepared
from degas3.runs import RunDir

log = logging.getLogger(__name__)

CELL_CHUNK = 20000


def bulk_targets(pheno: pd.DataFrame, outcome_type: str) -> np.ndarray:
    """1-D 0/1 outcome (binary) or an (n, 2) array of time and event (survival)."""
    if outcome_type == "binary":
        return pheno["outcome"].to_numpy().astype(int)
    return pheno[["time", "event"]].to_numpy().astype(float)


def preprocess_cells(counts, rows: np.ndarray, gene_idx: np.ndarray, chunk: int = CELL_CHUNK) -> np.ndarray:
    """``preprocess_counts`` of ``counts[rows][:, gene_idx]`` computed in row chunks (float32)."""
    sub = counts[:, gene_idx].tocsr() if hasattr(counts, "tocsr") else counts[:, gene_idx]
    out = np.empty((len(rows), len(gene_idx)), dtype=np.float32)
    for s in range(0, len(rows), chunk):
        block = sub[rows[s:s + chunk]]
        block = block.toarray() if hasattr(block, "toarray") else np.asarray(block)
        out[s:s + len(block)] = preprocess_counts(block)
    return out


def cell_classes(meta: pd.DataFrame, rows: np.ndarray, cfg: RunConfig) -> Optional[tuple[np.ndarray, list[str]]]:
    """Integer cluster codes of ``rows`` and the sorted class names, or None for a Blank model."""
    has = bool(cfg.cluster_col) and bool((meta["cluster"] != "").any())
    kind = cfg.resolved_model(has)
    if kind == "blank":
        return None
    if not has:
        raise ValueError("model='class' needs cluster labels (cluster_col); run `degas3 cluster RUN` or set model='blank'.")
    names = sorted(meta["cluster"].iloc[rows].unique())
    return pd.Categorical(meta["cluster"].iloc[rows], categories=names).codes.astype(np.int64), names


def fit_one(cfg: RunConfig, outcome_type: str, px: np.ndarray, py: np.ndarray, sx: np.ndarray,
            classes: Optional[tuple[np.ndarray, list[str]]], genes: list[str], seed: int, fold: int,
            device, ckpt_path: Path, size: int) -> tuple[pd.DataFrame, object]:
    """Build, fit and checkpoint one model; returns (loss table, model)."""
    over = dict(input_shape=len(genes), feature_dim=cfg.feature_dim, seed=seed, tot_iters=cfg.iterations,
                batch_size=cfg.batch_size, pat_batch_size=cfg.batch_size, fold=-1)
    if classes is not None:
        over["high_reso_output_shape"] = len(classes[1])
    opt = make_options(outcome_type, classes is not None, **over)
    model = build_model(opt, device)
    losses = fit(model, px, py, sx, classes[0] if classes else None, iterations=cfg.iterations, seed=seed)
    save_checkpoint(model, ckpt_path, genes=genes, seed=seed, size=size, fold=fold,
                    extra={"cell_classes": classes[1] if classes else None})
    return losses, model


def _seed_done(rd: RunDir, task: int, seed: int) -> bool:
    return rd.stage_complete("train", f"task_{task:03d}", f"seed_{seed}")


def run_task(rd: RunDir, cfg: RunConfig, prep: dict, bulk, task: dict, device) -> None:
    """Train all missing CV seeds of one task and write predictions for the held-out bulk rows."""
    tid = task["task"]
    seeds = [s for s in range(cfg.cv_seeds) if not _seed_done(rd, tid, s)]
    if not seeds:
        log.info("task %d already complete; skipping.", tid)
        return
    fold = next(f for f in prep["folds"]["folds"] if f["fold_id"] == task["fold_id"])
    genes = task["genes"]
    shared_idx = {g: i for i, g in enumerate(prep["genes"])}
    gi = np.array([shared_idx[g] for g in genes])
    bidx = {g: i for i, g in enumerate(bulk.genes)}
    bg = np.array([bidx[g] for g in genes])
    tr, te = np.asarray(fold["train_idx"]), np.asarray(fold["test_idx"])
    if set(bulk.pheno.index[tr]) & set(bulk.pheno.index[te]):
        raise ValueError("train and test share a sample; folds.json is corrupt, rerun prepare.")
    excluded = np.load(rd.stage_dir("prepare") / fold["excluded_file"])
    keep = np.ones(prep["counts"].shape[0], bool)
    keep[excluded] = False
    rows = np.flatnonzero(keep)
    if len(rows) < 2:
        raise ValueError(f"fold {fold['name']} has {len(rows)} eligible cells; check paired_datasets and donor ids.")
    px = preprocess_counts(bulk.counts[tr][:, bg], already_log=bulk.already_log).astype(np.float32)
    pxt = preprocess_counts(bulk.counts[te][:, bg], already_log=bulk.already_log).astype(np.float32)
    sx = preprocess_cells(prep["counts"], rows, gi)
    classes = cell_classes(prep["meta"], rows, cfg)
    y = bulk_targets(bulk.pheno, bulk.outcome_type)
    for seed in seeds:
        d = rd.cv_task_dir(tid, seed)
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        losses, model = fit_one(cfg, bulk.outcome_type, px, y[tr], sx, classes, genes, seed,
                                task["fold_id"], device, d / "checkpoint.pt", task["size"])
        losses.to_csv(d / "losses.csv", index=False)
        pred = bulk.pheno.iloc[te].drop(columns=[c for c in ("patient_id", "dataset") if c in bulk.pheno.columns]).copy()
        pred.insert(0, "sample_id", bulk.pheno.index[te])
        pred.insert(1, "fold_id", task["fold_id"])
        pred["score"] = predict_scores(model, pxt)
        pred.to_csv(d / "heldout_predictions.csv", index=False)
        rd.write_complete("train", {
            "task": tid, "seed": seed, "fold_id": task["fold_id"], "size": task["size"],
            "iterations": cfg.iterations, "n_train_bulk": int(len(tr)), "n_test_bulk": int(len(te)),
            "n_cells": int(len(rows)), "model_type": model.opt["model_type"], "device": str(device),
            "seconds": round(time.time() - t0, 2),
        }, f"task_{tid:03d}", f"seed_{seed}")
        log.info("task %d seed %d done (%.1fs).", tid, seed, time.time() - t0)
        del model


def cli_train(args: argparse.Namespace) -> int:
    rd = RunDir(args.run)
    if rd.stage_complete("train"):
        log.info("train already complete for %s; skipping.", rd.root)
        return 0
    cfg = RunConfig.load(rd.root)
    prep = load_prepared(rd)
    tasks = prep["tasks"]
    task_arg = getattr(args, "task", None)
    if task_arg is not None:
        if not 0 <= task_arg < len(tasks):
            raise ValueError(f"--task {task_arg} out of range; valid tasks are 0..{len(tasks) - 1}.")
        todo = [tasks[task_arg]]
    else:
        todo = tasks
    device = resolve_device("auto", cfg.allow_cpu or bool(getattr(args, "allow_cpu", False)))
    bulk = load_bulk_npz(rd.bulk_npz)
    for t in todo:
        run_task(rd, cfg, prep, bulk, t, device)
    if all(_seed_done(rd, t["task"], s) for t in tasks for s in range(cfg.cv_seeds)):
        rd.write_complete("train", {"n_tasks": len(tasks), "cv_seeds": cfg.cv_seeds, "iterations": cfg.iterations})
    return 0
