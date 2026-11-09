"""Run directory helper: paths, completion markers, hashing, and environment capture."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from degas3 import __version__
from degas3.contracts import LAYOUT, STAGE_COMPLETE_FILES, STAGE_DIRS


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_path(path: str | Path) -> str:
    """SHA-256 of a file, or of a directory (digest over sorted relative names and file digests)."""
    p = Path(path)
    if p.is_file():
        return sha256_file(p)
    if not p.is_dir():
        raise ValueError(f"cannot hash {p}: path does not exist.")
    h = hashlib.sha256()
    for f in sorted(q for q in p.rglob("*") if q.is_file()):
        h.update(str(f.relative_to(p)).encode())
        h.update(sha256_file(f).encode())
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    """Digest of a JSON-serialisable object with sorted keys."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def git_commit() -> Optional[str]:
    """Commit hash of the git checkout containing the degas3 package, if any."""
    pkg_dir = Path(__file__).resolve().parent
    try:
        out = subprocess.run(
            ["git", "-C", str(pkg_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None


def environment() -> dict[str, Any]:
    """Python, torch, CUDA, and GPU description for provenance records."""
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "degas3": __version__,
        "torch": None,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_name": None,
        "gpu_memory_gb": None,
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_version"] = torch.version.cuda
        if env["cuda_available"]:
            props = torch.cuda.get_device_properties(0)
            env["gpu_name"] = props.name
            env["gpu_memory_gb"] = round(props.total_memory / 1e9, 2)
    except ImportError:
        pass
    return env


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunDir:
    """Paths and completion markers for one run directory.

    A stage is complete when its completion file exists; stages skip when it does.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    def __repr__(self) -> str:
        return f"RunDir({str(self.root)!r})"

    # -- paths -------------------------------------------------------------
    def path(self, *parts: str | int) -> Path:
        return self.root.joinpath(*[str(p) for p in parts])

    def ensure(self, *parts: str | int) -> Path:
        p = self.path(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def config_path(self) -> Path:
        return self.path(LAYOUT.CONFIG)

    @property
    def inputs(self) -> Path:
        return self.path(LAYOUT.INPUTS)

    @property
    def bulk_npz(self) -> Path:
        return self.path(LAYOUT.INPUTS, LAYOUT.BULK_NPZ)

    @property
    def cells_h5ad(self) -> Path:
        return self.path(LAYOUT.INPUTS, LAYOUT.CELLS_H5AD)

    @property
    def input_manifest(self) -> Path:
        return self.path(LAYOUT.INPUTS, LAYOUT.INPUT_MANIFEST)

    def stage_dir(self, stage: str) -> Path:
        if stage not in STAGE_DIRS:
            raise ValueError(f"unknown stage {stage!r}; valid stages: {sorted(STAGE_DIRS)}.")
        return self.path(STAGE_DIRS[stage])

    def cv_task_dir(self, task: int, seed: int) -> Path:
        return self.path(LAYOUT.CV, f"task_{task:03d}", f"seed_{seed}")

    def model_dir(self, size: int, seed: int) -> Path:
        return self.path(LAYOUT.MODELS, f"size_{size:04d}", f"seed_{seed}")

    # -- completion markers -------------------------------------------------
    def complete_path(self, stage: str, *subdir: str | int) -> Path:
        """Completion file of a stage, or of a sub-unit (e.g. a CV task) below the stage dir."""
        if subdir:
            return self.path(STAGE_DIRS.get(stage, stage), *subdir, LAYOUT.COMPLETE)
        return self.stage_dir(stage) / STAGE_COMPLETE_FILES.get(stage, LAYOUT.COMPLETE)

    def stage_complete(self, stage: str, *subdir: str | int) -> bool:
        return self.complete_path(stage, *subdir).is_file()

    def read_complete(self, stage: str, *subdir: str | int) -> Optional[dict]:
        p = self.complete_path(stage, *subdir)
        return json.loads(p.read_text()) if p.is_file() else None

    def write_complete(
        self, stage: str, payload: Optional[dict] = None, *subdir: str | int
    ) -> Path:
        """Write the completion record with provenance fields merged under ``payload``."""
        p = self.complete_path(stage, *subdir)
        p.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "stage": stage,
            "completed_at": _now(),
            "package_version": __version__,
            "git_commit": git_commit(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        record.update(payload or {})
        p.write_text(json.dumps(record, indent=2, default=str) + "\n")
        return p
