"""Environment checks: python, torch/CUDA, numba cache, optional packages, R, disk."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from degas3.runs import environment

OK, WARN, FAIL = "ok", "warn", "fail"


def _check_import(name: str, required: bool, hint: str) -> dict:
    try:
        importlib.import_module(name)
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "installed"
        return {"name": name, "status": OK, "detail": version}
    except Exception as exc:  # broken installs raise more than ImportError
        return {"name": name, "status": FAIL if required else WARN, "detail": f"not importable ({type(exc).__name__}); {hint}"}


def _writable(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory):
            pass
        return True
    except OSError:
        return False


def _numba_check() -> dict:
    env_dir = os.environ.get("NUMBA_CACHE_DIR")
    if env_dir:
        ok = _writable(Path(env_dir))
        return {"name": "numba cache", "status": OK if ok else WARN, "detail": f"NUMBA_CACHE_DIR={env_dir} ({'writable' if ok else 'not writable'})"}
    spec = importlib.util.find_spec("scanpy")
    if spec is None or not spec.submodule_search_locations:
        return {"name": "numba cache", "status": WARN, "detail": "scanpy not found; cannot check"}
    cache = Path(list(spec.submodule_search_locations)[0]) / "__pycache__"
    if _writable(cache):
        return {"name": "numba cache", "status": OK, "detail": f"{cache} writable"}
    return {"name": "numba cache", "status": WARN,
            "detail": "package cache not writable; set NUMBA_CACHE_DIR to a writable directory (e.g. export NUMBA_CACHE_DIR=$TMPDIR/numba_cache)"}


def _r_checks() -> list[dict]:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return [{"name": "Rscript", "status": WARN, "detail": "not on PATH; edgeR and RCTD/C-SIDE paths are unavailable"}]
    rows = [{"name": "Rscript", "status": OK, "detail": rscript}]
    code = 'for (p in c("edgeR","spacexr")) cat(p, requireNamespace(p, quietly=TRUE), "\\n")'
    try:
        res = subprocess.run([rscript, "-e", code], capture_output=True, text=True, timeout=120)
        found = {ln.split()[0]: ln.split()[1] == "TRUE" for ln in res.stdout.splitlines() if len(ln.split()) == 2}
    except (OSError, subprocess.SubprocessError):
        found = {}
    for pkg in ("edgeR", "spacexr"):
        rows.append({"name": f"R:{pkg}", "status": OK if found.get(pkg) else WARN,
                     "detail": "installed" if found.get(pkg) else "not installed (optional)"})
    return rows


def run_checks(run: Optional[str | Path] = None) -> dict:
    """Run all checks; return {"checks": [...], "environment": {...}, "ok": bool}."""
    checks: list[dict] = [{
        "name": "python", "status": OK if sys.version_info >= (3, 10) else FAIL,
        "detail": sys.version.split()[0] + ("" if sys.version_info >= (3, 10) else " (need >= 3.10)"),
    }]
    torch_row = _check_import("torch", True, "install PyTorch (https://pytorch.org)")
    checks.append(torch_row)
    env = environment()
    if torch_row["status"] == OK:
        if env["cuda_available"]:
            checks.append({"name": "CUDA", "status": OK, "detail": f"CUDA {env['cuda_version']}"})
            checks.append({"name": "GPU", "status": OK, "detail": f"{env['gpu_name']} ({env['gpu_memory_gb']} GB)"})
        else:
            checks.append({"name": "CUDA", "status": WARN, "detail": "not available; training needs --allow-cpu (slow)"})
    checks.append(_numba_check())
    for name, required, hint in (
        ("scanpy", True, "pip install scanpy"), ("anndata", True, "pip install anndata"),
        ("pydeseq2", False, "needed for pseudobulk DE"), ("gseapy", False, "needed for pathways"),
        ("shap", False, "optional; pip install 'degas3[explain]'"),
    ):
        checks.append(_check_import(name, required, hint))
    checks += _r_checks()
    root = Path(run) if run else Path.cwd()
    probe = root if root.exists() else root.parent if root.parent.exists() else Path.cwd()
    free_gb = shutil.disk_usage(probe).free / 1e9
    checks.append({"name": "free disk", "status": OK if free_gb >= 10 else WARN,
                   "detail": f"{free_gb:.1f} GB free at {probe}"})
    result = {"checks": checks, "environment": env,
              "ok": torch_row["status"] == OK}
    if run:
        Path(run).mkdir(parents=True, exist_ok=True)
        (Path(run) / "doctor.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def format_table(result: dict) -> str:
    width = max(len(c["name"]) for c in result["checks"])
    return "\n".join(f"{c['status'].upper():<5} {c['name']:<{width}}  {c['detail']}" for c in result["checks"])


def exit_code(result: dict) -> int:
    """0 unless torch is missing (1)."""
    torch_ok = any(c["name"] == "torch" and c["status"] == OK for c in result["checks"])
    return 0 if torch_ok else 1
