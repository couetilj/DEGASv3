import json
import re
import subprocess

import pytest

from degas3 import cli
from degas3.config import RunConfig
from degas3.slurm import build_script, parse_times


@pytest.fixture()
def run(tmp_path):
    root = tmp_path / "my run"  # a space exercises quoting
    RunConfig().save(root)
    return root


def test_script_chain_and_dependencies(run):
    (run / "prepared").mkdir()
    (run / "prepared" / "tasks.json").write_text(json.dumps([{"task": i} for i in range(5)]))
    s = build_script(run, account="acct", partition="gpuq", cpu_partition="cpuq", gpus=2, venv="/some/venv", modules="python/3.11")
    stages = ["prepare", "train", "select", "fit", "score", "stratify", "de", "pathways", "report"]
    assert [m for m in re.findall(r"^J_(\w+)=\$\(submit (\w+)", s, re.M)] == [(x.upper(), x) for x in stages]
    sub = [ln for ln in s.splitlines() if ln.startswith("J_")]
    assert "afterok" in s and s.count("sbatch") == 1  # one helper that all jobs use
    assert 'submit train gpu "0-4" "$J_PREPARE"' in sub[1]
    assert '"$J_PATHWAYS"' in sub[-1] and 'submit prepare cpu "" ""' in sub[0]
    assert "--gres=gpu:2" in s and "--partition=gpuq" in s and "--partition=cpuq" in s and "--account=acct" in s
    assert "--parsable" in s and "--dependency=\"afterok:$dep\"" in s
    assert "module load python/3.11" in s and "/some/venv/bin/activate" in s
    assert "'" + str(run) + "'" in s or str(run) in s
    assert "$SLURM_ARRAY_TASK_ID" in s and "logs" in s
    bash = subprocess.run(["bash", "-n"], input=s, text=True, capture_output=True)
    assert bash.returncode == 0, bash.stderr


def test_no_tasks_json_uses_all_and_no_hardcoded_cluster_paths(run):
    s = build_script(run, account="a", partition="p")
    assert "train" in s and "--all" in s and "--task" not in s
    assert not re.search(r"/(N|scratch|home|project|gpfs)/", s.replace(str(run), ""))
    assert "J_EXPLAIN" not in s
    assert "J_EXPLAIN" in build_script(run, account="a", partition="p", explain=True)


def test_parse_times_and_validation(run):
    assert parse_times("train=12:00:00")["train"] == "12:00:00" and parse_times(None)["fit"] == "08:00:00"
    assert set(parse_times("02:00:00").values()) == {"02:00:00"}
    with pytest.raises(ValueError, match="unknown stage"):
        parse_times("bogus=01:00:00")
    with pytest.raises(ValueError, match="unsupported characters"):
        build_script(run, account="a; rm -rf /", partition="p")


def test_cli_writes_files_and_prints(run, capsys, monkeypatch):
    monkeypatch.setenv("DEGAS3_CPU_PARTITION", "envcpu")
    assert cli.main(["slurm", str(run), "--account", "a", "--partition", "g"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("#!/usr/bin/env bash") and "--partition=envcpu" in out
    assert (run / "slurm" / "jobs.sh").read_text() == out
    assert "Resume after a failure" in (run / "slurm" / "README.md").read_text()
    with pytest.raises(ValueError, match="not a run directory"):
        from degas3.slurm import cli_slurm
        import argparse
        cli_slurm(argparse.Namespace(run=str(run / "nope"), account="a", partition="p"))
