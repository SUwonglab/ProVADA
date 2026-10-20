"""Exercise the actual shell collector against legacy and current CSV layouts."""
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest


def run_collector(tmp_path):
    source = Path(__file__).resolve().parents[2] / "collect_finished_runs.sh"
    script = tmp_path / "collect_finished_runs.sh"
    shutil.copy2(source, script)
    return subprocess.run(["bash", str(script)], cwd=tmp_path, capture_output=True,
                          text=True, check=True, timeout=20)


@pytest.mark.skipif(shutil.which("zip") is None, reason="collector requires zip")
def test_collector_finds_iteration_by_header_across_csv_layouts(tmp_path):
    cases = {
        "legacy": ("sequence_hash,selected_count,iteration\nold,1,99\n", True),
        "current": ("trajectory_schema_version,iteration,selected_sequence_hash,selected_count\n2,99,new,0\n", True),
        "first_column": ("iteration,selected_count\n99,0\n", True),
        "quoted_comma": ('metadata,iteration,selected_count\n"one,two",99,0\n', True),
        "quoted_crlf": ('"iteration","selected_count"\r\n"99.0","0"\r\n', True),
        "bom": ("\ufeffiteration,selected_count\n99,0\n", True),
        "invalid_then_done": ("iteration,selected_count\nbad,0\n,0\n99,0\n", True),
        "incomplete": ("iteration,selected_count\n98,99\n", False),
        "initialization_only": ("iteration,selected_count\n-1,99\n", False),
        "missing_iteration": ("sequence_hash,selected_count\nnone,99\n", False),
        "empty": ("", False),
    }
    results = tmp_path / "results"
    for name, (contents, _) in cases.items():
        run = results / name
        run.mkdir(parents=True)
        (run / "trajectory.csv").write_bytes(contents.encode("utf-8"))
    result = run_collector(tmp_path)
    outputs = [path for path in tmp_path.glob("fmc63_deimmunize_caliby_*") if path.is_dir()]
    assert len(outputs) == 1, result.stdout
    expected = {name for name, (_, complete) in cases.items() if complete}
    assert {path.name for path in outputs[0].iterdir()} == expected
    for name in expected:
        assert (outputs[0] / name / "trajectory.csv").read_bytes() == (results / name / "trajectory.csv").read_bytes()
    archives = list(tmp_path.glob("fmc63_deimmunize_caliby_*.zip"))
    assert len(archives) == 1 and zipfile.is_zipfile(archives[0])


@pytest.mark.parametrize("has_incomplete_run", [False, True])
def test_collector_leaves_no_archive_when_no_run_is_complete(tmp_path, has_incomplete_run):
    results = tmp_path / "results"
    results.mkdir()
    if has_incomplete_run:
        run = results / "incomplete"
        run.mkdir()
        (run / "trajectory.csv").write_text("iteration,selected_count\n98,99\n")
    result = run_collector(tmp_path)
    assert "No completed runs found" in result.stdout
    assert list(tmp_path.glob("fmc63_deimmunize_caliby_*")) == []
