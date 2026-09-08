from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("filename", ["browser-lab.js", "deblur-core.js", "deblur-worker.js"])
def test_browser_lab_javascript_parses(filename: str) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable; the GitHub quality gate runs these checks")
    subprocess.run([node, "--check", str(ROOT / "docs" / filename)], check=True, capture_output=True, text=True)


def test_browser_numerical_quality_and_worker_protocol() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable; the GitHub quality gate runs these checks")
    result = subprocess.run(
        [node, "--test", str(ROOT / "tests" / "browser-quality.cjs")],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
