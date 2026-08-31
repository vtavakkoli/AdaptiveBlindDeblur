from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_browser_lab_javascript_parses() -> None:
    """Catch browser-only syntax failures before GitHub Pages deployment."""
    script = ROOT / "docs" / "browser-lab.js"
    assert script.is_file(), "docs/browser-lab.js must exist"

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable in this runtime; GitHub-hosted quality gate runs this check")

    result = subprocess.run(
        [node, "--check", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
