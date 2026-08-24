from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_browser_lab_inline_javascript_parses(tmp_path: Path) -> None:
    """Catch browser-only syntax failures before GitHub Pages deployment."""
    page = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    match = re.search(r"<script>([\s\S]*?)</script>", page)
    assert match is not None, "docs/index.html must contain its standalone inline script"

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable in this runtime; GitHub-hosted quality gate runs this check")

    script = tmp_path / "browser-lab.js"
    script.write_text(match.group(1), encoding="utf-8")
    result = subprocess.run(
        [node, "--check", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
