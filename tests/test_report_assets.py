from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_docker_report_workflow_files_exist() -> None:
    required = [
        ROOT / "scripts" / "run_docker_test.py",
        ROOT / "scripts" / "generate_report.py",
        ROOT / "results" / ".gitkeep",
    ]
    assert all(path.is_file() for path in required)
