#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    print("[1/2] Running unit and regression tests")
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT, check=True)

    print("[2/2] Running real-image benchmark and building results/report.html")
    subprocess.run([sys.executable, "scripts/generate_report.py"], cwd=ROOT, check=True)

    required = [
        ROOT / "results" / "new_python_result.png",
        ROOT / "results" / "new_python_kernel.png",
        ROOT / "results" / "report.json",
        ROOT / "results" / "report.html",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Docker test did not create required result files: {missing}")

    print("Docker validation complete. Open results/report.html for the visual comparison.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
