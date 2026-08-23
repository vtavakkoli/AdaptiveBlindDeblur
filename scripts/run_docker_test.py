#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_IMAGES = 23


def main() -> int:
    print("[1/3] Preparing the complete official image dataset", flush=True)
    subprocess.run([sys.executable, "dataset/prepare_dataset.py"], cwd=ROOT, check=True)
    images = sorted((ROOT / "dataset" / "image").glob("*.png"))
    if len(images) != EXPECTED_IMAGES:
        raise RuntimeError(f"dataset preparation produced {len(images)} images; expected {EXPECTED_IMAGES}")

    print("[2/3] Running unit and regression tests", flush=True)
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT, check=True)

    print("[3/3] Running 23-image × 3-method benchmark and building results/report.html", flush=True)
    subprocess.run([sys.executable, "scripts/generate_report.py"], cwd=ROOT, check=True)

    report_json = ROOT / "results" / "report.json"
    required = [ROOT / "results" / "report.html", report_json]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Docker test did not create required report files: {missing}")

    report = json.loads(report_json.read_text(encoding="utf-8"))
    if report["dataset"]["image_count"] != EXPECTED_IMAGES:
        raise RuntimeError("report.json does not contain all dataset images")
    if len(report["images"]) != EXPECTED_IMAGES:
        raise RuntimeError("report.json per-image section is incomplete")

    for case in report["images"]:
        case_dir = ROOT / "results" / case["result_dir"]
        for name in ("input.png", "baseline.png", "annealed_pnp.png", "extreme_channel.png", "kernel.png"):
            if not (case_dir / name).is_file():
                raise RuntimeError(f"missing generated output: {case_dir / name}")

    print(
        "Docker validation complete: 23 images, 69 restorations, 3 methods. "
        "Open results/report.html for the full comparison.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
