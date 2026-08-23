#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_IMAGES = 23
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main() -> int:
    print("[1/3] Validating the complete committed dataset/image folder", flush=True)
    dataset_dir = ROOT / "dataset" / "image"
    images = sorted(p for p in dataset_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED)
    if len(images) != EXPECTED_IMAGES:
        raise RuntimeError(f"dataset/image contains {len(images)} supported images; expected {EXPECTED_IMAGES}")

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
    if report["dataset"]["image_count"] != EXPECTED_IMAGES or len(report["images"]) != EXPECTED_IMAGES:
        raise RuntimeError("report.json does not contain all 23 dataset images")
    if set(report["aggregate"]) != {"baseline", "annealed_pnp", "extreme_channel"}:
        raise RuntimeError("report.json does not contain all three methods")

    for case in report["images"]:
        case_dir = ROOT / "results" / case["result_dir"]
        for name in ("input.png", "baseline.png", "annealed_pnp.png", "extreme_channel.png", "kernel.png"):
            if not (case_dir / name).is_file():
                raise RuntimeError(f"missing generated output: {case_dir / name}")

    print(
        "Docker validation complete: 23 source images, 69 restorations, 23 shared PSFs. "
        "Open results/report.html for the full comparison.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
