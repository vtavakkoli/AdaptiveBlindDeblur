#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_IMAGES = 23
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
METHODS = {"baseline", "annealed_pnp", "extreme_channel"}


def source_images() -> list[Path]:
    dataset_dir = ROOT / "dataset" / "image"
    if not dataset_dir.is_dir():
        raise RuntimeError(f"missing dataset directory: {dataset_dir}")
    return sorted(
        path
        for path in dataset_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED
    )


def validate_report(report: dict) -> None:
    dataset = report.get("dataset", {})
    if dataset.get("image_count") != EXPECTED_IMAGES:
        raise RuntimeError("report.json does not contain all 23 dataset images")
    if dataset.get("native_resolution") is not True:
        raise RuntimeError("benchmark is not marked as native resolution")
    if dataset.get("resizing_applied") is not False:
        raise RuntimeError("benchmark reports that resizing was applied")
    if set(report.get("aggregate", {})) != METHODS:
        raise RuntimeError("report.json does not contain all three methods")

    cases = report.get("images", [])
    if len(cases) != EXPECTED_IMAGES:
        raise RuntimeError(f"report contains {len(cases)} image cases; expected {EXPECTED_IMAGES}")

    for case in cases:
        source_shape = case.get("source_shape")
        output_shapes = case.get("output_shapes", {})
        if case.get("native_resolution") is not True or case.get("resizing_applied") is not False:
            raise RuntimeError(f"resolution invariant not recorded for {case.get('name')}")
        for method in METHODS:
            if output_shapes.get(method) != source_shape:
                raise RuntimeError(
                    f"{method} changed image dimensions for {case.get('name')}: "
                    f"{output_shapes.get(method)} != {source_shape}"
                )

        case_dir = ROOT / "results" / case["result_dir"]
        required = {
            case["source_copy"],
            "baseline.png",
            "annealed_pnp.png",
            "extreme_channel.png",
            "interim.png",
            "kernel.png",
        }
        for name in required:
            if not (case_dir / name).is_file():
                raise RuntimeError(f"missing generated output: {case_dir / name}")

        historical = case.get("historical_matlab", {})
        if historical.get("result_copy") and not (case_dir / historical["result_copy"]).is_file():
            raise RuntimeError(f"missing historical reference copy for {case.get('name')}")
        if historical.get("kernel_exact_shape_match") and not (
            case_dir / "historical_matlab_kernel.png"
        ).is_file():
            raise RuntimeError(f"missing historical kernel copy for {case.get('name')}")


def main() -> int:
    print("[1/4] Validating the complete committed dataset/image folder", flush=True)
    images = source_images()
    if len(images) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"dataset/image contains {len(images)} supported images; expected {EXPECTED_IMAGES}"
        )

    print("[2/4] Running Python unit and regression tests", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"],
        cwd=ROOT,
        check=True,
    )

    print(
        "[3/4] Running 23 native-resolution images × 3 methods; no resizing is permitted",
        flush=True,
    )
    subprocess.run(
        [sys.executable, "scripts/generate_report.py"],
        cwd=ROOT,
        check=True,
    )

    print("[4/4] Verifying report contract, dimensions, and generated files", flush=True)
    report_json = ROOT / "results" / "report.json"
    required_reports = [
        ROOT / "results" / "report.html",
        report_json,
        ROOT / "results" / "SUMMARY.md",
    ]
    missing = [str(path) for path in required_reports if not path.is_file()]
    if missing:
        raise RuntimeError(f"Docker test did not create required report files: {missing}")

    report = json.loads(report_json.read_text(encoding="utf-8"))
    validate_report(report)

    historical = report.get("historical_reference", {})
    print(
        "Docker validation complete: "
        f"{EXPECTED_IMAGES} native-resolution sources, "
        f"{EXPECTED_IMAGES * len(METHODS)} restorations, "
        f"{EXPECTED_IMAGES} shared PSFs, "
        f"{historical.get('exact_shape_result_matches', 0)} exact-shape MATLAB references. "
        "Open results/report.html for the complete comparison.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
