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
SATURATED_CASES = {
    "26.png",
    "IMG_0650_small_patch.png",
    "IMG_0664_small_patch.png",
    "IMG_4548_small.png",
    "IMG_4561.JPG",
    "blurry_2_small.png",
    "blurry_7.png",
    "my_test_car6.png",
}


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
    if dataset.get("profile_file") != "dataset/benchmark_profiles.json":
        raise RuntimeError("benchmark did not record the explicit profile file")
    if set(report.get("aggregate", {})) != METHODS:
        raise RuntimeError("report.json does not contain all three methods")

    legacy = report.get("legacy_comparison", {})
    if legacy.get("role") != "evaluation_only":
        raise RuntimeError("legacy outputs must be evaluation-only")
    if legacy.get("resampling_permitted") is not False:
        raise RuntimeError("legacy reference resampling must remain disabled")

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

        profile = case.get("profile", {})
        kernel_size = int(profile.get("kernel_size", 0))
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise RuntimeError(f"invalid benchmark kernel support for {case.get('name')}: {kernel_size}")
        if case.get("kernel_shape") != [kernel_size, kernel_size]:
            raise RuntimeError(
                f"estimated kernel shape does not match profile for {case.get('name')}: "
                f"{case.get('kernel_shape')} != {[kernel_size, kernel_size]}"
            )
        expected_saturated = case.get("name") in SATURATED_CASES
        if bool(profile.get("saturated", False)) is not expected_saturated:
            raise RuntimeError(f"incorrect MATLAB saturation mode for {case.get('name')}")

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

        if case.get("legacy_reference_status") == "exact_shape" and not (
            case_dir / "legacy_result.png"
        ).is_file():
            raise RuntimeError(f"missing legacy result copy for {case.get('name')}")
        if case.get("legacy_kernel_shape") is not None and not (
            case_dir / "legacy_kernel.png"
        ).is_file():
            raise RuntimeError(f"missing legacy kernel copy for {case.get('name')}")


def main() -> int:
    print("[1/4] Validating dataset and MATLAB-equivalent benchmark profiles", flush=True)
    images = source_images()
    if len(images) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"dataset/image contains {len(images)} supported images; expected {EXPECTED_IMAGES}"
        )
    profiles = json.loads((ROOT / "dataset" / "benchmark_profiles.json").read_text(encoding="utf-8"))
    if set(profiles) != {path.name for path in images}:
        raise RuntimeError("benchmark_profiles.json must define exactly one profile for every source image")
    configured_saturated = {name for name, profile in profiles.items() if profile.get("saturated")}
    if configured_saturated != SATURATED_CASES:
        raise RuntimeError(
            "benchmark saturation modes do not match the original MATLAB demo configuration"
        )

    print("[2/4] Running Python unit and MATLAB-parity regression tests", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"],
        cwd=ROOT,
        check=True,
    )

    print(
        "[3/4] Running 23 MATLAB-parity native-resolution images × 3 methods; no resizing is permitted",
        flush=True,
    )
    subprocess.run(
        [sys.executable, "scripts/generate_matlab_parity_report.py"],
        cwd=ROOT,
        check=True,
    )

    print("[4/4] Verifying report contract, dimensions, PSFs, modes, and generated files", flush=True)
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

    legacy = report.get("legacy_comparison", {})
    print(
        "Docker validation complete: "
        f"{EXPECTED_IMAGES} native-resolution sources, "
        f"{EXPECTED_IMAGES * len(METHODS)} restorations, "
        f"{EXPECTED_IMAGES} independently estimated PSFs, "
        f"{legacy.get('exact_shape_results', 0)} legacy outputs compared. "
        "Open results/report.html for the complete MATLAB-parity comparison.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
