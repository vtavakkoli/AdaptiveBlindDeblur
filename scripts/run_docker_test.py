#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_IMAGES = 23
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
METHODS = {
    "baseline",
    "motion_constrained",
    "annealed_pnp",
    "extreme_channel",
    "rgac",
}
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and validate the complete Docker benchmark.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Force a clean regeneration of all benchmark outputs. This makes "
            "`docker compose run --rm test --rebuild` a supported project command. "
            "To rebuild the Docker image itself, also pass Compose's --build before the service name."
        ),
    )
    return parser


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
        raise RuntimeError("report.json does not contain all five benchmark methods")
    if set(report.get("method_order", [])) != METHODS:
        raise RuntimeError("report.json method_order is incomplete")
    if report.get("overall_winner") not in METHODS:
        raise RuntimeError("report.json does not name a valid overall winner")

    winner_selection = report.get("winner_selection", {})
    if winner_selection.get("reference_free") is not True:
        raise RuntimeError("winner selection must remain reference-free")
    if winner_selection.get("legacy_inputs_used") is not False:
        raise RuntimeError("legacy assets must not be used to choose a winner")
    if winner_selection.get("legacy_metrics_used") is not False:
        raise RuntimeError("legacy metrics must not be used to choose a winner")

    kernel_comparison = report.get("kernel_comparison", {})
    if kernel_comparison.get("diagnostic_refits_used_for_restoration") is not False:
        raise RuntimeError("diagnostic refinement kernels must not be fed back into restoration")
    if kernel_comparison.get("legacy_inputs_used") is not False:
        raise RuntimeError("legacy kernels must remain evaluation-only")

    rgac = report.get("rgac_design", {})
    if rgac.get("learned_weights") is not False:
        raise RuntimeError("RGAC must remain a non-learned deterministic method")
    if rgac.get("legacy_inputs_used") is not False:
        raise RuntimeError("RGAC must not use legacy assets as inference inputs")

    legacy = report.get("legacy_comparison", {})
    if legacy.get("role") != "evaluation_only":
        raise RuntimeError("legacy outputs must be evaluation-only")
    if legacy.get("resampling_permitted") is not False:
        raise RuntimeError("legacy reference resampling must remain disabled")

    aggregate = report.get("aggregate", {})
    if sum(int(aggregate[key].get("win_count", 0)) for key in METHODS) != EXPECTED_IMAGES:
        raise RuntimeError("aggregate per-image win counts do not sum to 23")

    cases = report.get("images", [])
    if len(cases) != EXPECTED_IMAGES:
        raise RuntimeError(f"report contains {len(cases)} image cases; expected {EXPECTED_IMAGES}")

    for case in cases:
        source_shape = case.get("source_shape")
        output_shapes = case.get("output_shapes", {})
        kernel_shapes = case.get("kernel_shapes", {})
        metrics = case.get("metrics", {})
        if case.get("native_resolution") is not True or case.get("resizing_applied") is not False:
            raise RuntimeError(f"resolution invariant not recorded for {case.get('name')}")
        if case.get("winner") not in METHODS:
            raise RuntimeError(f"invalid per-image winner for {case.get('name')}")

        profile = case.get("profile", {})
        kernel_size = int(profile.get("kernel_size", 0))
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise RuntimeError(f"invalid benchmark kernel support for {case.get('name')}: {kernel_size}")
        expected_kernel_shape = [kernel_size, kernel_size]

        for method in METHODS:
            if output_shapes.get(method) != source_shape:
                raise RuntimeError(
                    f"{method} changed image dimensions for {case.get('name')}: "
                    f"{output_shapes.get(method)} != {source_shape}"
                )
            if kernel_shapes.get(method) != expected_kernel_shape:
                raise RuntimeError(
                    f"{method} kernel shape mismatch for {case.get('name')}: "
                    f"{kernel_shapes.get(method)} != {expected_kernel_shape}"
                )
            row = metrics.get(method, {})
            score = float(row.get("reference_free_score", -1.0))
            if not 0.0 <= score <= 1.0:
                raise RuntimeError(f"invalid reference-free score for {method}/{case.get('name')}")
            kernel_info = row.get("kernel", {})
            if int(kernel_info.get("component_count", -1)) < 0:
                raise RuntimeError(f"invalid PSF component count for {method}/{case.get('name')}")
            if float(kernel_info.get("plausibility_score", -1.0)) < 0.0:
                raise RuntimeError(f"invalid PSF plausibility score for {method}/{case.get('name')}")

        expected_saturated = case.get("name") in SATURATED_CASES
        if bool(profile.get("saturated", False)) is not expected_saturated:
            raise RuntimeError(f"incorrect saturation mode for {case.get('name')}")

        rgac_info = case.get("rgac", {})
        confidence = float(rgac_info.get("psf_confidence", -1.0))
        if not 0.0 <= confidence <= 1.0:
            raise RuntimeError(f"invalid RGAC PSF confidence for {case.get('name')}")
        if set(rgac_info.get("candidate_scores", {})) != {
            "baseline",
            "conservative",
            "annealed_pnp",
            "extreme_channel",
        }:
            raise RuntimeError(f"RGAC candidate diagnostics missing for {case.get('name')}")

        case_dir = ROOT / "results" / case["result_dir"]
        required = {
            case["source_copy"],
            "interim.png",
            "interim_motion.png",
            "kernel.png",
        }
        for method in METHODS:
            required.add(f"{method}.png")
            required.add(f"{method}_kernel.png")
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.rebuild:
        print(
            "Clean benchmark regeneration requested via --rebuild. "
            "The report generator clears prior results before processing.",
            flush=True,
        )

    print("[1/4] Validating dataset and explicit full-quality profiles", flush=True)
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
        raise RuntimeError("benchmark saturation modes do not match the validated dataset configuration")

    print("[2/4] Running Python unit, parity, artifact-safety, motion, and RGAC regression tests", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"],
        cwd=ROOT,
        check=True,
    )

    print(
        "[3/4] Running 23 native-resolution images × 5 methods with per-method PSF inspection",
        flush=True,
    )
    subprocess.run(
        [sys.executable, "scripts/generate_best_report.py"],
        cwd=ROOT,
        check=True,
    )

    print("[4/4] Verifying winner, dimensions, five PSFs per image, and generated files", flush=True)
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

    winner = report["overall_winner"]
    winner_name = report["methods"][winner]["name"]
    legacy = report.get("legacy_comparison", {})
    print(
        "Docker validation complete: "
        f"{EXPECTED_IMAGES} native-resolution sources, "
        f"{EXPECTED_IMAGES * len(METHODS)} restorations, "
        f"{EXPECTED_IMAGES * len(METHODS)} PSF visualizations, "
        f"overall reference-free winner={winner_name}, "
        f"{legacy.get('exact_shape_results', 0)} legacy outputs compared. "
        "Open results/report.html for the complete five-method comparison.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
