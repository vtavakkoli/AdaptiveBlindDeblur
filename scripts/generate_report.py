#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import platform
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np
import scipy

from dark_channel_deblur import (
    DeblurConfig,
    annealed_pnp_refine,
    deblur_image,
    extreme_channel_refine,
    reblur_image,
)
from dark_channel_deblur.io import read_image, write_image

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset" / "image"
HISTORICAL_RESULTS = ROOT / "dataset" / "results"
RESULTS = ROOT / "results"
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EXPECTED_IMAGES = 23
DEFAULT_KERNEL_SIZE = 25

METHODS = {
    "baseline": {
        "name": "Dark Channel Baseline",
        "short": "DCP",
        "description": (
            "Optimized Python port of Pan et al. CVPR 2016: blind PSF estimation "
            "followed by TV/L0 non-blind restoration."
        ),
    },
    "annealed_pnp": {
        "name": "Annealed Gaussian PnP",
        "short": "A-PnP",
        "description": (
            "Diffusion-inspired, weight-free refinement using a Gaussian noise schedule, "
            "NLM denoising, and explicit FFT measurement consistency."
        ),
    },
    "extreme_channel": {
        "name": "Extreme-Channel Guided",
        "short": "ECP-R",
        "description": (
            "Dark + bright local-extrema guided detail refinement with explicit FFT "
            "measurement consistency."
        ),
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the native-resolution three-method deblurring benchmark."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS,
        help="Directory for report.html, report.json, SUMMARY.md, and per-image outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Developer-only smoke-test limit. Docker validation intentionally does not use this.",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=None,
        help=(
            "Force one odd kernel size for every image. By default, use the historical "
            "MATLAB kernel size when an exact reference exists, otherwise 25."
        ),
    )
    return parser


def dataset_images() -> list[Path]:
    if not DATASET.is_dir():
        raise RuntimeError(f"dataset directory is missing: {DATASET}")
    return sorted(
        path
        for path in DATASET.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED
    )


def slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("._") or "image"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr


def sharpness(image: np.ndarray) -> float:
    g = gray(image)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))


def noise_mad(image: np.ndarray) -> float:
    lap = cv2.Laplacian(gray(image), cv2.CV_32F)
    med = float(np.median(lap))
    return float(np.median(np.abs(lap - med)))


def extreme_metrics(image: np.ndarray, patch: int = 9) -> tuple[float, float]:
    arr = np.asarray(image, dtype=np.float32)
    footprint = np.ones((patch, patch), dtype=np.uint8)
    if arr.ndim == 2:
        local_dark = cv2.erode(arr, footprint)
        local_bright = cv2.dilate(arr, footprint)
    else:
        local_dark = cv2.erode(np.min(arr, axis=2), footprint)
        local_bright = cv2.dilate(np.max(arr, axis=2), footprint)
    return float(np.mean(local_dark < 0.03)), float(np.mean(local_bright > 0.97))


def psnr(candidate: np.ndarray, reference: np.ndarray) -> float:
    if candidate.shape != reference.shape:
        raise ValueError(f"PSNR shape mismatch: {candidate.shape} != {reference.shape}")
    mse = float(
        np.mean(
            (
                np.asarray(candidate, dtype=np.float64)
                - np.asarray(reference, dtype=np.float64)
            )
            ** 2
        )
    )
    return 120.0 if mse <= 1e-12 else float(10.0 * math.log10(1.0 / mse))


def ssim(candidate: np.ndarray, reference: np.ndarray) -> float:
    a = np.asarray(candidate, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"SSIM shape mismatch: {a.shape} != {b.shape}")
    if a.ndim == 2:
        a, b = a[..., None], b[..., None]

    c1 = 0.01**2
    c2 = 0.03**2
    scores: list[float] = []
    for channel in range(a.shape[2]):
        x = a[..., channel]
        y = b[..., channel]
        mx = cv2.GaussianBlur(x, (11, 11), 1.5)
        my = cv2.GaussianBlur(y, (11, 11), 1.5)
        vx = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mx * mx
        vy = cv2.GaussianBlur(y * y, (11, 11), 1.5) - my * my
        vxy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mx * my
        score = ((2 * mx * my + c1) * (2 * vxy + c2)) / (
            (mx * mx + my * my + c1) * (vx + vy + c2) + 1e-12
        )
        core = score[5:-5, 5:-5] if min(score.shape) > 12 else score
        scores.append(float(np.mean(core)))
    return float(np.mean(scores))


def kernel_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float] | None:
    a = np.asarray(candidate, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape:
        return None
    a /= max(float(a.sum()), 1e-12)
    b /= max(float(b.sum()), 1e-12)
    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    return {
        "correlation": correlation,
        "l1_distance": float(np.abs(a - b).sum()),
    }


def historical_paths(source: Path) -> tuple[Path | None, Path | None]:
    result = HISTORICAL_RESULTS / f"{source.stem}_result.png"
    kernel = HISTORICAL_RESULTS / f"{source.stem}_kernel.png"
    return (result if result.is_file() else None, kernel if kernel.is_file() else None)


def read_kernel(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    arr = image.astype(np.float32) / 255.0
    total = float(arr.sum())
    return arr / total if total > 0 else None


def choose_kernel_size(
    reference_kernel: np.ndarray | None,
    forced_kernel_size: int | None,
) -> tuple[int, str]:
    if forced_kernel_size is not None:
        if forced_kernel_size < 3 or forced_kernel_size % 2 == 0:
            raise ValueError("--kernel-size must be an odd integer >= 3")
        return forced_kernel_size, "command_line"

    if reference_kernel is not None:
        h, w = reference_kernel.shape[:2]
        if h == w and h >= 3 and h % 2 == 1 and h <= 75:
            return int(h), "historical_matlab_kernel_shape"

    return DEFAULT_KERNEL_SIZE, "default"


def diagnostics(
    output: np.ndarray,
    observed: np.ndarray,
    kernel: np.ndarray,
    *,
    workers: int,
    historical_reference: np.ndarray | None,
) -> dict[str, float | None]:
    predicted = reblur_image(output, kernel, workers=workers)
    dark_fraction, bright_fraction = extreme_metrics(output)
    row: dict[str, float | None] = {
        "reblur_rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
        "sharpness": sharpness(output),
        "noise_mad": noise_mad(output),
        "dark_fraction": dark_fraction,
        "bright_fraction": bright_fraction,
        "psnr_vs_historical_matlab_db": None,
        "ssim_vs_historical_matlab": None,
    }
    if historical_reference is not None:
        row["psnr_vs_historical_matlab_db"] = psnr(output, historical_reference)
        row["ssim_vs_historical_matlab"] = ssim(output, historical_reference)
    return row


def pct_change(value: float, baseline: float, *, lower_is_better: bool = False) -> float:
    if abs(baseline) < 1e-12:
        return 0.0
    if lower_is_better:
        return 100.0 * (baseline - value) / baseline
    return 100.0 * (value - baseline) / baseline


def mean_optional(rows: list[dict[str, float | None]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(values) if values else None


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def pct(value: float) -> str:
    return f"{value:+.1f}%"


def shape_text(shape: list[int] | tuple[int, ...]) -> str:
    if len(shape) == 3:
        return f"{shape[1]}×{shape[0]}×{shape[2]}"
    return f"{shape[1]}×{shape[0]}"


def card(title: str, image_path: str, subtitle: str, badge: str = "") -> str:
    tag = f'<span class="badge">{html.escape(badge)}</span>' if badge else ""
    return (
        '<article class="card"><div class="card-head"><div>'
        f"<h4>{html.escape(title)}</h4><p>{html.escape(subtitle)}</p></div>{tag}</div>"
        f'<img loading="lazy" src="{html.escape(image_path)}" alt="{html.escape(title)}"></article>'
    )


def method_table(
    rows: dict[str, dict[str, float | None]],
    runtimes: dict[str, float],
) -> str:
    baseline = rows["baseline"]
    baseline_rmse = float(baseline["reblur_rmse"])
    baseline_sharp = float(baseline["sharpness"])
    baseline_noise = float(baseline["noise_mad"])
    body: list[str] = []

    for key in ("baseline", "annealed_pnp", "extreme_channel"):
        values = rows[key]
        rmse = float(values["reblur_rmse"])
        sharp = float(values["sharpness"])
        noise = float(values["noise_mad"])
        body.append(
            "<tr>"
            f"<td><strong>{html.escape(METHODS[key]['name'])}</strong></td>"
            f"<td>{runtimes[key]:.3f}s</td>"
            f"<td>{fmt(rmse, 5)}</td>"
            f"<td>{pct(pct_change(rmse, baseline_rmse, lower_is_better=True))}</td>"
            f"<td>{fmt(sharp, 4)}</td>"
            f"<td>{pct(pct_change(sharp, baseline_sharp))}</td>"
            f"<td>{fmt(noise, 5)}</td>"
            f"<td>{pct(pct_change(noise, baseline_noise))}</td>"
            f"<td>{fmt(values['psnr_vs_historical_matlab_db'], 2)}</td>"
            f"<td>{fmt(values['ssim_vs_historical_matlab'], 4)}</td>"
            "</tr>"
        )
    return "".join(body)


def clean_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.name == ".gitkeep":
            continue
        shutil.rmtree(path) if path.is_dir() else path.unlink()


def generate_report(
    output_dir: Path = RESULTS,
    *,
    limit: int | None = None,
    forced_kernel_size: int | None = None,
) -> Path:
    images = dataset_images()
    if len(images) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"dataset/image contains {len(images)} supported images; expected {EXPECTED_IMAGES}"
        )
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be >= 1")
        images = images[:limit]

    clean_output_dir(output_dir)
    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    common_config = {
        "lambda_dark": 4e-3,
        "lambda_grad": 4e-3,
        "gamma_correct": 1.0,
        "xk_iter": 2,
        "lambda_tv": 3e-3,
        "lambda_l0": 5e-4,
        "weight_ring": 1.0,
        "dark_patch_size": 35,
        "max_grad_steps": 8,
        "max_dark_steps": 3,
        "fft_workers": -1,
    }
    refinement_config = {
        "annealed_pnp": {
            "steps": 3,
            "sigma_start": 0.025,
            "sigma_end": 0.004,
            "candidates": 1,
        },
        "extreme_channel": {
            "steps": 3,
            "patch_size": 15,
        },
    }

    per_image: list[dict[str, Any]] = []
    metric_sets: dict[str, list[dict[str, float | None]]] = {
        key: [] for key in METHODS
    }
    runtime_sets: dict[str, list[float]] = {key: [] for key in METHODS}
    historical_candidates = 0
    historical_matches = 0
    historical_kernel_matches = 0
    total_started = time.perf_counter()

    for index, source in enumerate(images, 1):
        print(
            f"[{index:02d}/{len(images):02d}] native-resolution benchmark: {source.name}",
            flush=True,
        )
        observed = read_image(source)
        source_shape = list(observed.shape)
        case_dir = images_out / f"{index:02d}_{slugify(source.stem)}"
        case_dir.mkdir(parents=True, exist_ok=True)

        source_copy = f"input{source.suffix.lower()}"
        shutil.copy2(source, case_dir / source_copy)

        historical_result_path, historical_kernel_path = historical_paths(source)
        historical_kernel = read_kernel(historical_kernel_path)
        kernel_size, kernel_size_source = choose_kernel_size(
            historical_kernel,
            forced_kernel_size,
        )

        historical_reference: np.ndarray | None = None
        historical_reference_file: str | None = None
        historical_status = "not_available"
        if historical_result_path is not None:
            historical_candidates += 1
            candidate = read_image(historical_result_path)
            if candidate.shape == observed.shape:
                historical_reference = candidate
                historical_reference_file = "historical_matlab.png"
                shutil.copy2(historical_result_path, case_dir / historical_reference_file)
                historical_status = "exact_shape_match"
                historical_matches += 1
            else:
                historical_status = (
                    "shape_mismatch:"
                    f"{tuple(candidate.shape)}!={tuple(observed.shape)}"
                )

        cfg = DeblurConfig(kernel_size=kernel_size, **common_config)

        started = time.perf_counter()
        baseline, kernel, interim = deblur_image(observed, cfg)
        baseline_time = time.perf_counter() - started

        started = time.perf_counter()
        annealed = annealed_pnp_refine(
            observed,
            baseline,
            kernel,
            steps=int(refinement_config["annealed_pnp"]["steps"]),
            sigma_start=float(refinement_config["annealed_pnp"]["sigma_start"]),
            sigma_end=float(refinement_config["annealed_pnp"]["sigma_end"]),
            candidates=int(refinement_config["annealed_pnp"]["candidates"]),
            seed=index,
            workers=cfg.fft_workers,
        )
        annealed_time = time.perf_counter() - started

        started = time.perf_counter()
        extreme = extreme_channel_refine(
            observed,
            baseline,
            kernel,
            steps=int(refinement_config["extreme_channel"]["steps"]),
            patch_size=int(refinement_config["extreme_channel"]["patch_size"]),
            workers=cfg.fft_workers,
        )
        extreme_time = time.perf_counter() - started

        outputs = {
            "baseline": baseline,
            "annealed_pnp": annealed,
            "extreme_channel": extreme,
        }
        runtimes = {
            "baseline": baseline_time,
            "annealed_pnp": annealed_time,
            "extreme_channel": extreme_time,
        }

        output_shapes: dict[str, list[int]] = {}
        for method, output in outputs.items():
            if output.shape != observed.shape:
                raise RuntimeError(
                    f"{method} changed resolution for {source.name}: "
                    f"{output.shape} != {observed.shape}"
                )
            if not np.isfinite(output).all():
                raise RuntimeError(f"{method} produced non-finite output for {source.name}")
            output_shapes[method] = list(output.shape)
            write_image(case_dir / f"{method}.png", output)

        if interim.shape[:2] != observed.shape[:2]:
            raise RuntimeError(
                f"interim latent changed spatial resolution for {source.name}: "
                f"{interim.shape} != {observed.shape}"
            )
        write_image(case_dir / "interim.png", interim)
        write_image(
            case_dir / "kernel.png",
            kernel / max(float(kernel.max()), 1e-12),
        )

        rows: dict[str, dict[str, float | None]] = {}
        for method, output in outputs.items():
            row = diagnostics(
                output,
                observed,
                kernel,
                workers=cfg.fft_workers,
                historical_reference=historical_reference,
            )
            rows[method] = row
            metric_sets[method].append(row)
            runtime_sets[method].append(runtimes[method])

        historical_kernel_metrics = None
        if historical_kernel is not None:
            historical_kernel_metrics = kernel_metrics(kernel, historical_kernel)
            if historical_kernel_metrics is not None:
                historical_kernel_matches += 1
                shutil.copy2(
                    historical_kernel_path,
                    case_dir / "historical_matlab_kernel.png",
                )

        baseline_metrics = rows["baseline"]
        gains = {}
        for method, row in rows.items():
            gains[method] = {
                "reblur_rmse_improvement_pct": pct_change(
                    float(row["reblur_rmse"]),
                    float(baseline_metrics["reblur_rmse"]),
                    lower_is_better=True,
                ),
                "sharpness_change_pct": pct_change(
                    float(row["sharpness"]),
                    float(baseline_metrics["sharpness"]),
                ),
                "noise_mad_change_pct": pct_change(
                    float(row["noise_mad"]),
                    float(baseline_metrics["noise_mad"]),
                ),
            }

        per_image.append(
            {
                "index": index,
                "name": source.name,
                "stem": source.stem,
                "source_shape": source_shape,
                "output_shapes": output_shapes,
                "native_resolution": True,
                "resizing_applied": False,
                "source_sha256": sha256_file(source),
                "source_copy": source_copy,
                "result_dir": str(case_dir.relative_to(output_dir)).replace("\\", "/"),
                "kernel_size": kernel_size,
                "kernel_size_source": kernel_size_source,
                "kernel_sum": float(kernel.sum()),
                "kernel_peak": float(kernel.max()),
                "historical_matlab": {
                    "result_source": (
                        str(historical_result_path.relative_to(ROOT))
                        if historical_result_path is not None
                        else None
                    ),
                    "result_status": historical_status,
                    "result_copy": historical_reference_file,
                    "result_sha256": (
                        sha256_file(historical_result_path)
                        if historical_reference is not None
                        and historical_result_path is not None
                        else None
                    ),
                    "kernel_source": (
                        str(historical_kernel_path.relative_to(ROOT))
                        if historical_kernel_path is not None
                        else None
                    ),
                    "kernel_exact_shape_match": historical_kernel_metrics is not None,
                    "kernel_metrics": historical_kernel_metrics,
                },
                "runtimes_seconds": runtimes,
                "metrics": rows,
                "gains_vs_baseline": gains,
            }
        )

    total_runtime = time.perf_counter() - total_started
    aggregate: dict[str, dict[str, float | int | None]] = {}
    for method in METHODS:
        rows = metric_sets[method]
        aggregate[method] = {
            "stage_runtime_seconds_mean": mean(runtime_sets[method]),
            "stage_runtime_seconds_total": sum(runtime_sets[method]),
            "reblur_rmse_mean": mean(float(row["reblur_rmse"]) for row in rows),
            "sharpness_mean": mean(float(row["sharpness"]) for row in rows),
            "noise_mad_mean": mean(float(row["noise_mad"]) for row in rows),
            "dark_fraction_mean": mean(float(row["dark_fraction"]) for row in rows),
            "bright_fraction_mean": mean(float(row["bright_fraction"]) for row in rows),
            "psnr_vs_historical_matlab_db_mean": mean_optional(
                rows, "psnr_vs_historical_matlab_db"
            ),
            "ssim_vs_historical_matlab_mean": mean_optional(
                rows, "ssim_vs_historical_matlab"
            ),
            "historical_reference_count": sum(
                row["psnr_vs_historical_matlab_db"] is not None for row in rows
            ),
        }

    base_agg = aggregate["baseline"]
    base_rmse = float(base_agg["reblur_rmse_mean"])
    base_sharp = float(base_agg["sharpness_mean"])
    base_noise = float(base_agg["noise_mad_mean"])
    base_runtime = float(base_agg["stage_runtime_seconds_total"])

    for method, row in aggregate.items():
        row["reblur_rmse_improvement_vs_baseline_pct"] = pct_change(
            float(row["reblur_rmse_mean"]),
            base_rmse,
            lower_is_better=True,
        )
        row["sharpness_change_vs_baseline_pct"] = pct_change(
            float(row["sharpness_mean"]),
            base_sharp,
        )
        row["noise_mad_change_vs_baseline_pct"] = pct_change(
            float(row["noise_mad_mean"]),
            base_noise,
        )
        stage_total = float(row["stage_runtime_seconds_total"])
        row["end_to_end_runtime_seconds_total"] = (
            stage_total if method == "baseline" else base_runtime + stage_total
        )

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "scipy": scipy.__version__,
        "git_commit": os.getenv("BENCHMARK_GIT_COMMIT", "local"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    report: dict[str, Any] = {
        "schema_version": 2,
        "benchmark": "Native-resolution dataset/image three-method benchmark",
        "dataset": {
            "path": "dataset/image",
            "historical_results_path": "dataset/results",
            "source": (
                "23 source images committed in the repository from the supplied "
                "CVPR 2016 release."
            ),
            "image_count": len(images),
            "expected_full_image_count": EXPECTED_IMAGES,
            "native_resolution": True,
            "resizing_applied": False,
            "resolution_policy": (
                "Every algorithm receives the decoded source image at its original "
                "height and width. No benchmark resizing, cropping, or reference "
                "resampling is permitted."
            ),
        },
        "historical_reference": {
            "meaning": (
                "Same-input historical MATLAB/release outputs found under dataset/results. "
                "PSNR/SSIM measure agreement with those outputs, not ground-truth quality."
            ),
            "candidate_result_files": historical_candidates,
            "exact_shape_result_matches": historical_matches,
            "exact_shape_kernel_matches": historical_kernel_matches,
            "resampling_permitted": False,
        },
        "methods": METHODS,
        "fairness": (
            "Within each source image, the optimized Python baseline estimates one blind "
            "PSF and both new refinements reuse that exact PSF. When a historical MATLAB "
            "kernel exists, its native kernel dimensions select the Python kernel size, "
            "but the historical kernel values are never used by the new methods."
        ),
        "environment": environment,
        "total_runtime_seconds": total_runtime,
        "benchmark_profile": {
            "name": "native-resolution-reproducible",
            "common_baseline_config": common_config,
            "refinement_config": refinement_config,
            "forced_kernel_size": forced_kernel_size,
            "default_kernel_size_without_reference": DEFAULT_KERNEL_SIZE,
        },
        "aggregate": aggregate,
        "images": per_image,
        "metric_notes": {
            "reblur_rmse": (
                "Lower is better measurement consistency: reblur(restored, estimated_kernel) "
                "vs the native-resolution observed source."
            ),
            "sharpness": (
                "Mean Sobel magnitude. Higher means more edge energy but can also reward "
                "ringing/noise."
            ),
            "noise_mad": (
                "Laplacian median absolute deviation. Diagnostic of high-frequency/noise "
                "content, not a perceptual quality score."
            ),
            "psnr_vs_historical_matlab_db": (
                "Agreement with the same-input historical MATLAB/release output only when "
                "the reference has exactly the same dimensions. It is not ground-truth PSNR."
            ),
            "ssim_vs_historical_matlab": (
                "Structural agreement with the same-input historical MATLAB/release output; "
                "not a clean-image ground-truth score."
            ),
        },
    }
    report_json = output_dir / "report.json"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary_lines = [
        "# Native-resolution deblurring benchmark",
        "",
        f"- Images: **{len(images)}**",
        f"- Restored outputs: **{len(images) * 3}**",
        "- Benchmark resizing: **none**",
        f"- Historical MATLAB exact-shape references: **{historical_matches}**",
        f"- Total benchmark runtime: **{total_runtime:.2f}s**",
        f"- Git commit: `{environment['git_commit']}`",
        "",
        "## Aggregate metrics",
        "",
        "| Method | E2E runtime | Reblur RMSE ↓ | RMSE gain | Sharpness | Noise MAD | PSNR vs MATLAB* | SSIM vs MATLAB* |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("baseline", "annealed_pnp", "extreme_channel"):
        row = aggregate[key]
        summary_lines.append(
            "| "
            f"{METHODS[key]['name']} | "
            f"{float(row['end_to_end_runtime_seconds_total']):.2f}s | "
            f"{float(row['reblur_rmse_mean']):.5f} | "
            f"{float(row['reblur_rmse_improvement_vs_baseline_pct']):+.1f}% | "
            f"{float(row['sharpness_mean']):.4f} | "
            f"{float(row['noise_mad_mean']):.5f} | "
            f"{fmt(row['psnr_vs_historical_matlab_db_mean'], 2)} | "
            f"{fmt(row['ssim_vs_historical_matlab_mean'], 4)} |"
        )
    summary_lines.extend(
        [
            "",
            "*Historical MATLAB metrics are same-input agreement metrics, not clean-image ground truth.",
            "",
            "Open `report.html` for all 23 native-resolution visual comparisons.",
        ]
    )
    (output_dir / "SUMMARY.md").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    aggregate_rows = []
    for key in ("baseline", "annealed_pnp", "extreme_channel"):
        row = aggregate[key]
        aggregate_rows.append(
            "<tr>"
            f"<td><strong>{html.escape(METHODS[key]['name'])}</strong>"
            f"<small>{html.escape(METHODS[key]['description'])}</small></td>"
            f"<td>{float(row['end_to_end_runtime_seconds_total']):.2f}s</td>"
            f"<td>{float(row['reblur_rmse_mean']):.5f}</td>"
            f"<td>{float(row['reblur_rmse_improvement_vs_baseline_pct']):+.1f}%</td>"
            f"<td>{float(row['sharpness_mean']):.4f}</td>"
            f"<td>{float(row['noise_mad_mean']):.5f}</td>"
            f"<td>{fmt(row['psnr_vs_historical_matlab_db_mean'], 2)}</td>"
            f"<td>{fmt(row['ssim_vs_historical_matlab_mean'], 4)}</td>"
            f"<td>{int(row['historical_reference_count'])}</td>"
            "</tr>"
        )

    cases_html: list[str] = []
    for case in per_image:
        idx = int(case["index"])
        name = str(case["name"])
        rel = str(case["result_dir"])
        source_shape = list(case["source_shape"])
        rows = case["metrics"]
        runtimes = case["runtimes_seconds"]
        historical = case["historical_matlab"]

        cards = [
            card(
                "Observed source",
                f"{rel}/{case['source_copy']}",
                f"Native {shape_text(source_shape)} · exact source bytes copied",
                "Input",
            )
        ]
        if historical["result_copy"]:
            cards.append(
                card(
                    "Historical MATLAB",
                    f"{rel}/{historical['result_copy']}",
                    "Same-input exact-dimension reference from dataset/results",
                    "Reference",
                )
            )
        cards.extend(
            [
                card(
                    METHODS["baseline"]["name"],
                    f"{rel}/baseline.png",
                    "Optimized Python DCP baseline",
                    "Baseline",
                ),
                card(
                    METHODS["annealed_pnp"]["name"],
                    f"{rel}/annealed_pnp.png",
                    "Gaussian annealing + PnP consistency",
                    "New",
                ),
                card(
                    METHODS["extreme_channel"]["name"],
                    f"{rel}/extreme_channel.png",
                    "Dark + bright extrema refinement",
                    "New",
                ),
                card(
                    "Estimated Python kernel",
                    f"{rel}/kernel.png",
                    f"{case['kernel_size']}×{case['kernel_size']} shared PSF",
                    "PSF",
                ),
            ]
        )
        if historical["kernel_exact_shape_match"]:
            cards.append(
                card(
                    "Historical MATLAB kernel",
                    f"{rel}/historical_matlab_kernel.png",
                    (
                        "Exact kernel-size comparison · "
                        f"corr {historical['kernel_metrics']['correlation']:.4f}"
                    ),
                    "Reference",
                )
            )

        historical_note = (
            "Exact-shape MATLAB reference available"
            if historical["result_copy"]
            else f"MATLAB reference: {historical['result_status']}"
        )
        cases_html.append(
            f'<details class="case" {"open" if idx <= 2 else ""}>'
            "<summary>"
            f"<span><b>{idx:02d}</b> {html.escape(name)}</span>"
            f'<span class="summary-note">{shape_text(source_shape)} · '
            f"{html.escape(historical_note)}</span>"
            "</summary>"
            f'<div class="cards">{"".join(cards)}</div>'
            '<div class="table-wrap"><table><thead><tr>'
            "<th>Method</th><th>Stage runtime</th><th>Reblur RMSE ↓</th>"
            "<th>RMSE gain</th><th>Sharpness*</th><th>Sharp Δ</th>"
            "<th>Noise MAD*</th><th>Noise Δ</th>"
            "<th>PSNR vs MATLAB*</th><th>SSIM vs MATLAB*</th>"
            "</tr></thead>"
            f"<tbody>{method_table(rows, runtimes)}</tbody></table></div>"
            "</details>"
        )

    css = """
:root {
  --bg:#f4f7fb; --panel:#fff; --ink:#132033; --muted:#64748b;
  --line:#dfe6ef; --blue:#315efb; --navy:#101a31; --green:#0b8f68;
  --shadow:0 12px 34px rgba(25,42,70,.08);
}
* { box-sizing:border-box; }
body {
  margin:0; background:var(--bg); color:var(--ink);
  font:14px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
}
main { max-width:1500px; margin:auto; padding:38px 24px 64px; }
.hero {
  background:linear-gradient(135deg,#0d172c 0%,#24407f 58%,#49358c 100%);
  color:#fff; border-radius:26px; padding:34px 38px; box-shadow:var(--shadow);
}
.eyebrow {
  font-size:11px; font-weight:800; letter-spacing:.15em; text-transform:uppercase;
  color:#bdd0ff;
}
h1 { font-size:35px; line-height:1.13; margin:7px 0 10px; }
.hero>p { max-width:1050px; color:#dce7ff; margin:0; }
.kpis {
  display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:12px; margin-top:22px;
}
.kpi {
  padding:13px 15px; background:rgba(255,255,255,.1);
  border:1px solid rgba(255,255,255,.15); border-radius:14px;
}
.kpi b { display:block; font-size:22px; }
.kpi span { font-size:11px; color:#d1dcf8; }
section { margin-top:30px; }
h2 { font-size:23px; margin:0 0 6px; }
.lead { color:var(--muted); margin:0 0 15px; }
.notice {
  background:#eef3ff; border-left:4px solid var(--blue); padding:14px 16px;
  border-radius:10px; margin:16px 0; color:#30466f;
}
.notice.good { background:#eaf8f3; border-left-color:var(--green); color:#245c4b; }
.warning { background:#fff8e8; border-left-color:#e79b17; color:#665020; }
.methods {
  display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px;
}
.method {
  background:#fff; border:1px solid var(--line); border-radius:17px;
  padding:17px; box-shadow:var(--shadow);
}
.method b { font-size:16px; }
.method p { color:var(--muted); margin:5px 0 0; }
.table-wrap {
  overflow:auto; background:#fff; border:1px solid var(--line);
  border-radius:17px; box-shadow:var(--shadow); margin-top:14px;
}
table { border-collapse:collapse; width:100%; min-width:1120px; }
th,td {
  padding:11px 13px; border-bottom:1px solid var(--line);
  text-align:right; vertical-align:top;
}
th:first-child,td:first-child { text-align:left; }
th {
  background:#f8fafc; color:var(--muted); font-size:10px;
  text-transform:uppercase; letter-spacing:.055em;
}
td small {
  display:block; color:var(--muted); font-weight:400;
  max-width:430px; margin-top:3px;
}
tr:last-child td { border-bottom:0; }
.case {
  margin-top:13px; background:#fff; border:1px solid var(--line);
  border-radius:18px; box-shadow:var(--shadow); overflow:hidden;
}
summary {
  cursor:pointer; padding:15px 17px; font-size:15px; display:flex;
  justify-content:space-between; gap:12px; background:#fbfcfe;
}
summary b {
  display:inline-grid; place-items:center; width:29px; height:29px; margin-right:9px;
  border-radius:9px; background:#eaf0ff; color:#294fd2;
}
.summary-note { font-size:12px; color:var(--muted); font-weight:400; }
.cards {
  display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
  gap:12px; padding:14px;
}
.card {
  border:1px solid var(--line); border-radius:14px; padding:10px; background:#fff;
}
.card-head {
  display:flex; justify-content:space-between; gap:7px; align-items:flex-start;
  margin-bottom:8px;
}
.card h4 { font-size:13px; margin:0; }
.card p { font-size:10px; color:var(--muted); margin:2px 0 0; }
.card img {
  display:block; width:100%; height:auto; border-radius:9px; background:#eef2f7;
}
.badge {
  font-size:9px; font-weight:800; letter-spacing:.05em; text-transform:uppercase;
  padding:3px 6px; border-radius:999px; background:#eaf0ff; color:#294fd2;
  white-space:nowrap;
}
.meta {
  display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px;
}
.meta div {
  background:#fff; border:1px solid var(--line); border-radius:14px;
  padding:13px 15px; box-shadow:var(--shadow);
}
.meta b { display:block; font-size:12px; color:var(--muted); margin-bottom:4px; }
.meta span { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }
.foot { font-size:12px; color:var(--muted); margin-top:24px; }
code { background:#eaf0f5; border-radius:5px; padding:2px 5px; }
@media(max-width:900px) {
  main { padding:18px 12px 38px; }
  .hero { padding:25px 20px; }
  h1 { font-size:27px; }
  .kpis,.meta { grid-template-columns:1fr 1fr; }
  .methods { grid-template-columns:1fr; }
}
@media(max-width:520px) {
  .kpis,.meta { grid-template-columns:1fr; }
}
"""

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dark Channel Deblur · Native-Resolution Benchmark</title>
<style>{css}</style>
</head>
<body><main>
<header class="hero">
  <div class="eyebrow">Reproducible native-resolution Docker benchmark</div>
  <h1>Dark-channel deblurring · three-method research comparison</h1>
  <p>
    All {len(images)} committed source images are processed at their original decoded dimensions.
    No benchmark resizing, cropping, or reference resampling is permitted. The two research
    refinements reuse the exact PSF estimated by the Python dark-channel baseline for each image.
  </p>
  <div class="kpis">
    <div class="kpi"><b>{len(images)}</b><span>native source images</span></div>
    <div class="kpi"><b>{len(images) * 3}</b><span>restored outputs</span></div>
    <div class="kpi"><b>{historical_matches}</b><span>exact-shape MATLAB references</span></div>
    <div class="kpi"><b>{historical_kernel_matches}</b><span>kernel references compared</span></div>
    <div class="kpi"><b>{total_runtime:.1f}s</b><span>benchmark runtime</span></div>
  </div>
</header>

<section>
  <h2>Benchmark contract</h2>
  <p class="lead">The report makes the resolution and reference rules explicit so results remain reproducible.</p>
  <div class="notice good">
    <strong>Native-resolution invariant:</strong> every method receives the original decoded
    source dimensions and must return exactly the same dimensions. A resolution change is a
    hard test failure.
  </div>
  <div class="notice">
    <strong>Historical MATLAB comparison:</strong> when <code>dataset/results/&lt;stem&gt;_result.png</code>
    exists and has the exact same dimensions, PSNR/SSIM are computed as agreement with that
    historical output. No reference is resized. These are fidelity metrics, not clean-image ground truth.
  </div>
  <div class="notice warning">
    <strong>Interpretation:</strong> reblur RMSE measures consistency with the estimated physical
    blur model. Sobel sharpness and Laplacian MAD are diagnostics, not standalone perceptual-quality scores.
  </div>
</section>

<section>
  <h2>Methods</h2>
  <div class="methods">
    <div class="method"><b>Dark Channel Baseline</b><p>{html.escape(METHODS['baseline']['description'])}</p></div>
    <div class="method"><b>Annealed Gaussian PnP</b><p>{html.escape(METHODS['annealed_pnp']['description'])}</p></div>
    <div class="method"><b>Extreme-Channel Guided</b><p>{html.escape(METHODS['extreme_channel']['description'])}</p></div>
  </div>
</section>

<section>
  <h2>Aggregate results</h2>
  <p class="lead">
    End-to-end time for each refinement includes the shared baseline PSF/restoration stage plus
    that refinement stage. MATLAB metrics average only exact-dimension historical references.
  </p>
  <div class="table-wrap"><table>
    <thead><tr>
      <th>Method</th><th>E2E runtime</th><th>Reblur RMSE ↓</th><th>RMSE gain</th>
      <th>Sharpness*</th><th>Noise MAD*</th><th>PSNR vs MATLAB*</th>
      <th>SSIM vs MATLAB*</th><th>Reference n</th>
    </tr></thead>
    <tbody>{''.join(aggregate_rows)}</tbody>
  </table></div>
</section>

<section>
  <h2>Reproducibility metadata</h2>
  <div class="meta">
    <div><b>Git commit</b><span>{html.escape(str(environment['git_commit']))}</span></div>
    <div><b>Python</b><span>{html.escape(str(environment['python']))}</span></div>
    <div><b>OpenCV / NumPy / SciPy</b><span>{environment['opencv']} / {environment['numpy']} / {environment['scipy']}</span></div>
    <div><b>Generated UTC</b><span>{html.escape(str(environment['generated_at_utc']))}</span></div>
  </div>
</section>

<section>
  <h2>All source images</h2>
  <p class="lead">
    Expand a case for native-resolution source, historical MATLAB output where available,
    all three Python results, PSF comparison, timing, and diagnostics.
  </p>
  {''.join(cases_html)}
</section>

<p class="foot">
  Generated by <code>docker compose run --rm test</code>.
  Machine-readable values are in <code>report.json</code>; the CI-friendly summary is
  <code>SUMMARY.md</code>. Historical MATLAB PSNR/SSIM values are agreement metrics only.
</p>
</main></body></html>
"""
    report_html = output_dir / "report.html"
    report_html.write_text(html_doc, encoding="utf-8")
    print(f"Native-resolution report written to {report_html}", flush=True)
    return report_html


def main() -> int:
    args = build_parser().parse_args()
    generate_report(
        args.output_dir,
        limit=args.limit,
        forced_kernel_size=args.kernel_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
