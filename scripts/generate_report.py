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
LEGACY_RESULTS = ROOT / "dataset" / "results"
PROFILE_FILE = ROOT / "dataset" / "benchmark_profiles.json"
RESULTS = ROOT / "results"
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EXPECTED_IMAGES = 23

METHODS = {
    "baseline": {
        "name": "Adaptive Blind Baseline",
        "short": "ABB",
        "description": (
            "Multi-scale blind PSF estimation with sparse local-extrema and gradient "
            "regularization, followed by full-quality TV/L0 restoration."
        ),
    },
    "annealed_pnp": {
        "name": "Annealed PnP Refinement",
        "short": "A-PnP",
        "description": (
            "Annealed Gaussian perturbation, non-local denoising, explicit blur-model "
            "consistency, and an artifact-safety acceptance guard."
        ),
    },
    "extreme_channel": {
        "name": "Dual-Extreme Refinement",
        "short": "DER",
        "description": (
            "Dark/bright local-extrema detail guidance with explicit blur-model consistency "
            "and an artifact-safety acceptance guard."
        ),
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full-quality native-resolution deblurring benchmark."
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
        help="Developer smoke-test limit. Docker validation intentionally does not use this.",
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


def load_profiles() -> dict[str, dict[str, float | int]]:
    if not PROFILE_FILE.is_file():
        raise RuntimeError(f"benchmark profile file is missing: {PROFILE_FILE}")
    raw = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("benchmark_profiles.json must contain one object keyed by filename")
    return raw


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


def psnr(candidate: np.ndarray, reference: np.ndarray) -> float:
    if candidate.shape != reference.shape:
        raise ValueError(f"PSNR shape mismatch: {candidate.shape} != {reference.shape}")
    mse = float(np.mean((candidate.astype(np.float64) - reference.astype(np.float64)) ** 2))
    return 120.0 if mse <= 1e-12 else float(10.0 * math.log10(1.0 / mse))


def ssim(candidate: np.ndarray, reference: np.ndarray) -> float:
    a = np.asarray(candidate, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"SSIM shape mismatch: {a.shape} != {b.shape}")
    if a.ndim == 2:
        a, b = a[..., None], b[..., None]
    c1, c2 = 0.01**2, 0.03**2
    values: list[float] = []
    for channel in range(a.shape[2]):
        x, y = a[..., channel], b[..., channel]
        mx = cv2.GaussianBlur(x, (11, 11), 1.5)
        my = cv2.GaussianBlur(y, (11, 11), 1.5)
        vx = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mx * mx
        vy = cv2.GaussianBlur(y * y, (11, 11), 1.5) - my * my
        vxy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mx * my
        score = ((2 * mx * my + c1) * (2 * vxy + c2)) / (
            (mx * mx + my * my + c1) * (vx + vy + c2) + 1e-12
        )
        core = score[5:-5, 5:-5] if min(score.shape) > 12 else score
        values.append(float(np.mean(core)))
    return float(np.mean(values))


def read_kernel(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    arr = image.astype(np.float32) / 255.0
    total = float(arr.sum())
    return arr / total if total > 0 else None


def kernel_metrics(candidate: np.ndarray, reference: np.ndarray | None) -> dict[str, float] | None:
    if reference is None or candidate.shape != reference.shape:
        return None
    a = candidate.astype(np.float64)
    b = reference.astype(np.float64)
    a /= max(float(a.sum()), 1e-12)
    b /= max(float(b.sum()), 1e-12)
    correlation = 0.0
    if float(np.std(a)) > 1e-12 and float(np.std(b)) > 1e-12:
        correlation = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    return {
        "correlation": correlation,
        "l1_distance": float(np.abs(a - b).sum()),
    }


def legacy_paths(source: Path) -> tuple[Path | None, Path | None]:
    result = LEGACY_RESULTS / f"{source.stem}_result.png"
    kernel = LEGACY_RESULTS / f"{source.stem}_kernel.png"
    return result if result.is_file() else None, kernel if kernel.is_file() else None


def diagnostics(
    output: np.ndarray,
    observed: np.ndarray,
    kernel: np.ndarray,
    *,
    workers: int,
    legacy_reference: np.ndarray | None,
) -> dict[str, float | None]:
    predicted = reblur_image(output, kernel, workers=workers)
    values: dict[str, float | None] = {
        "reblur_rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
        "sharpness": sharpness(output),
        "noise_mad": noise_mad(output),
        "psnr_vs_legacy_db": None,
        "ssim_vs_legacy": None,
    }
    if legacy_reference is not None:
        values["psnr_vs_legacy_db"] = psnr(output, legacy_reference)
        values["ssim_vs_legacy"] = ssim(output, legacy_reference)
    return values


def pct_change(value: float, baseline: float, *, lower_is_better: bool = False) -> float:
    if abs(baseline) < 1e-12:
        return 0.0
    if lower_is_better:
        return 100.0 * (baseline - value) / baseline
    return 100.0 * (value - baseline) / baseline


def fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def pct(value: float) -> str:
    return f"{value:+.1f}%"


def shape_text(shape: list[int] | tuple[int, ...]) -> str:
    if len(shape) == 3:
        return f"{shape[1]}×{shape[0]}×{shape[2]}"
    return f"{shape[1]}×{shape[0]}"


def clean_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.name == ".gitkeep":
            continue
        shutil.rmtree(path) if path.is_dir() else path.unlink()


def copy_source(source: Path, case_dir: Path) -> str:
    target_name = f"input{source.suffix.lower()}"
    shutil.copy2(source, case_dir / target_name)
    return target_name


def card(title: str, image_path: str, subtitle: str, badge: str) -> str:
    return (
        '<article class="card"><div class="card-head"><div>'
        f"<h4>{html.escape(title)}</h4><p>{html.escape(subtitle)}</p></div>"
        f'<span class="badge">{html.escape(badge)}</span></div>'
        f'<img loading="lazy" src="{html.escape(image_path)}" alt="{html.escape(title)}"></article>'
    )


def method_table(rows: dict[str, dict[str, float | None]], runtimes: dict[str, float]) -> str:
    base = rows["baseline"]
    base_rmse = float(base["reblur_rmse"])
    base_sharp = float(base["sharpness"])
    base_noise = float(base["noise_mad"])
    body: list[str] = []
    for key in ("baseline", "annealed_pnp", "extreme_channel"):
        row = rows[key]
        rmse = float(row["reblur_rmse"])
        sharp = float(row["sharpness"])
        noise = float(row["noise_mad"])
        body.append(
            "<tr>"
            f"<td><strong>{html.escape(METHODS[key]['name'])}</strong></td>"
            f"<td>{runtimes[key]:.3f}s</td>"
            f"<td>{fmt(rmse, 5)}</td>"
            f"<td>{pct(pct_change(rmse, base_rmse, lower_is_better=True))}</td>"
            f"<td>{fmt(sharp, 4)}</td>"
            f"<td>{pct(pct_change(sharp, base_sharp))}</td>"
            f"<td>{fmt(noise, 5)}</td>"
            f"<td>{pct(pct_change(noise, base_noise))}</td>"
            f"<td>{fmt(row['psnr_vs_legacy_db'], 2)}</td>"
            f"<td>{fmt(row['ssim_vs_legacy'], 4)}</td>"
            "</tr>"
        )
    return "".join(body)


def config_from_profile(profile: dict[str, float | int]) -> DeblurConfig:
    return DeblurConfig(
        kernel_size=int(profile["kernel_size"]),
        lambda_dark=float(profile["lambda_dark"]),
        lambda_grad=float(profile["lambda_grad"]),
        gamma_correct=float(profile["gamma"]),
        xk_iter=5,
        lambda_tv=float(profile["lambda_tv"]),
        lambda_l0=float(profile["lambda_l0"]),
        weight_ring=float(profile["weight_ring"]),
        dark_patch_size=35,
        max_grad_steps=None,
        max_dark_steps=None,
        fft_workers=-1,
    )


def git_commit() -> str:
    return os.environ.get("GITHUB_SHA", "local")


def generate_report(output_dir: Path = RESULTS, *, limit: int | None = None) -> Path:
    images = dataset_images()
    if len(images) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"dataset/image contains {len(images)} supported images; expected {EXPECTED_IMAGES}"
        )
    profiles = load_profiles()
    image_names = {path.name for path in images}
    if set(profiles) != image_names:
        missing = sorted(image_names - set(profiles))
        extra = sorted(set(profiles) - image_names)
        raise RuntimeError(f"benchmark profile mismatch: missing={missing}, extra={extra}")

    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be >= 1")
        images = images[:limit]

    clean_output_dir(output_dir)
    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    per_image: list[dict[str, Any]] = []
    metric_sets: dict[str, list[dict[str, float | None]]] = {key: [] for key in METHODS}
    runtime_sets: dict[str, list[float]] = {key: [] for key in METHODS}
    total_started = time.perf_counter()
    legacy_matches = 0
    legacy_kernel_matches = 0

    for index, source in enumerate(images, 1):
        profile = profiles[source.name]
        cfg = config_from_profile(profile)
        print(
            f"[{index:02d}/{len(images):02d}] full-quality native benchmark: {source.name} "
            f"(kernel {cfg.kernel_size}×{cfg.kernel_size})",
            flush=True,
        )

        observed = read_image(source)
        source_shape = list(observed.shape)
        case_dir = images_out / f"{index:02d}_{slugify(source.stem)}"
        case_dir.mkdir(parents=True, exist_ok=True)
        source_copy = copy_source(source, case_dir)

        legacy_result_path, legacy_kernel_path = legacy_paths(source)
        legacy_reference: np.ndarray | None = None
        legacy_reference_status = "missing"
        if legacy_result_path is not None:
            candidate = read_image(legacy_result_path)
            if candidate.shape == observed.shape:
                legacy_reference = candidate
                legacy_reference_status = "exact_shape"
                shutil.copy2(legacy_result_path, case_dir / "legacy_result.png")
                legacy_matches += 1
            else:
                legacy_reference_status = "shape_mismatch"

        legacy_kernel = read_kernel(legacy_kernel_path)
        if legacy_kernel_path is not None:
            shutil.copy2(legacy_kernel_path, case_dir / "legacy_kernel.png")

        started = time.perf_counter()
        baseline, kernel, interim = deblur_image(observed, cfg)
        baseline_time = time.perf_counter() - started

        started = time.perf_counter()
        annealed = annealed_pnp_refine(
            observed,
            baseline,
            kernel,
            steps=3,
            sigma_start=0.020,
            sigma_end=0.004,
            candidates=1,
            seed=index,
            workers=cfg.fft_workers,
        )
        annealed_time = time.perf_counter() - started

        started = time.perf_counter()
        extreme = extreme_channel_refine(
            observed,
            baseline,
            kernel,
            steps=3,
            patch_size=15,
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
        for method, output in outputs.items():
            if output.shape != observed.shape:
                raise RuntimeError(
                    f"{method} changed resolution for {source.name}: "
                    f"{output.shape} != {observed.shape}"
                )
            if not np.isfinite(output).all():
                raise RuntimeError(f"{method} produced non-finite values for {source.name}")
            write_image(case_dir / f"{method}.png", output)
        write_image(case_dir / "interim.png", interim)
        write_image(case_dir / "kernel.png", kernel / max(float(kernel.max()), 1e-12))

        rows: dict[str, dict[str, float | None]] = {}
        for method, output in outputs.items():
            row = diagnostics(
                output,
                observed,
                kernel,
                workers=cfg.fft_workers,
                legacy_reference=legacy_reference,
            )
            rows[method] = row
            metric_sets[method].append(row)
            runtime_sets[method].append(runtimes[method])

        k_metrics = kernel_metrics(kernel, legacy_kernel)
        if k_metrics is not None:
            legacy_kernel_matches += 1

        per_image.append(
            {
                "index": index,
                "name": source.name,
                "source_shape": source_shape,
                "output_shapes": {key: list(value.shape) for key, value in outputs.items()},
                "native_resolution": True,
                "resizing_applied": False,
                "source_sha256": sha256_file(source),
                "source_copy": source_copy,
                "result_dir": str(case_dir.relative_to(output_dir)).replace("\\", "/"),
                "profile": profile,
                "kernel_sum": float(kernel.sum()),
                "kernel_peak": float(kernel.max()),
                "kernel_shape": list(kernel.shape),
                "legacy_reference_status": legacy_reference_status,
                "legacy_kernel_shape": list(legacy_kernel.shape) if legacy_kernel is not None else None,
                "legacy_kernel_metrics": k_metrics,
                "runtimes_seconds": runtimes,
                "metrics": rows,
            }
        )

    total_runtime = time.perf_counter() - total_started
    aggregate: dict[str, dict[str, float | int | None]] = {}
    for method in METHODS:
        rows = metric_sets[method]
        psnrs = [float(row["psnr_vs_legacy_db"]) for row in rows if row["psnr_vs_legacy_db"] is not None]
        ssims = [float(row["ssim_vs_legacy"]) for row in rows if row["ssim_vs_legacy"] is not None]
        aggregate[method] = {
            "stage_runtime_seconds_total": float(sum(runtime_sets[method])),
            "stage_runtime_seconds_mean": float(mean(runtime_sets[method])),
            "reblur_rmse_mean": float(mean(float(row["reblur_rmse"]) for row in rows)),
            "sharpness_mean": float(mean(float(row["sharpness"]) for row in rows)),
            "noise_mad_mean": float(mean(float(row["noise_mad"]) for row in rows)),
            "psnr_vs_legacy_db_mean": float(mean(psnrs)) if psnrs else None,
            "ssim_vs_legacy_mean": float(mean(ssims)) if ssims else None,
            "legacy_reference_count": len(psnrs),
        }

    base = aggregate["baseline"]
    for method, row in aggregate.items():
        row["reblur_rmse_improvement_vs_baseline_pct"] = pct_change(
            float(row["reblur_rmse_mean"]),
            float(base["reblur_rmse_mean"]),
            lower_is_better=True,
        )
        row["sharpness_change_vs_baseline_pct"] = pct_change(
            float(row["sharpness_mean"]), float(base["sharpness_mean"])
        )
        row["noise_mad_change_vs_baseline_pct"] = pct_change(
            float(row["noise_mad_mean"]), float(base["noise_mad_mean"])
        )
        if method == "baseline":
            row["end_to_end_runtime_seconds_total"] = row["stage_runtime_seconds_total"]
        else:
            row["end_to_end_runtime_seconds_total"] = float(base["stage_runtime_seconds_total"]) + float(
                row["stage_runtime_seconds_total"]
            )

    report = {
        "schema_version": 2,
        "benchmark": "Full-quality native-resolution three-method deblurring comparison",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "opencv": cv2.__version__,
            "platform": platform.platform(),
        },
        "dataset": {
            "path": "dataset/image",
            "image_count": len(images),
            "native_resolution": True,
            "resizing_applied": False,
            "profile_file": "dataset/benchmark_profiles.json",
            "profile_note": (
                "Profiles define search support and regularization only. Legacy result pixels and "
                "legacy kernel values are never used by the restoration algorithms."
            ),
        },
        "legacy_comparison": {
            "path": "dataset/results",
            "exact_shape_results": legacy_matches,
            "comparable_kernels": legacy_kernel_matches,
            "resampling_permitted": False,
            "role": "evaluation_only",
            "note": (
                "Legacy outputs are previous-method snapshots used only for side-by-side and "
                "agreement metrics. They are not ground truth and are not algorithm inputs."
            ),
        },
        "methods": METHODS,
        "aggregate": aggregate,
        "total_runtime_seconds": total_runtime,
        "images": per_image,
        "metric_notes": {
            "reblur_rmse": "Lower means the restored image reproduces the observed blur more closely when reblurred with its estimated PSF.",
            "sharpness": "Mean Sobel gradient magnitude; diagnostic only.",
            "noise_mad": "Laplacian median absolute deviation; diagnostic for high-frequency/noise amplification.",
            "psnr_vs_legacy": "Agreement with the previous saved result for the same input; not ground-truth PSNR.",
            "ssim_vs_legacy": "Structural agreement with the previous saved result; not ground-truth SSIM.",
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    agg_rows = "".join(
        "<tr>"
        f"<td><strong>{html.escape(METHODS[key]['name'])}</strong><small>{html.escape(METHODS[key]['description'])}</small></td>"
        f"<td>{float(aggregate[key]['end_to_end_runtime_seconds_total']):.2f}s</td>"
        f"<td>{fmt(float(aggregate[key]['reblur_rmse_mean']), 5)}</td>"
        f"<td>{pct(float(aggregate[key]['reblur_rmse_improvement_vs_baseline_pct']))}</td>"
        f"<td>{fmt(float(aggregate[key]['sharpness_mean']), 4)}</td>"
        f"<td>{fmt(float(aggregate[key]['noise_mad_mean']), 5)}</td>"
        f"<td>{fmt(aggregate[key]['psnr_vs_legacy_db_mean'], 2)}</td>"
        f"<td>{fmt(aggregate[key]['ssim_vs_legacy_mean'], 4)}</td>"
        f"<td>{int(aggregate[key]['legacy_reference_count'])}</td>"
        "</tr>"
        for key in ("baseline", "annealed_pnp", "extreme_channel")
    )

    cases_html: list[str] = []
    for case in per_image:
        idx = int(case["index"])
        rel = str(case["result_dir"])
        source_shape = case["source_shape"]
        profile = case["profile"]
        rows = case["metrics"]
        runtimes = case["runtimes_seconds"]
        cards = [
            card(
                "Observed source",
                f"{rel}/{case['source_copy']}",
                f"Native {shape_text(source_shape)} · exact source bytes copied",
                "Input",
            )
        ]
        if case["legacy_reference_status"] == "exact_shape":
            cards.append(
                card(
                    "Legacy output",
                    f"{rel}/legacy_result.png",
                    "Previous saved result · evaluation only",
                    "Legacy",
                )
            )
        cards.extend(
            [
                card(METHODS["baseline"]["name"], f"{rel}/baseline.png", "Full-quality blind restoration", "Baseline"),
                card(METHODS["annealed_pnp"]["name"], f"{rel}/annealed_pnp.png", "Annealed prior + safety guard", "Refinement"),
                card(METHODS["extreme_channel"]["name"], f"{rel}/extreme_channel.png", "Dual-extreme prior + safety guard", "Refinement"),
                card("Estimated kernel", f"{rel}/kernel.png", f"{profile['kernel_size']}×{profile['kernel_size']} independently estimated PSF", "PSF"),
            ]
        )
        if case["legacy_kernel_shape"] is not None:
            km = case["legacy_kernel_metrics"]
            subtitle = "Previous saved kernel · evaluation only"
            if km is not None:
                subtitle += f" · corr {km['correlation']:.4f}"
            cards.append(card("Legacy kernel", f"{rel}/legacy_kernel.png", subtitle, "Legacy"))

        profile_text = (
            f"kernel={int(profile['kernel_size'])}, γ={float(profile['gamma']):g}, "
            f"λtv={float(profile['lambda_tv']):g}, λl0={float(profile['lambda_l0']):g}"
        )
        cases_html.append(
            f'<details class="case" {"open" if idx <= 3 else ""}>'
            f'<summary><span><b>{idx:02d}</b> {html.escape(str(case["name"]))}</span>'
            f'<span class="summary-note">{html.escape(shape_text(source_shape))} · {html.escape(profile_text)}</span></summary>'
            f'<div class="cards">{"".join(cards)}</div>'
            '<div class="table-wrap"><table><thead><tr><th>Method</th><th>Stage runtime</th>'
            '<th>Reblur RMSE ↓</th><th>RMSE gain</th><th>Sharpness*</th><th>Sharp Δ</th>'
            '<th>Noise MAD*</th><th>Noise Δ</th><th>PSNR vs legacy*</th><th>SSIM vs legacy*</th>'
            f'</tr></thead><tbody>{method_table(rows, runtimes)}</tbody></table></div></details>'
        )

    html_doc = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Adaptive Blind Deblur · Native-Resolution Benchmark</title>
<style>
:root{{--bg:#f4f7fb;--panel:#fff;--ink:#132033;--muted:#64748b;--line:#dfe6ef;--blue:#315efb;--green:#0b8f68;--shadow:0 12px 34px rgba(25,42,70,.08)}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}main{{max-width:1540px;margin:auto;padding:36px 22px 64px}}
.hero{{background:linear-gradient(135deg,#0d172c 0%,#24407f 58%,#49358c 100%);color:#fff;border-radius:25px;padding:34px 38px;box-shadow:var(--shadow)}}.eyebrow{{font-size:11px;font-weight:800;letter-spacing:.15em;text-transform:uppercase;color:#bdd0ff}}h1{{font-size:34px;line-height:1.13;margin:7px 0 10px}}.hero>p{{max-width:1080px;color:#dce7ff;margin:0}}.kpis{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin-top:22px}}.kpi{{padding:13px 15px;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.15);border-radius:14px}}.kpi b{{display:block;font-size:22px}}.kpi span{{font-size:11px;color:#d1dcf8}}
section{{margin-top:30px}}h2{{font-size:23px;margin:0 0 6px}}.lead{{color:var(--muted);margin:0 0 15px}}.notice{{background:#eef3ff;border-left:4px solid var(--blue);padding:14px 16px;border-radius:10px;margin:14px 0;color:#30466f}}.notice.good{{background:#eaf8f3;border-left-color:var(--green);color:#245c4b}}.methods{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}.method{{background:#fff;border:1px solid var(--line);border-radius:17px;padding:17px;box-shadow:var(--shadow)}}.method b{{font-size:16px}}.method p{{color:var(--muted);margin:5px 0 0}}
.table-wrap{{overflow:auto;background:#fff;border:1px solid var(--line);border-radius:17px;box-shadow:var(--shadow);margin-top:14px}}table{{border-collapse:collapse;width:100%;min-width:1120px}}th,td{{padding:11px 13px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}}th:first-child,td:first-child{{text-align:left}}th{{background:#f8fafc;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.055em}}td small{{display:block;color:var(--muted);font-weight:400;max-width:430px;margin-top:3px}}tr:last-child td{{border-bottom:0}}
.case{{margin-top:13px;background:#fff;border:1px solid var(--line);border-radius:18px;box-shadow:var(--shadow);overflow:hidden}}summary{{cursor:pointer;padding:15px 17px;font-size:15px;display:flex;justify-content:space-between;gap:12px;background:#fbfcfe}}summary b{{display:inline-grid;place-items:center;width:29px;height:29px;margin-right:9px;border-radius:9px;background:#eaf0ff;color:#294fd2}}.summary-note{{font-size:12px;color:var(--muted);font-weight:400}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;padding:14px}}.card{{border:1px solid var(--line);border-radius:14px;padding:10px;background:#fff}}.card-head{{display:flex;justify-content:space-between;gap:7px;align-items:flex-start;margin-bottom:8px}}.card h4{{font-size:13px;margin:0}}.card p{{font-size:10px;color:var(--muted);margin:2px 0 0}}.card img{{display:block;width:100%;height:auto;border-radius:9px;background:#eef2f7}}.badge{{font-size:9px;font-weight:800;letter-spacing:.05em;text-transform:uppercase;padding:3px 6px;border-radius:999px;background:#eaf0ff;color:#294fd2;white-space:nowrap}}
.meta{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}.meta div{{background:#fff;border:1px solid var(--line);border-radius:14px;padding:13px 15px;box-shadow:var(--shadow)}}.meta b{{display:block;font-size:12px;color:var(--muted);margin-bottom:4px}}.meta span{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}}code{{background:#eaf0f5;border-radius:5px;padding:2px 5px}}.foot{{font-size:12px;color:var(--muted);margin-top:24px}}@media(max-width:900px){{main{{padding:18px 12px 38px}}.hero{{padding:25px 20px}}h1{{font-size:27px}}.kpis,.meta{{grid-template-columns:1fr 1fr}}.methods{{grid-template-columns:1fr}}}}@media(max-width:520px){{.kpis,.meta{{grid-template-columns:1fr}}}}
</style></head><body><main>
<header class="hero"><div class="eyebrow">Full-quality native-resolution Docker benchmark</div><h1>Adaptive blind deblurring · three-method comparison</h1><p>Every source image is processed at its native decoded dimensions. The benchmark uses explicit dataset profiles for PSF support and regularization. Previous saved outputs are evaluation-only and are never fed into the algorithms.</p><div class="kpis"><div class="kpi"><b>{len(images)}</b><span>native source images</span></div><div class="kpi"><b>{len(images)*3}</b><span>restored outputs</span></div><div class="kpi"><b>{legacy_matches}</b><span>legacy outputs compared</span></div><div class="kpi"><b>{legacy_kernel_matches}</b><span>legacy kernels comparable</span></div><div class="kpi"><b>{total_runtime:.1f}s</b><span>benchmark runtime</span></div></div></header>
<section><h2>Benchmark contract</h2><p class="lead">The quality run no longer uses the old 192-pixel shortcut or one generic PSF support for all images.</p><div class="notice good"><strong>Native resolution:</strong> source dimensions must be preserved exactly by all three methods. Any resize is a hard failure.</div><div class="notice"><strong>Independent inference:</strong> <code>dataset/benchmark_profiles.json</code> supplies only configuration. Legacy pixels and legacy kernel values are evaluation-only.</div><div class="notice"><strong>Artifact guard:</strong> both refinements are blended back toward the stable baseline when lower reblur error is accompanied by excessive high-frequency amplification.</div></section>
<section><h2>Methods</h2><div class="methods">{''.join(f'<div class="method"><b>{html.escape(METHODS[k]["name"])}</b><p>{html.escape(METHODS[k]["description"])}</p></div>' for k in ("baseline","annealed_pnp","extreme_channel"))}</div></section>
<section><h2>Aggregate results</h2><p class="lead">Legacy metrics are agreement with previous saved outputs, not clean-image ground-truth scores.</p><div class="table-wrap"><table><thead><tr><th>Method</th><th>E2E runtime</th><th>Reblur RMSE ↓</th><th>RMSE gain</th><th>Sharpness*</th><th>Noise MAD*</th><th>PSNR vs legacy*</th><th>SSIM vs legacy*</th><th>Reference n</th></tr></thead><tbody>{agg_rows}</tbody></table></div></section>
<section><h2>Reproducibility metadata</h2><div class="meta"><div><b>Git commit</b><span>{html.escape(git_commit())}</span></div><div><b>Python</b><span>{platform.python_version()}</span></div><div><b>OpenCV / NumPy / SciPy</b><span>{cv2.__version__} / {np.__version__} / {scipy.__version__}</span></div><div><b>Generated UTC</b><span>{html.escape(report['generated_utc'])}</span></div></div></section>
<section><h2>All source images</h2><p class="lead">Expand each case to inspect source, previous saved output, all current methods, PSF, profile, timing, and diagnostics.</p>{''.join(cases_html)}</section>
<p class="foot">A lower reblur RMSE is useful only when visual artifacts and high-frequency amplification remain controlled. Use the image comparisons together with the metrics.</p></main></body></html>'''
    report_path = output_dir / "report.html"
    report_path.write_text(html_doc, encoding="utf-8")

    summary_lines = [
        "# Native-resolution deblurring benchmark",
        "",
        f"- Images: **{len(images)}**",
        f"- Restorations: **{len(images) * 3}**",
        f"- Native resolution: **yes**",
        f"- Resizing: **none**",
        f"- Legacy exact-shape outputs compared: **{legacy_matches}**",
        f"- Comparable legacy kernels: **{legacy_kernel_matches}**",
        f"- Total benchmark runtime: **{total_runtime:.2f} s**",
        "",
        "| Method | Reblur RMSE ↓ | Gain vs baseline | Noise MAD | PSNR vs legacy* | SSIM vs legacy* |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in ("baseline", "annealed_pnp", "extreme_channel"):
        row = aggregate[key]
        summary_lines.append(
            f"| {METHODS[key]['name']} | {float(row['reblur_rmse_mean']):.5f} | "
            f"{float(row['reblur_rmse_improvement_vs_baseline_pct']):+.1f}% | "
            f"{float(row['noise_mad_mean']):.5f} | {fmt(row['psnr_vs_legacy_db_mean'], 2)} | "
            f"{fmt(row['ssim_vs_legacy_mean'], 4)} |"
        )
    summary_lines.extend(
        [
            "",
            "*Legacy metrics measure agreement with previous saved outputs; they are not ground-truth quality scores.*",
        ]
    )
    (output_dir / "SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Full-quality native-resolution report written to {report_path}", flush=True)
    return report_path


def main() -> int:
    args = build_parser().parse_args()
    generate_report(args.output_dir, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
