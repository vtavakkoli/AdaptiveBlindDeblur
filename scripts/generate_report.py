#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import re
import shutil
import time
from pathlib import Path
from statistics import mean

import cv2
import numpy as np

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
RESULTS = ROOT / "results"
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EXPECTED_IMAGES = 23
CI_MAX_SIDE = 192

METHODS = {
    "baseline": {
        "name": "Dark Channel Baseline",
        "short": "DCP",
        "description": "Fast Python port of Pan et al. CVPR 2016: blind PSF estimation plus TV/L0 restoration.",
    },
    "annealed_pnp": {
        "name": "Annealed Gaussian PnP",
        "short": "A-PnP",
        "description": "Diffusion-inspired, weight-free refinement with Gaussian annealing, NLM denoising, and FFT measurement consistency.",
    },
    "extreme_channel": {
        "name": "Extreme-Channel Guided",
        "short": "ECP-R",
        "description": "Dark+bright local-extrema guided detail refinement with FFT measurement consistency.",
    },
}


def dataset_images() -> list[Path]:
    if not DATASET.is_dir():
        raise RuntimeError(f"dataset directory is missing: {DATASET}")
    return sorted(p for p in DATASET.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED)


def slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("._") or "image"


def working_copy(image: np.ndarray, max_side: int = CI_MAX_SIDE) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    h, w = arr.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale >= 1.0:
        return arr.copy()
    return cv2.resize(
        arr,
        (max(2, round(w * scale)), max(2, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)


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


def diagnostics(
    output: np.ndarray,
    observed: np.ndarray,
    kernel: np.ndarray,
    *,
    workers: int,
) -> dict[str, float]:
    predicted = reblur_image(output, kernel, workers=workers)
    dark_fraction, bright_fraction = extreme_metrics(output)
    return {
        "reblur_rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
        "sharpness": sharpness(output),
        "noise_mad": noise_mad(output),
        "dark_fraction": dark_fraction,
        "bright_fraction": bright_fraction,
    }


def pct_change(value: float, baseline: float, *, lower_is_better: bool = False) -> float:
    if abs(baseline) < 1e-12:
        return 0.0
    if lower_is_better:
        return 100.0 * (baseline - value) / baseline
    return 100.0 * (value - baseline) / baseline


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _pct(value: float) -> str:
    return f"{value:+.1f}%"


def _card(title: str, image_path: str, subtitle: str, badge: str = "") -> str:
    tag = f'<span class="badge">{html.escape(badge)}</span>' if badge else ""
    return (
        '<article class="card"><div class="card-head"><div>'
        f'<h4>{html.escape(title)}</h4><p>{html.escape(subtitle)}</p></div>{tag}</div>'
        f'<img loading="lazy" src="{html.escape(image_path)}" alt="{html.escape(title)}"></article>'
    )


def _method_table(rows: dict[str, dict[str, float]], runtimes: dict[str, float]) -> str:
    baseline = rows["baseline"]
    body: list[str] = []
    for key in ("baseline", "annealed_pnp", "extreme_channel"):
        d = rows[key]
        rmse_gain = pct_change(d["reblur_rmse"], baseline["reblur_rmse"], lower_is_better=True)
        sharp_gain = pct_change(d["sharpness"], baseline["sharpness"])
        noise_change = pct_change(d["noise_mad"], baseline["noise_mad"])
        body.append(
            "<tr>"
            f"<td><strong>{html.escape(METHODS[key]['name'])}</strong></td>"
            f"<td>{runtimes[key]:.3f}s</td>"
            f"<td>{_fmt(d['reblur_rmse'], 5)}</td>"
            f"<td>{_pct(rmse_gain)}</td>"
            f"<td>{_fmt(d['sharpness'], 4)}</td>"
            f"<td>{_pct(sharp_gain)}</td>"
            f"<td>{_fmt(d['noise_mad'], 5)}</td>"
            f"<td>{_pct(noise_change)}</td>"
            f"<td>{_fmt(d['dark_fraction'], 3)}</td>"
            f"<td>{_fmt(d['bright_fraction'], 3)}</td>"
            "</tr>"
        )
    return "".join(body)


def generate_report(output_dir: Path = RESULTS) -> Path:
    images = dataset_images()
    if len(images) != EXPECTED_IMAGES:
        raise RuntimeError(f"dataset/image contains {len(images)} supported images; expected {EXPECTED_IMAGES}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.name == ".gitkeep":
            continue
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    cfg = DeblurConfig(
        kernel_size=15,
        lambda_dark=4e-3,
        lambda_grad=4e-3,
        gamma_correct=1.0,
        xk_iter=2,
        lambda_tv=3e-3,
        lambda_l0=5e-4,
        weight_ring=1.0,
        dark_patch_size=21,
        max_grad_steps=8,
        max_dark_steps=3,
        fft_workers=-1,
    )

    original_shapes: dict[str, list[int]] = {}
    per_image: list[dict[str, object]] = []
    metric_sets: dict[str, list[dict[str, float]]] = {key: [] for key in METHODS}
    runtime_sets: dict[str, list[float]] = {key: [] for key in METHODS}
    total_started = time.perf_counter()

    for index, path in enumerate(images, 1):
        print(f"[{index:02d}/{len(images)}] benchmarking {path.name}", flush=True)
        original = read_image(path)
        original_shapes[path.name] = list(original.shape)
        observed = working_copy(original)
        case_dir = images_out / f"{index:02d}_{slugify(path.stem)}"
        case_dir.mkdir(parents=True, exist_ok=True)
        write_image(case_dir / "input.png", observed)

        started = time.perf_counter()
        baseline, kernel, interim = deblur_image(observed, cfg)
        baseline_time = time.perf_counter() - started

        started = time.perf_counter()
        annealed = annealed_pnp_refine(
            observed,
            baseline,
            kernel,
            steps=4,
            sigma_start=0.025,
            sigma_end=0.004,
            candidates=2,
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
        write_image(case_dir / "baseline.png", baseline)
        write_image(case_dir / "annealed_pnp.png", annealed)
        write_image(case_dir / "extreme_channel.png", extreme)
        write_image(case_dir / "interim.png", interim)
        write_image(case_dir / "kernel.png", kernel / max(float(kernel.max()), 1e-12))

        rows: dict[str, dict[str, float]] = {}
        for method, output in outputs.items():
            if output.shape != observed.shape or not np.isfinite(output).all():
                raise RuntimeError(f"{method} produced invalid output for {path.name}")
            row = diagnostics(output, observed, kernel, workers=cfg.fft_workers)
            rows[method] = row
            metric_sets[method].append(row)
            runtime_sets[method].append(runtimes[method])

        base_row = rows["baseline"]
        gains = {
            method: {
                "reblur_rmse_improvement_pct": pct_change(
                    row["reblur_rmse"], base_row["reblur_rmse"], lower_is_better=True
                ),
                "sharpness_change_pct": pct_change(row["sharpness"], base_row["sharpness"]),
                "noise_mad_change_pct": pct_change(row["noise_mad"], base_row["noise_mad"]),
            }
            for method, row in rows.items()
        }
        per_image.append(
            {
                "index": index,
                "name": path.name,
                "stem": path.stem,
                "original_shape": list(original.shape),
                "working_shape": list(observed.shape),
                "result_dir": str(case_dir.relative_to(output_dir)).replace("\\", "/"),
                "kernel_sum": float(kernel.sum()),
                "kernel_peak": float(kernel.max()),
                "runtimes_seconds": runtimes,
                "metrics": rows,
                "gains_vs_baseline": gains,
            }
        )

    total_runtime = time.perf_counter() - total_started
    aggregate: dict[str, dict[str, float]] = {}
    for method in METHODS:
        rows = metric_sets[method]
        aggregate[method] = {
            "stage_runtime_seconds_mean": mean(runtime_sets[method]),
            "stage_runtime_seconds_total": sum(runtime_sets[method]),
            "reblur_rmse_mean": mean(row["reblur_rmse"] for row in rows),
            "sharpness_mean": mean(row["sharpness"] for row in rows),
            "noise_mad_mean": mean(row["noise_mad"] for row in rows),
            "dark_fraction_mean": mean(row["dark_fraction"] for row in rows),
            "bright_fraction_mean": mean(row["bright_fraction"] for row in rows),
        }

    base_agg = aggregate["baseline"]
    for method, row in aggregate.items():
        row["reblur_rmse_improvement_vs_baseline_pct"] = pct_change(
            row["reblur_rmse_mean"], base_agg["reblur_rmse_mean"], lower_is_better=True
        )
        row["sharpness_change_vs_baseline_pct"] = pct_change(
            row["sharpness_mean"], base_agg["sharpness_mean"]
        )
        row["noise_mad_change_vs_baseline_pct"] = pct_change(
            row["noise_mad_mean"], base_agg["noise_mad_mean"]
        )
        if method == "baseline":
            row["end_to_end_runtime_seconds_total"] = row["stage_runtime_seconds_total"]
        else:
            row["end_to_end_runtime_seconds_total"] = (
                base_agg["stage_runtime_seconds_total"] + row["stage_runtime_seconds_total"]
            )

    report = {
        "benchmark": "Complete repository dataset/image three-method benchmark",
        "dataset": {
            "path": "dataset/image",
            "source": "23 source images committed in the repository from the supplied CVPR 2016 release",
            "image_count": len(images),
            "working_max_side_pixels": CI_MAX_SIDE,
            "original_shapes": original_shapes,
            "ground_truth": None,
            "ground_truth_note": (
                "This image folder does not provide verified pixel-aligned clean targets for these 23 cases. "
                "PSNR/SSIM are therefore intentionally not reported."
            ),
        },
        "methods": METHODS,
        "fairness": "Each image uses one blind DCP kernel estimate; both new refinements reuse that same kernel.",
        "total_runtime_seconds": total_runtime,
        "config": {
            "kernel_size": cfg.kernel_size,
            "xk_iter": cfg.xk_iter,
            "lambda_dark": cfg.lambda_dark,
            "lambda_grad": cfg.lambda_grad,
            "dark_patch_size": cfg.dark_patch_size,
            "max_grad_steps": cfg.max_grad_steps,
            "max_dark_steps": cfg.max_dark_steps,
        },
        "aggregate": aggregate,
        "images": per_image,
        "metric_notes": {
            "reblur_rmse": "Lower is better measurement consistency: reblur(restored, estimated_kernel) vs working input.",
            "sharpness": "Mean Sobel magnitude. Higher means more edge energy, but can also reward ringing/noise.",
            "noise_mad": "Laplacian median absolute deviation. A diagnostic of high-frequency/noise content, not a quality score.",
            "dark_fraction": "Fraction of 9x9 local dark-channel values below 0.03.",
            "bright_fraction": "Fraction of 9x9 local bright-channel values above 0.97.",
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    agg_rows = "".join(
        "<tr>"
        f"<td><strong>{html.escape(METHODS[key]['name'])}</strong><small>{html.escape(METHODS[key]['description'])}</small></td>"
        f"<td>{aggregate[key]['end_to_end_runtime_seconds_total']:.2f}s</td>"
        f"<td>{_fmt(aggregate[key]['reblur_rmse_mean'], 5)}</td>"
        f"<td>{_pct(aggregate[key]['reblur_rmse_improvement_vs_baseline_pct'])}</td>"
        f"<td>{_fmt(aggregate[key]['sharpness_mean'], 4)}</td>"
        f"<td>{_pct(aggregate[key]['sharpness_change_vs_baseline_pct'])}</td>"
        f"<td>{_fmt(aggregate[key]['noise_mad_mean'], 5)}</td>"
        f"<td>{_pct(aggregate[key]['noise_mad_change_vs_baseline_pct'])}</td>"
        f"<td>{_fmt(aggregate[key]['dark_fraction_mean'], 3)}</td>"
        f"<td>{_fmt(aggregate[key]['bright_fraction_mean'], 3)}</td>"
        "</tr>"
        for key in ("baseline", "annealed_pnp", "extreme_channel")
    )

    cases_html: list[str] = []
    for case in per_image:
        idx = int(case["index"])
        name = str(case["name"])
        rel = str(case["result_dir"])
        rows = case["metrics"]  # type: ignore[assignment]
        runtimes = case["runtimes_seconds"]  # type: ignore[assignment]
        cards = [
            _card("Observed input", f"{rel}/input.png", f"CI working copy of {name}", "Input"),
            _card(METHODS["baseline"]["name"], f"{rel}/baseline.png", "Blind DCP result and shared kernel", "Baseline"),
            _card(METHODS["annealed_pnp"]["name"], f"{rel}/annealed_pnp.png", "Gaussian annealing + PnP consistency", "New"),
            _card(METHODS["extreme_channel"]["name"], f"{rel}/extreme_channel.png", "Dark + bright extrema refinement", "New"),
            _card("Estimated kernel", f"{rel}/kernel.png", "Shared PSF used by all three methods", "PSF"),
        ]
        cases_html.append(
            f'<details class="case" {"open" if idx <= 2 else ""}>'
            f'<summary><span><b>{idx:02d}</b> {html.escape(name)}</span><span class="summary-note">'
            f'{html.escape(str(case["original_shape"]))} → {html.escape(str(case["working_shape"]))}</span></summary>'
            f'<div class="cards">{"".join(cards)}</div>'
            '<div class="table-wrap"><table><thead><tr><th>Method</th><th>Stage runtime</th><th>Reblur RMSE ↓</th>'
            '<th>RMSE gain</th><th>Sharpness*</th><th>Sharp Δ</th><th>Noise MAD*</th><th>Noise Δ</th>'
            '<th>Dark frac.</th><th>Bright frac.</th></tr></thead>'
            f'<tbody>{_method_table(rows, runtimes)}</tbody></table></div></details>'
        )

    html_doc = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dark Channel Deblur · Full Dataset Research Report</title>
<style>
:root{{--bg:#f4f7fb;--panel:#fff;--ink:#142033;--muted:#65748b;--line:#dfe6ef;--blue:#315efb;--shadow:0 12px 35px rgba(25,42,70,.08)}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}main{{max-width:1460px;margin:auto;padding:38px 24px 64px}}
.hero{{background:linear-gradient(135deg,#101a31 0%,#253d7a 58%,#49388d 100%);color:white;border-radius:26px;padding:34px 38px;box-shadow:var(--shadow)}}.eyebrow{{font-size:11px;font-weight:800;letter-spacing:.15em;text-transform:uppercase;color:#bcd0ff}}h1{{font-size:34px;line-height:1.15;margin:7px 0 10px}}.hero>p{{max-width:980px;color:#dce6ff;margin:0}}.kpis{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:22px}}.kpi{{padding:13px 15px;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.15);border-radius:14px}}.kpi b{{display:block;font-size:22px}}.kpi span{{font-size:11px;color:#d1dcf8}}
section{{margin-top:30px}}h2{{font-size:23px;margin:0 0 6px}}.lead{{color:var(--muted);margin:0 0 15px}}.notice{{background:#eef3ff;border-left:4px solid var(--blue);padding:14px 16px;border-radius:10px;margin:16px 0;color:#30466f}}.warning{{background:#fff8e8;border-left-color:#e79b17;color:#665020}}.methods{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}.method{{background:white;border:1px solid var(--line);border-radius:17px;padding:17px;box-shadow:var(--shadow)}}.method b{{font-size:16px}}.method p{{color:var(--muted);margin:5px 0 0}}
.table-wrap{{overflow:auto;background:white;border:1px solid var(--line);border-radius:17px;box-shadow:var(--shadow);margin-top:14px}}table{{border-collapse:collapse;width:100%;min-width:1100px}}th,td{{padding:11px 13px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}}th:first-child,td:first-child{{text-align:left}}th{{background:#f8fafc;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.055em}}td small{{display:block;color:var(--muted);font-weight:400;max-width:430px;margin-top:3px}}tr:last-child td{{border-bottom:0}}
.case{{margin-top:13px;background:white;border:1px solid var(--line);border-radius:18px;box-shadow:var(--shadow);overflow:hidden}}summary{{cursor:pointer;padding:15px 17px;font-size:15px;display:flex;justify-content:space-between;gap:12px;background:#fbfcfe}}summary b{{display:inline-grid;place-items:center;width:29px;height:29px;margin-right:9px;border-radius:9px;background:#eaf0ff;color:#294fd2}}.summary-note{{font-size:12px;color:var(--muted);font-weight:400}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;padding:14px}}.card{{border:1px solid var(--line);border-radius:14px;padding:10px;background:#fff}}.card-head{{display:flex;justify-content:space-between;gap:7px;align-items:flex-start;margin-bottom:8px}}.card h4{{font-size:13px;margin:0}}.card p{{font-size:10px;color:var(--muted);margin:2px 0 0}}.card img{{display:block;width:100%;height:auto;border-radius:9px;background:#eef2f7}}.badge{{font-size:9px;font-weight:800;letter-spacing:.05em;text-transform:uppercase;padding:3px 6px;border-radius:999px;background:#eaf0ff;color:#294fd2;white-space:nowrap}}
.foot{{font-size:12px;color:var(--muted);margin-top:24px}}code{{background:#eaf0f5;border-radius:5px;padding:2px 5px}}@media(max-width:800px){{main{{padding:18px 12px 38px}}.hero{{padding:25px 20px}}h1{{font-size:27px}}.kpis,.methods{{grid-template-columns:1fr 1fr}}}}@media(max-width:520px){{.kpis,.methods{{grid-template-columns:1fr}}}}
</style></head><body><main>
<header class="hero"><div class="eyebrow">Automated full-dataset Docker benchmark</div><h1>Dark-channel deblurring · three-method research comparison</h1><p>Every image committed under <code>dataset/image</code> is processed by the baseline and two new weight-free refinement variants. Both refinements reuse the same estimated blur kernel, isolating the restoration-prior effect.</p>
<div class="kpis"><div class="kpi"><b>{len(images)}</b><span>dataset images</span></div><div class="kpi"><b>{len(images)*3}</b><span>restored outputs</span></div><div class="kpi"><b>{len(images)}</b><span>blind PSF estimates</span></div><div class="kpi"><b>{total_runtime:.1f}s</b><span>benchmark runtime</span></div></div></header>
<section><h2>Methods</h2><p class="lead">The additions are dependency-light research variants, not claims of reproducing a trained diffusion SOTA system.</p><div class="methods">
<div class="method"><b>Dark Channel Baseline</b><p>{html.escape(METHODS['baseline']['description'])}</p></div>
<div class="method"><b>Annealed Gaussian PnP</b><p>{html.escape(METHODS['annealed_pnp']['description'])}</p></div>
<div class="method"><b>Extreme-Channel Guided</b><p>{html.escape(METHODS['extreme_channel']['description'])}</p></div></div>
<div class="notice"><strong>Fair comparison.</strong> One blind DCP kernel is estimated per image and shared by all three outputs. This makes differences in the two new rows attributable to their restoration prior rather than a different estimated PSF.</div>
<div class="notice warning"><strong>No fabricated ground truth.</strong> Inspection of the release folder confirms that similarly named files are not verified pixel-aligned clean targets. Therefore this report intentionally does not calculate PSNR/SSIM. Reblur RMSE is the physical consistency metric; sharpness and noise MAD are diagnostics and must be read together.</div></section>
<section><h2>Aggregate diagnostics</h2><p class="lead">Averages over all {len(images)} inputs. Positive RMSE gain means better consistency than the baseline. Sharpness gain can be useful detail or ringing; Noise Δ exposes that trade-off.</p><div class="table-wrap"><table><thead><tr><th>Method</th><th>End-to-end total</th><th>Reblur RMSE ↓</th><th>RMSE gain</th><th>Sharpness*</th><th>Sharp Δ</th><th>Noise MAD*</th><th>Noise Δ</th><th>Dark frac.</th><th>Bright frac.</th></tr></thead><tbody>{agg_rows}</tbody></table></div></section>
<section><h2>Per-image visual comparison</h2><p class="lead">Open any case to compare input, baseline, both new variants, the shared kernel, and method-relative diagnostics.</p>{''.join(cases_html)}</section>
<p class="foot">Generated by <code>docker compose run --rm test</code>. Baseline: Pan et al., CVPR 2016. Extreme-channel motivation: Yan et al., CVPR 2017. Annealed PnP is diffusion-inspired but uses no neural checkpoint. Machine-readable results: <code>report.json</code>.</p>
</main></body></html>'''
    report_path = output_dir / "report.html"
    report_path.write_text(html_doc, encoding="utf-8")
    print(f"Full dataset report written to {report_path}")
    return report_path


def main() -> int:
    generate_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
