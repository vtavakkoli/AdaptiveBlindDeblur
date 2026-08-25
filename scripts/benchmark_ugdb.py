#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np

from dark_channel_deblur import (
    DeblurConfig,
    annealed_pnp_refine,
    deblur_image,
    extreme_channel_refine,
    residual_guided_adaptive_consensus_refine,
    ugdb_restore,
)
from dark_channel_deblur.io import read_image, write_image
from dark_channel_deblur.quality import restoration_score
from dark_channel_deblur.refinement import reblur_image

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset" / "image"
PROFILES = ROOT / "dataset" / "benchmark_profiles.json"
DEFAULT_OUTPUT = ROOT / "results" / "ugdb_experiment"
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

METHOD_ORDER = (
    "baseline",
    "annealed_pnp",
    "extreme_channel",
    "rgac",
    "ugdb_linear",
    "ugdb_null",
    "ugdb_kernel",
    "ugdb_full",
)
METHOD_NAMES = {
    "baseline": "Adaptive Blind Baseline",
    "annealed_pnp": "Annealed PnP",
    "extreme_channel": "Dual-Extreme",
    "rgac": "RGAC",
    "ugdb_linear": "UGDB-linear",
    "ugdb_null": "UGDB-null",
    "ugdb_kernel": "UGDB-kernel",
    "ugdb_full": "UGDB-full",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare UGDB ablations against current methods.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test image limit.")
    parser.add_argument("--steps", type=int, default=4, help="UGDB iteration count.")
    parser.add_argument("--kernel-hypotheses", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def _config(profile: dict[str, float | int]) -> DeblurConfig:
    return DeblurConfig(
        kernel_size=int(profile["kernel_size"]),
        gamma_correct=float(profile["gamma"]),
        xk_iter=5,
        lambda_dark=float(profile["lambda_dark"]),
        lambda_grad=float(profile["lambda_grad"]),
        lambda_tv=float(profile["lambda_tv"]),
        lambda_l0=float(profile["lambda_l0"]),
        weight_ring=float(profile["weight_ring"]),
        dark_patch_size=35,
        max_grad_steps=None,
        max_dark_steps=None,
        fft_workers=-1,
    )


def _evaluate(
    observed: np.ndarray,
    output: np.ndarray,
    kernel: np.ndarray,
    *,
    workers: int,
) -> dict[str, float]:
    predicted = reblur_image(output, kernel, workers=workers)
    score, diag = restoration_score(observed, output, predicted)
    return {
        "blind_score": float(score),
        "reblur_rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
        "edge_ratio": float(diag.edge_ratio),
        "noise_ratio": float(diag.noise_ratio),
        "highpass_ratio": float(diag.highpass_ratio),
        "clipping_growth": float(diag.clipping_growth),
    }


def _run_timed(callable_: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    value = callable_()
    return value, time.perf_counter() - start


def _clean(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        shutil.rmtree(child) if child.is_dir() else child.unlink()


def _render_html(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    rows = []
    for method in METHOD_ORDER:
        row = aggregate[method]
        winner = " ★" if method == report["overall_winner"] else ""
        rows.append(
            "<tr>"
            f"<td><b>{html.escape(METHOD_NAMES[method])}{winner}</b></td>"
            f"<td>{row['win_count']}</td>"
            f"<td>{row['blind_score_mean']:.6f}</td>"
            f"<td>{row['reblur_rmse_mean']:.6f}</td>"
            f"<td>{row['edge_ratio_mean']:.3f}</td>"
            f"<td>{row['noise_ratio_mean']:.3f}</td>"
            f"<td>{row['runtime_seconds_mean']:.3f}s</td>"
            "</tr>"
        )

    cases = []
    for case in report["images"]:
        cards = []
        for method in METHOD_ORDER:
            metric = case["methods"][method]["metrics"]
            badge = "winner" if method == case["winner"] else ""
            cards.append(
                '<article class="card">'
                f"<h4>{html.escape(METHOD_NAMES[method])} {badge}</h4>"
                f'<img loading="lazy" src="{html.escape(case["directory"] + "/" + method + ".png")}">'
                f"<p>score {metric['blind_score']:.5f} · RMSE {metric['reblur_rmse']:.5f} · "
                f"{case['methods'][method]['runtime_seconds']:.2f}s</p></article>"
            )
        cases.append(
            '<details class="case"><summary>'
            f"{html.escape(case['name'])} — winner: {html.escape(METHOD_NAMES[case['winner']])}"
            f"</summary><div class="cards">{''.join(cards)}</div></details>"
        )

    css = """
body{margin:0;background:#f6f8fb;color:#162033;font:14px/1.5 system-ui,sans-serif}
main{max-width:1500px;margin:auto;padding:30px}h1{margin-bottom:4px}.lead{color:#667085}
.hero,.case,.table{background:#fff;border:1px solid #e1e6ef;border-radius:16px;box-shadow:0 8px 28px #1b29400d}
.hero{padding:24px}.winner{font-size:22px;font-weight:800}.table{overflow:auto;margin:22px 0}
table{border-collapse:collapse;width:100%;min-width:850px}th,td{padding:11px 13px;border-bottom:1px solid #e8ecf2;text-align:right}
th:first-child,td:first-child{text-align:left}th{background:#f8fafc}.case{margin:12px 0;overflow:hidden}
summary{cursor:pointer;padding:15px 17px;font-weight:700}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px;padding:12px}
.card{background:#fff;border:1px solid #e5e9f0;border-radius:12px;padding:9px}.card h4{margin:2px 0 7px}.card p{color:#667085;font-size:11px}
.card img{width:100%;height:auto;border-radius:8px;display:block;background:#eef2f6}
"""
    return (
        '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>UGDB experiment</title><style>{css}</style></head><body><main>"
        '<section class="hero"><h1>UGDB blind-deblurring experiment</h1>'
        '<p class="lead">Reference-free comparison of current methods and four uncertainty-guided Gaussian ablations. '
        'Lower blind score is better; it combines reblur fidelity with artifact penalties.</p>'
        f'<div class="winner">Overall lowest mean score: {html.escape(METHOD_NAMES[report["overall_winner"]])}</div>'
        f'<p>{report["dataset_count"]} native-resolution images · seed {report["seed"]}</p></section>'
        '<div class="table"><table><thead><tr><th>Method</th><th>Wins</th><th>Mean blind score ↓</th>'
        '<th>Mean reblur RMSE ↓</th><th>Edge ratio</th><th>Noise ratio</th><th>Mean stage runtime</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div><h2>Per-image results</h2>{''.join(cases)}"
        '</main></body></html>'
    )


def main() -> int:
    args = build_parser().parse_args()
    if args.steps < 1 or args.kernel_hypotheses < 1:
        raise SystemExit("--steps and --kernel-hypotheses must be >= 1")

    images = sorted(p for p in DATASET.iterdir() if p.suffix.lower() in SUPPORTED)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise RuntimeError("no benchmark images found")
    profiles = json.loads(PROFILES.read_text(encoding="utf-8"))
    _clean(args.output_dir)

    report_images: list[dict[str, Any]] = []
    metric_sets: dict[str, list[dict[str, float]]] = {method: [] for method in METHOD_ORDER}
    runtime_sets: dict[str, list[float]] = {method: [] for method in METHOD_ORDER}
    win_counts = {method: 0 for method in METHOD_ORDER}

    for index, source in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] {source.name}", flush=True)
        profile = profiles[source.name]
        cfg = _config(profile)
        observed = read_image(source)
        case_dir = args.output_dir / f"{index:02d}_{source.stem}"
        case_dir.mkdir(parents=True, exist_ok=True)

        (baseline_pack, baseline_runtime) = _run_timed(lambda: deblur_image(observed, cfg))
        baseline, base_kernel, _ = baseline_pack
        methods: dict[str, dict[str, Any]] = {}

        def record(method: str, output: np.ndarray, kernel: np.ndarray, runtime: float, extra: Any = None) -> None:
            metrics = _evaluate(observed, output, kernel, workers=cfg.fft_workers)
            write_image(case_dir / f"{method}.png", output)
            methods[method] = {
                "runtime_seconds": float(runtime),
                "metrics": metrics,
                "extra": extra,
            }
            metric_sets[method].append(metrics)
            runtime_sets[method].append(float(runtime))

        record("baseline", baseline, base_kernel, baseline_runtime)

        annealed, runtime = _run_timed(
            lambda: annealed_pnp_refine(
                observed, baseline, base_kernel, seed=args.seed, workers=cfg.fft_workers
            )
        )
        record("annealed_pnp", annealed, base_kernel, runtime)

        extreme, runtime = _run_timed(
            lambda: extreme_channel_refine(observed, baseline, base_kernel, workers=cfg.fft_workers)
        )
        record("extreme_channel", extreme, base_kernel, runtime)

        rgac, runtime = _run_timed(
            lambda: residual_guided_adaptive_consensus_refine(
                observed,
                baseline,
                base_kernel,
                annealed=annealed,
                extreme=extreme,
                seed=args.seed,
                workers=cfg.fft_workers,
            )
        )
        record("rgac", rgac, base_kernel, runtime)

        variants = {
            "ugdb_linear": "linear",
            "ugdb_null": "nullspace",
            "ugdb_kernel": "kernel",
            "ugdb_full": "full",
        }
        for method, variant in variants.items():
            pack, runtime = _run_timed(
                lambda variant=variant: ugdb_restore(
                    observed,
                    baseline,
                    base_kernel,
                    variant=variant,
                    steps=args.steps,
                    kernel_hypotheses=args.kernel_hypotheses,
                    seed=args.seed,
                    workers=cfg.fft_workers,
                )
            )
            output, updated_kernel, diagnostics = pack
            record(method, output, updated_kernel, runtime, asdict(diagnostics))

        winner = min(METHOD_ORDER, key=lambda name: methods[name]["metrics"]["blind_score"])
        win_counts[winner] += 1
        report_images.append(
            {
                "name": source.name,
                "directory": case_dir.relative_to(args.output_dir).as_posix(),
                "shape": list(observed.shape),
                "profile": profile,
                "winner": winner,
                "methods": methods,
            }
        )

    aggregate: dict[str, dict[str, float | int]] = {}
    for method in METHOD_ORDER:
        aggregate[method] = {
            "win_count": int(win_counts[method]),
            "blind_score_mean": float(mean(row["blind_score"] for row in metric_sets[method])),
            "reblur_rmse_mean": float(mean(row["reblur_rmse"] for row in metric_sets[method])),
            "edge_ratio_mean": float(mean(row["edge_ratio"] for row in metric_sets[method])),
            "noise_ratio_mean": float(mean(row["noise_ratio"] for row in metric_sets[method])),
            "runtime_seconds_mean": float(mean(runtime_sets[method])),
        }
    overall_winner = min(METHOD_ORDER, key=lambda name: aggregate[name]["blind_score_mean"])
    report = {
        "dataset_count": len(images),
        "seed": args.seed,
        "steps": args.steps,
        "kernel_hypotheses": args.kernel_hypotheses,
        "overall_winner": overall_winner,
        "aggregate": aggregate,
        "images": report_images,
    }

    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "report.html").write_text(_render_html(report), encoding="utf-8")
    summary = [
        "# UGDB experiment",
        "",
        f"Overall lowest mean blind score: **{METHOD_NAMES[overall_winner]}**",
        "",
        "| Method | Wins | Mean score ↓ | Reblur RMSE ↓ | Mean runtime |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        row = aggregate[method]
        summary.append(
            f"| {METHOD_NAMES[method]} | {row['win_count']} | {row['blind_score_mean']:.6f} | "
            f"{row['reblur_rmse_mean']:.6f} | {row['runtime_seconds_mean']:.3f}s |"
        )
    (args.output_dir / "SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Winner: {METHOD_NAMES[overall_winner]}")
    print(f"Report: {args.output_dir / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
