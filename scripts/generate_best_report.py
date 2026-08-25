#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from dataclasses import asdict

import generate_report as benchmark

from dark_channel_deblur import (
    DeblurConfig,
    residual_guided_adaptive_consensus_refine,
)


RGAC_KEY = "rgac"
RGAC_METHOD = {
    "name": "Residual-Guided Adaptive Consensus",
    "description": (
        "Reference-free spatial consensus over robust, conservative, Annealed PnP, "
        "and Dual-Extreme candidates using local reblur/artifact confidence maps, "
        "PSF-aware weighting, and a final blur-consistency projection."
    ),
}


def config_from_profile(profile: dict[str, float | int | bool]) -> DeblurConfig:
    """Build the best-quality benchmark configuration from an explicit profile.

    The blind core keeps the MATLAB-equivalent numerical implementation introduced
    by the parity work. Robust selection is enabled above that core so suspicious
    long-kernel restorations can use a safer restoration or an independent
    gradient-only PSF retry. Legacy result/kernel pixels remain evaluation-only.
    """
    return DeblurConfig(
        kernel_size=int(profile["kernel_size"]),
        lambda_dark=float(profile["lambda_dark"]),
        lambda_grad=float(profile["lambda_grad"]),
        gamma_correct=float(profile["gamma"]),
        xk_iter=5,
        lambda_tv=float(profile["lambda_tv"]),
        lambda_l0=float(profile["lambda_l0"]),
        weight_ring=float(profile["weight_ring"]),
        saturated=bool(profile.get("saturated", False)),
        saturation_iterations=50,
        dark_patch_size=35,
        robust_selection=True,
        retry_gradient_only=True,
        conservative_restoration=True,
        max_grad_steps=None,
        max_dark_steps=None,
        fft_workers=-1,
    )


def _augment_with_rgac(captures: list[dict[str, object]]) -> None:
    """Add the fourth method to the just-generated three-method benchmark artifacts."""
    report_path = benchmark.RESULTS / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cases = report["images"]
    if len(cases) != len(captures):
        raise RuntimeError("RGAC capture count does not match generated report cases")

    benchmark.METHODS[RGAC_KEY] = RGAC_METHOD
    benchmark.METHOD_ORDER = (*benchmark.METHOD_ORDER, RGAC_KEY)

    for case, capture in zip(cases, captures, strict=True):
        observed = capture["observed"]
        kernel = capture["kernel"]
        rgac = capture["rgac"]
        config = capture["config"]
        diagnostics = capture["diagnostics"]
        if rgac.shape != observed.shape:
            raise RuntimeError(f"RGAC changed resolution for {case['name']}")

        case_dir = benchmark.RESULTS / case["result_dir"]
        benchmark.write_image(case_dir / "rgac.png", rgac)
        legacy_reference = None
        if case["legacy_reference_status"] == "exact_shape":
            legacy_reference = benchmark.read_image(case_dir / "legacy_result.png")

        row = benchmark.diagnostics(
            rgac,
            observed,
            kernel,
            workers=config.fft_workers,
            legacy_reference=legacy_reference,
        )
        case["metrics"][RGAC_KEY] = row
        # RGAC uses the already-generated PnP and Dual-Extreme candidates. For a
        # fair standalone E2E comparison, its stage time includes those candidate
        # generation costs plus the additional consensus/projection work.
        case["runtimes_seconds"][RGAC_KEY] = (
            float(case["runtimes_seconds"]["annealed_pnp"])
            + float(case["runtimes_seconds"]["extreme_channel"])
            + float(capture["rgac_extra_seconds"])
        )
        case["output_shapes"][RGAC_KEY] = list(rgac.shape)
        case["rgac"] = asdict(diagnostics)

    metric_sets = {
        key: [case["metrics"][key] for case in cases]
        for key in benchmark.METHOD_ORDER
    }
    runtime_sets = {
        key: [float(case["runtimes_seconds"][key]) for case in cases]
        for key in benchmark.METHOD_ORDER
    }
    aggregate = benchmark.aggregate_metrics(metric_sets, runtime_sets)
    report["schema_version"] = 3
    report["benchmark"] = "Full-quality native-resolution four-method deblurring comparison"
    report["methods"] = benchmark.METHODS
    report["aggregate"] = aggregate
    report["rgac_design"] = {
        "name": RGAC_METHOD["name"],
        "learned_weights": False,
        "legacy_inputs_used": False,
        "candidate_set": ["baseline", "conservative", "annealed_pnp", "extreme_channel"],
        "selection": "local reference-free residual/artifact soft consensus plus global safety guard",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    html_text = benchmark.build_html(report, cases, aggregate)
    # build_html predates RGAC and has three hard-coded visual cards. Insert the
    # RGAC output before each PSF card while leaving all metric/aggregate rows driven
    # by the now-four-element METHOD_ORDER.
    for case in cases:
        rel = str(case["result_dir"])
        profile = case["profile"]
        psf_card = benchmark.card(
            "Estimated kernel",
            f"{rel}/kernel.png",
            f"{profile['kernel_size']}×{profile['kernel_size']} independently estimated PSF",
            "PSF",
        )
        rgac_card = benchmark.card(
            RGAC_METHOD["name"],
            f"{rel}/rgac.png",
            "Local residual consensus + PSF-aware safety + final data consistency",
            "Consensus",
        )
        html_text = html_text.replace(psf_card, rgac_card + psf_card, 1)

    image_count = int(report["dataset"]["image_count"])
    html_text = html_text.replace("three-method comparison", "four-method comparison")
    html_text = html_text.replace(
        f"<b>{image_count * 3}</b><span>restored outputs</span>",
        f"<b>{image_count * 4}</b><span>restored outputs</span>",
    )
    html_text = html_text.replace(
        ".methods{display:grid;grid-template-columns:repeat(3,1fr)",
        ".methods{display:grid;grid-template-columns:repeat(4,1fr)",
    )
    (benchmark.RESULTS / "report.html").write_text(html_text, encoding="utf-8")

    legacy = report["legacy_comparison"]
    benchmark.write_summary(
        benchmark.RESULTS,
        image_count,
        float(report["total_runtime_seconds"]),
        int(legacy["exact_shape_results"]),
        int(legacy["comparable_kernels"]),
        aggregate,
    )
    summary_path = benchmark.RESULTS / "SUMMARY.md"
    summary = summary_path.read_text(encoding="utf-8")
    summary = summary.replace(
        f"- Restorations: **{image_count * 3}**",
        f"- Restorations: **{image_count * 4}**",
    )
    summary_path.write_text(summary, encoding="utf-8")


def main() -> int:
    benchmark.config_from_profile = config_from_profile
    benchmark.METHODS["baseline"] = {
        "name": "Adaptive Robust Baseline",
        "description": (
            "MATLAB-equivalent blind kernel core plus reference-free ripple detection, "
            "conservative restoration selection, saturated-scene early stopping, and "
            "an independent gradient-only PSF fallback when the primary solution is suspicious."
        ),
    }

    captures: list[dict[str, object]] = []
    original_deblur = benchmark.deblur_image
    original_annealed = benchmark.annealed_pnp_refine
    original_extreme = benchmark.extreme_channel_refine

    def capture_deblur(observed, config):
        baseline, kernel, interim = original_deblur(observed, config)
        captures.append(
            {
                "observed": observed,
                "baseline": baseline,
                "kernel": kernel,
                "config": config,
            }
        )
        return baseline, kernel, interim

    def capture_annealed(observed, baseline, kernel, **kwargs):
        result = original_annealed(observed, baseline, kernel, **kwargs)
        captures[-1]["annealed"] = result
        return result

    def capture_extreme(observed, baseline, kernel, **kwargs):
        result = original_extreme(observed, baseline, kernel, **kwargs)
        captures[-1]["extreme"] = result
        started = time.perf_counter()
        rgac, diagnostics = residual_guided_adaptive_consensus_refine(
            observed,
            baseline,
            kernel,
            annealed=captures[-1]["annealed"],
            extreme=result,
            seed=len(captures),
            workers=kwargs.get("workers", -1),
            return_diagnostics=True,
        )
        captures[-1]["rgac"] = rgac
        captures[-1]["diagnostics"] = diagnostics
        captures[-1]["rgac_extra_seconds"] = time.perf_counter() - started
        return result

    benchmark.deblur_image = capture_deblur
    benchmark.annealed_pnp_refine = capture_annealed
    benchmark.extreme_channel_refine = capture_extreme
    try:
        benchmark.generate_report()
    finally:
        benchmark.deblur_image = original_deblur
        benchmark.annealed_pnp_refine = original_annealed
        benchmark.extreme_channel_refine = original_extreme

    _augment_with_rgac(captures)
    print(
        "RGAC augmentation complete: report now compares four native-resolution methods.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
