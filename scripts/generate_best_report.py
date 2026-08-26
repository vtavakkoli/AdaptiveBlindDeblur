#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import time
from dataclasses import asdict, replace
from statistics import mean

import cv2
import numpy as np

import generate_report as benchmark

from dark_channel_deblur import (
    DeblurConfig,
    residual_guided_adaptive_consensus_refine,
)
from dark_channel_deblur.kernel import (
    adjust_psf_center,
    estimate_psf,
    prune_kernel,
    valid_gradients,
)
from dark_channel_deblur.psf_quality import psf_plausibility, refine_psf_structure
from dark_channel_deblur.quality import kernel_component_count, restoration_score
from dark_channel_deblur.refinement import reblur_image


MOTION_KEY = "motion_constrained"
RGAC_KEY = "rgac"
MOTION_METHOD = {
    "name": "Motion-Constrained",
    "description": (
        "Independent blind restoration whose PSF optimization is restricted to a thin "
        "connected motion-trajectory corridor."
    ),
}
RGAC_METHOD = {
    "name": "Residual-Guided Adaptive Consensus",
    "description": (
        "Reference-free spatial consensus over robust, conservative, Annealed PnP, "
        "and Dual-Extreme candidates using local reblur/artifact confidence maps."
    ),
}
METHOD_ORDER = ("baseline", MOTION_KEY, "annealed_pnp", "extreme_channel", RGAC_KEY)
KERNEL_ROLES = {
    "baseline": "Operational inference PSF from the free 2-D blind optimizer.",
    MOTION_KEY: "Operational inference PSF from the trajectory-constrained optimizer.",
    "annealed_pnp": "Diagnostic PSF refit from the final PnP latent; not fed back into restoration.",
    "extreme_channel": "Diagnostic PSF refit from the final Dual-Extreme latent; not fed back into restoration.",
    RGAC_KEY: "Diagnostic PSF refit from the final RGAC latent; not fed back into restoration.",
}


def config_from_profile(profile: dict[str, float | int | bool]) -> DeblurConfig:
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


def _gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr


def _diagnostic_kernel(
    observed: np.ndarray,
    restored: np.ndarray,
    config: DeblurConfig,
) -> np.ndarray:
    """Refit one PSF from a final latent image for comparison only."""
    bx, by = valid_gradients(_gray(observed))
    lx, ly = valid_gradients(_gray(restored))
    kernel = estimate_psf(
        bx,
        by,
        lx,
        ly,
        weight=2.0,
        psf_shape=(config.kernel_size, config.kernel_size),
        workers=config.fft_workers,
        max_iter=16,
        tol=1e-5,
        peak_fraction=0.025,
    )
    kernel = prune_kernel(kernel, min_component_mass=0.025)
    kernel = refine_psf_structure(kernel)
    kernel = adjust_psf_center(kernel)
    total = float(kernel.sum())
    if total > 0:
        kernel /= total
    return kernel.astype(np.float32)


def _save_kernel(path, kernel: np.ndarray) -> None:
    visual = kernel / max(float(kernel.max()), 1e-12)
    benchmark.write_image(path, visual)


def _kernel_diagnostics(
    kernel: np.ndarray,
    legacy_kernel: np.ndarray | None,
) -> dict[str, float | int | None]:
    diag = psf_plausibility(kernel)
    legacy = benchmark.kernel_metrics(kernel, legacy_kernel)
    return {
        "component_count": int(kernel_component_count(kernel)),
        "largest_component_mass": float(diag.largest_component_mass),
        "secondary_component_mass": float(diag.secondary_component_mass),
        "anisotropy": float(diag.anisotropy),
        "off_axis_mass": float(diag.off_axis_mass),
        "weak_line_mass": float(diag.weak_line_mass),
        "plausibility_score": float(diag.score),
        "legacy_correlation": None if legacy is None else float(legacy["correlation"]),
        "legacy_l1_distance": None if legacy is None else float(legacy["l1_distance"]),
    }


def _extend_metrics(
    row: dict[str, float | None],
    observed: np.ndarray,
    restored: np.ndarray,
    operational_kernel: np.ndarray,
    diagnostic_kernel: np.ndarray,
    config: DeblurConfig,
    legacy_kernel: np.ndarray | None,
) -> None:
    predicted = reblur_image(restored, operational_kernel, workers=config.fft_workers)
    blind_score, artifact = restoration_score(observed, restored, predicted)
    diagnostic_predicted = reblur_image(
        restored,
        diagnostic_kernel,
        workers=config.fft_workers,
    )
    row.update(
        {
            "blind_score": float(blind_score),
            "operational_reblur_rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
            "diagnostic_reblur_rmse": float(
                np.sqrt(np.mean((diagnostic_predicted - observed) ** 2))
            ),
            "edge_ratio": float(artifact.edge_ratio),
            "noise_ratio": float(artifact.noise_ratio),
            "highpass_ratio": float(artifact.highpass_ratio),
            "clipping_growth": float(artifact.clipping_growth),
            "kernel": _kernel_diagnostics(diagnostic_kernel, legacy_kernel),
        }
    )


def _normalized(values: dict[str, float]) -> dict[str, float]:
    low = min(values.values())
    high = max(values.values())
    if high - low <= 1e-12:
        return {key: 0.0 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def _assign_reference_free_winner(case: dict[str, object]) -> str:
    metrics = case["metrics"]
    blind = _normalized({key: float(metrics[key]["blind_score"]) for key in METHOD_ORDER})
    reblur = _normalized(
        {key: float(metrics[key]["diagnostic_reblur_rmse"]) for key in METHOD_ORDER}
    )
    plausibility = _normalized(
        {key: float(metrics[key]["kernel"]["plausibility_score"]) for key in METHOD_ORDER}
    )
    fragmentation = _normalized(
        {key: float(metrics[key]["kernel"]["component_count"]) for key in METHOD_ORDER}
    )
    for key in METHOD_ORDER:
        metrics[key]["reference_free_score"] = (
            0.60 * blind[key]
            + 0.20 * reblur[key]
            + 0.15 * plausibility[key]
            + 0.05 * fragmentation[key]
        )
    winner = min(METHOD_ORDER, key=lambda key: float(metrics[key]["reference_free_score"]))
    case["winner"] = winner
    return winner


def _fmt(value: float | int | None, digits: int = 4) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _ranking_table(aggregate: dict[str, dict[str, object]], overall_winner: str) -> str:
    rows = []
    for key in METHOD_ORDER:
        row = aggregate[key]
        label = " ★ overall" if key == overall_winner else ""
        rows.append(
            "<tr>"
            f"<td><strong>{html.escape(benchmark.METHODS[key]['name'])}</strong>{label}</td>"
            f"<td>{int(row['win_count'])}</td>"
            f"<td>{float(row['reference_free_score_mean']):.3f}</td>"
            f"<td>{float(row['blind_score_mean']):.5f}</td>"
            f"<td>{float(row['diagnostic_reblur_rmse_mean']):.5f}</td>"
            f"<td>{float(row['kernel_component_count_mean']):.2f}</td>"
            f"<td>{float(row['kernel_plausibility_score_mean']):.4f}</td>"
            f"<td>{_fmt(row['kernel_legacy_correlation_mean'], 4)}</td>"
            "</tr>"
        )
    return (
        '<div class="winner-ranking"><h2>Reference-free ranking</h2>'
        '<p>Winner score uses no legacy pixels: 60% guarded blind score + 20% diagnostic '
        'reblur RMSE + 15% PSF plausibility + 5% PSF fragmentation. Each term is '
        'min-max normalized per image; lower is better.</p>'
        '<div class="table-wrap"><table><thead><tr><th>Method</th><th>Wins</th>'
        '<th>Score ↓</th><th>Blind score ↓</th><th>Diagnostic RMSE ↓</th>'
        '<th>PSF components ↓</th><th>PSF penalty ↓</th><th>Kernel corr.* ↑</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div></div>'
    )


def _augment_report(captures: list[dict[str, object]], original_deblur) -> None:
    report_path = benchmark.RESULTS / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cases = report["images"]
    if len(cases) != len(captures):
        raise RuntimeError("capture count does not match generated report cases")

    benchmark.METHODS[MOTION_KEY] = MOTION_METHOD
    benchmark.METHODS[RGAC_KEY] = RGAC_METHOD
    benchmark.METHOD_ORDER = METHOD_ORDER
    augmentation_started = time.perf_counter()

    for case, capture in zip(cases, captures, strict=True):
        observed = capture["observed"]
        config = capture["config"]
        baseline_kernel = capture["kernel"]
        outputs = {
            "baseline": capture["baseline"],
            "annealed_pnp": capture["annealed"],
            "extreme_channel": capture["extreme"],
            RGAC_KEY: capture["rgac"],
        }
        case_dir = benchmark.RESULTS / case["result_dir"]

        motion_config = replace(config, kernel_model="motion-trajectory")
        started = time.perf_counter()
        motion, motion_kernel, motion_interim = original_deblur(observed, motion_config)
        motion_time = time.perf_counter() - started
        outputs[MOTION_KEY] = motion
        benchmark.write_image(case_dir / f"{MOTION_KEY}.png", motion)
        benchmark.write_image(case_dir / "interim_motion.png", motion_interim)

        benchmark.write_image(case_dir / f"{RGAC_KEY}.png", outputs[RGAC_KEY])
        case["runtimes_seconds"][MOTION_KEY] = motion_time
        case["runtimes_seconds"][RGAC_KEY] = (
            float(case["runtimes_seconds"]["annealed_pnp"])
            + float(case["runtimes_seconds"]["extreme_channel"])
            + float(capture["rgac_extra_seconds"])
        )
        case["output_shapes"][MOTION_KEY] = list(motion.shape)
        case["output_shapes"][RGAC_KEY] = list(outputs[RGAC_KEY].shape)
        case["rgac"] = asdict(capture["diagnostics"])

        legacy_reference = None
        if case["legacy_reference_status"] == "exact_shape":
            legacy_reference = benchmark.read_image(case_dir / "legacy_result.png")
        legacy_kernel = None
        if case["legacy_kernel_shape"] is not None:
            legacy_kernel = benchmark.read_kernel(case_dir / "legacy_kernel.png")

        motion_row = benchmark.diagnostics(
            motion,
            observed,
            motion_kernel,
            workers=config.fft_workers,
            legacy_reference=legacy_reference,
        )
        rgac_row = benchmark.diagnostics(
            outputs[RGAC_KEY],
            observed,
            baseline_kernel,
            workers=config.fft_workers,
            legacy_reference=legacy_reference,
        )
        case["metrics"][MOTION_KEY] = motion_row
        case["metrics"][RGAC_KEY] = rgac_row

        diagnostic_kernels = {
            "baseline": baseline_kernel,
            MOTION_KEY: motion_kernel,
            "annealed_pnp": _diagnostic_kernel(observed, outputs["annealed_pnp"], config),
            "extreme_channel": _diagnostic_kernel(observed, outputs["extreme_channel"], config),
            RGAC_KEY: _diagnostic_kernel(observed, outputs[RGAC_KEY], config),
        }
        operational_kernels = {
            "baseline": baseline_kernel,
            MOTION_KEY: motion_kernel,
            "annealed_pnp": baseline_kernel,
            "extreme_channel": baseline_kernel,
            RGAC_KEY: baseline_kernel,
        }

        case["kernel_shapes"] = {}
        case["kernel_roles"] = {}
        for key in METHOD_ORDER:
            kernel = np.asarray(diagnostic_kernels[key], dtype=np.float32)
            if kernel.shape != (config.kernel_size, config.kernel_size):
                raise RuntimeError(f"{key} kernel support mismatch for {case['name']}")
            if not np.isfinite(kernel).all() or float(kernel.sum()) <= 0:
                raise RuntimeError(f"{key} produced an invalid diagnostic kernel for {case['name']}")
            kernel /= float(kernel.sum())
            _save_kernel(case_dir / f"{key}_kernel.png", kernel)
            case["kernel_shapes"][key] = list(kernel.shape)
            case["kernel_roles"][key] = KERNEL_ROLES[key]
            _extend_metrics(
                case["metrics"][key],
                observed,
                outputs[key],
                operational_kernels[key],
                kernel,
                config,
                legacy_kernel,
            )

        _assign_reference_free_winner(case)

    metric_sets = {
        key: [case["metrics"][key] for case in cases]
        for key in METHOD_ORDER
    }
    runtime_sets = {
        key: [float(case["runtimes_seconds"][key]) for case in cases]
        for key in METHOD_ORDER
    }
    aggregate = benchmark.aggregate_metrics(metric_sets, runtime_sets)
    for key in METHOD_ORDER:
        rows = metric_sets[key]
        correlations = [
            float(row["kernel"]["legacy_correlation"])
            for row in rows
            if row["kernel"]["legacy_correlation"] is not None
        ]
        aggregate[key].update(
            {
                "win_count": sum(case["winner"] == key for case in cases),
                "reference_free_score_mean": float(
                    mean(float(row["reference_free_score"]) for row in rows)
                ),
                "blind_score_mean": float(mean(float(row["blind_score"]) for row in rows)),
                "diagnostic_reblur_rmse_mean": float(
                    mean(float(row["diagnostic_reblur_rmse"]) for row in rows)
                ),
                "kernel_component_count_mean": float(
                    mean(float(row["kernel"]["component_count"]) for row in rows)
                ),
                "kernel_plausibility_score_mean": float(
                    mean(float(row["kernel"]["plausibility_score"]) for row in rows)
                ),
                "kernel_legacy_correlation_mean": (
                    float(mean(correlations)) if correlations else None
                ),
            }
        )
    # Motion-constrained is a complete independent pipeline, not a refinement stage.
    aggregate[MOTION_KEY]["end_to_end_runtime_seconds_total"] = aggregate[MOTION_KEY][
        "stage_runtime_seconds_total"
    ]
    overall_winner = min(
        METHOD_ORDER,
        key=lambda key: (
            float(aggregate[key]["reference_free_score_mean"]),
            -int(aggregate[key]["win_count"]),
        ),
    )

    report["schema_version"] = 4
    report["benchmark"] = "Full-quality native-resolution five-method deblurring and PSF comparison"
    report["methods"] = benchmark.METHODS
    report["method_order"] = list(METHOD_ORDER)
    report["aggregate"] = aggregate
    report["overall_winner"] = overall_winner
    report["winner_selection"] = {
        "reference_free": True,
        "legacy_inputs_used": False,
        "legacy_metrics_used": False,
        "normalization": "per-image min-max across five methods",
        "lower_is_better": True,
        "weights": {
            "guarded_blind_score": 0.60,
            "diagnostic_reblur_rmse": 0.20,
            "psf_plausibility": 0.15,
            "psf_fragmentation": 0.05,
        },
    }
    report["kernel_comparison"] = {
        "baseline_and_motion": "operational inference kernels",
        "refinement_methods": "diagnostic PSF refits from final latent outputs",
        "diagnostic_refits_used_for_restoration": False,
        "legacy_inputs_used": False,
    }
    report["rgac_design"] = {
        "name": RGAC_METHOD["name"],
        "learned_weights": False,
        "legacy_inputs_used": False,
        "candidate_set": ["baseline", "conservative", "annealed_pnp", "extreme_channel"],
        "selection": "local reference-free residual/artifact soft consensus plus global safety guard",
    }
    report["total_runtime_seconds"] = float(report["total_runtime_seconds"]) + (
        time.perf_counter() - augmentation_started
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    html_text = benchmark.build_html(report, cases, aggregate)
    for case in cases:
        rel = str(case["result_dir"])
        profile = case["profile"]
        old_psf = benchmark.card(
            "Estimated kernel",
            f"{rel}/kernel.png",
            f"{profile['kernel_size']}×{profile['kernel_size']} independently estimated PSF",
            "PSF",
        )
        extra_cards = [
            benchmark.card(
                MOTION_METHOD["name"],
                f"{rel}/{MOTION_KEY}.png",
                "Independent trajectory-constrained blind restoration",
                "Motion",
            ),
            benchmark.card(
                RGAC_METHOD["name"],
                f"{rel}/{RGAC_KEY}.png",
                "Residual-guided adaptive consensus",
                "Consensus",
            ),
        ]
        for key in METHOD_ORDER:
            label = "Inference PSF" if key in {"baseline", MOTION_KEY} else "Diagnostic PSF refit"
            extra_cards.append(
                benchmark.card(
                    f"{benchmark.METHODS[key]['name']} · kernel",
                    f"{rel}/{key}_kernel.png",
                    KERNEL_ROLES[key],
                    label,
                )
            )
        html_text = html_text.replace(old_psf, "".join(extra_cards), 1)
        winner = benchmark.METHODS[case["winner"]]["name"]
        winner_score = float(case["metrics"][case["winner"]]["reference_free_score"])
        notice = (
            '<div class="case-winner"><strong>★ Reference-free winner: '
            f"{html.escape(winner)}</strong> · score {winner_score:.3f}</div>"
        )
        html_text = html_text.replace("</summary>", "</summary>" + notice, 1)

    image_count = int(report["dataset"]["image_count"])
    html_text = html_text.replace("three-method comparison", "five-method + PSF comparison")
    html_text = html_text.replace(
        f"<b>{image_count * 3}</b><span>restored outputs</span>",
        f"<b>{image_count * 5}</b><span>restored outputs</span>",
    )
    html_text = html_text.replace(
        ".methods{display:grid;grid-template-columns:repeat(3,1fr)",
        ".methods{display:grid;grid-template-columns:repeat(5,1fr)",
    )
    extra_css = (
        ".winner-ranking{background:#fff;border:1px solid var(--line);border-radius:18px;"
        "padding:18px;margin-top:28px;box-shadow:var(--shadow)}"
        ".winner-ranking>p{color:var(--muted);max-width:1100px}"
        ".case-winner{margin:0 14px 4px;padding:9px 12px;border-radius:10px;"
        "background:#eaf8f3;color:#176249;border:1px solid #c7eadc}"
    )
    html_text = html_text.replace("</style>", extra_css + "</style>", 1)
    winner_name = benchmark.METHODS[overall_winner]["name"]
    ranking = _ranking_table(aggregate, overall_winner)
    headline = (
        '<div class="notice good"><strong>Overall reference-free winner: '
        f"{html.escape(winner_name)}</strong> · {int(aggregate[overall_winner]['win_count'])} "
        f"of {image_count} image wins · mean score "
        f"{float(aggregate[overall_winner]['reference_free_score_mean']):.3f}. "
        "This is a blind benchmark result, not a clean-ground-truth claim.</div>"
    )
    html_text = html_text.replace(
        "<section><h2>Aggregate results</h2>",
        ranking + "<section><h2>Aggregate results</h2>" + headline,
        1,
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
        f"- Restorations: **{image_count * 5}**\n- PSF visualizations: **{image_count * 5}**\n"
        f"- Reference-free overall winner: **{winner_name}**",
    )
    summary += (
        "\n## Winner selection\n\n"
        "Legacy outputs and kernels are not used for winner selection. Per-image score: "
        "60% guarded blind score + 20% diagnostic reblur RMSE + 15% PSF plausibility + "
        "5% PSF fragmentation, min-max normalized across the five methods.\n"
    )
    summary_path.write_text(summary, encoding="utf-8")


def main() -> int:
    benchmark.config_from_profile = config_from_profile
    benchmark.METHODS["baseline"] = {
        "name": "Adaptive Robust Baseline",
        "description": (
            "Free 2-D blind kernel core plus reference-free ripple detection, conservative "
            "restoration selection, saturation guarding, and gradient-only retry."
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

    _augment_report(captures, original_deblur)
    print(
        "Five-method augmentation complete: output and PSF comparison written to results/report.html.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
