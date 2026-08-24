#!/usr/bin/env python3
from __future__ import annotations

import generate_report as benchmark

from dark_channel_deblur import DeblurConfig


def config_from_profile(profile: dict[str, float | int | bool]) -> DeblurConfig:
    """Build the best-quality benchmark configuration from an explicit profile.

    The blind core keeps the MATLAB-equivalent numerical implementation introduced
    by the parity work.  Robust selection is enabled above that core so suspicious
    long-kernel restorations can use a safer restoration or an independent
    gradient-only PSF retry.  Legacy result/kernel pixels remain evaluation-only.
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
    benchmark.generate_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
