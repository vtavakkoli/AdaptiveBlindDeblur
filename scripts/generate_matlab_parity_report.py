#!/usr/bin/env python3
from __future__ import annotations

import generate_report as benchmark

from dark_channel_deblur import DeblurConfig


def config_from_profile(profile: dict[str, float | int | bool]) -> DeblurConfig:
    """Build the exact configured baseline used for MATLAB-parity evaluation.

    Robust alternate-kernel/restoration selectors are intentionally disabled here:
    this benchmark is meant to expose implementation differences directly rather
    than hide them behind a candidate-selection heuristic.
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
        robust_selection=False,
        retry_gradient_only=False,
        conservative_restoration=False,
        max_grad_steps=None,
        max_dark_steps=None,
        fft_workers=-1,
    )


def main() -> int:
    benchmark.config_from_profile = config_from_profile
    benchmark.METHODS["baseline"] = {
        "name": "MATLAB-Parity Dark-Channel Baseline",
        "description": (
            "Release-equivalent dark-channel projection, multi-scale PSF estimation, "
            "Liu boundary wrapping, and the original final restoration branch: "
            "TV/L0 for ordinary scenes or Whyte RL for saturated scenes."
        ),
    }
    benchmark.generate_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
