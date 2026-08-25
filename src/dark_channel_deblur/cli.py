from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from .config import DeblurConfig
from .consensus import residual_guided_adaptive_consensus_refine
from .deblur import deblur_image
from .io import read_image, write_image
from .refinement import annealed_pnp_refine, extreme_channel_refine
from .ugdb import ugdb_restore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experimental blind image deblurring with guarded restoration refinements"
    )
    parser.add_argument("input", type=Path, help="Blurred input image")
    parser.add_argument("output", type=Path, help="Deblurred output image")
    parser.add_argument("--kernel-output", type=Path, default=None, help="Optional estimated PSF PNG")
    parser.add_argument("--interim-output", type=Path, default=None, help="Optional interim latent PNG")
    parser.add_argument(
        "--method",
        choices=(
            "baseline",
            "annealed-pnp",
            "extreme-channel",
            "rgac",
            "ugdb-linear",
            "ugdb-null",
            "ugdb-kernel",
            "ugdb-full",
        ),
        default="baseline",
        help=(
            "baseline=robust blind restoration, annealed-pnp=stochastic guarded refinement, "
            "extreme-channel=dual-extreme guarded refinement, rgac=residual-guided adaptive "
            "multi-prior consensus, ugdb-*=uncertainty-guided Gaussian blind-deblurring ablations"
        ),
    )
    parser.add_argument("--kernel-size", type=int, default=25, help="Odd PSF support size")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma used during PSF estimation")
    parser.add_argument("--iterations", type=int, default=5, help="Blind image/kernel alternations per scale")
    parser.add_argument("--lambda-dark", type=float, default=4e-3)
    parser.add_argument("--lambda-grad", type=float, default=4e-3)
    parser.add_argument("--lambda-tv", type=float, default=3e-3)
    parser.add_argument("--lambda-l0", type=float, default=5e-4)
    parser.add_argument("--ring-weight", type=float, default=1.0)
    parser.add_argument(
        "--no-robust",
        action="store_true",
        help="Disable PSF retry and conservative restoration selection.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Preview mode: cap optimization loops and disable robust retries.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for stochastic refinement candidates")
    parser.add_argument(
        "--ugdb-steps",
        type=int,
        default=4,
        help="Gaussian/diffusion-surrogate iterations for UGDB methods.",
    )
    parser.add_argument(
        "--ugdb-kernel-hypotheses",
        type=int,
        default=4,
        help="PSF posterior particles for ugdb-kernel and ugdb-full.",
    )
    parser.add_argument("--opencv-threads", type=int, default=0, help="0 lets OpenCV decide")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.opencv_threads > 0:
        cv2.setNumThreads(args.opencv_threads)
    robust = not args.no_robust and not args.fast
    cfg = DeblurConfig(
        kernel_size=args.kernel_size,
        gamma_correct=args.gamma,
        xk_iter=args.iterations,
        lambda_dark=args.lambda_dark,
        lambda_grad=args.lambda_grad,
        lambda_tv=args.lambda_tv,
        lambda_l0=args.lambda_l0,
        weight_ring=args.ring_weight,
        robust_selection=robust,
        retry_gradient_only=robust,
        conservative_restoration=robust,
        max_grad_steps=12 if args.fast else None,
        max_dark_steps=5 if args.fast else None,
    )
    image = read_image(args.input)
    start = time.perf_counter()
    baseline, kernel, interim = deblur_image(image, cfg)
    ugdb_diag = None
    if args.method == "annealed-pnp":
        result = annealed_pnp_refine(
            image,
            baseline,
            kernel,
            seed=args.seed,
            workers=cfg.fft_workers,
        )
    elif args.method == "extreme-channel":
        result = extreme_channel_refine(
            image,
            baseline,
            kernel,
            workers=cfg.fft_workers,
        )
    elif args.method == "rgac":
        result = residual_guided_adaptive_consensus_refine(
            image,
            baseline,
            kernel,
            seed=args.seed,
            workers=cfg.fft_workers,
        )
    elif args.method.startswith("ugdb-"):
        variant = args.method.removeprefix("ugdb-")
        variant = "nullspace" if variant == "null" else variant
        result, kernel, ugdb_diag = ugdb_restore(
            image,
            baseline,
            kernel,
            variant=variant,
            steps=args.ugdb_steps,
            kernel_hypotheses=args.ugdb_kernel_hypotheses,
            seed=args.seed,
            workers=cfg.fft_workers,
        )
    else:
        result = baseline
    elapsed = time.perf_counter() - start
    write_image(args.output, result)
    if args.kernel_output:
        vis = kernel / max(float(kernel.max()), 1e-12)
        write_image(args.kernel_output, vis)
    if args.interim_output:
        write_image(args.interim_output, interim)
    print(f"Deblurred {args.input} -> {args.output} using {args.method} in {elapsed:.2f}s")
    if ugdb_diag is not None:
        print(
            "UGDB diagnostics: "
            f"observable={ugdb_diag.mean_observable_fraction:.3f}, "
            f"kernel_uncertainty={ugdb_diag.mean_kernel_uncertainty:.5f}, "
            f"score={ugdb_diag.final_score:.6f}, "
            f"kernel_update={ugdb_diag.accepted_kernel_update}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
