from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from .config import DeblurConfig
from .deblur import deblur_image
from .io import read_image, write_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Blind image deblurring with a dark-channel prior")
    parser.add_argument("input", type=Path, help="Blurred input image")
    parser.add_argument("output", type=Path, help="Deblurred output image")
    parser.add_argument("--kernel-output", type=Path, default=None, help="Optional estimated kernel PNG")
    parser.add_argument("--interim-output", type=Path, default=None, help="Optional interim latent PNG")
    parser.add_argument("--kernel-size", type=int, default=25)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--lambda-dark", type=float, default=4e-3)
    parser.add_argument("--lambda-grad", type=float, default=4e-3)
    parser.add_argument("--fast", action="store_true", help="Use capped optimization loops for a quicker preview")
    parser.add_argument("--opencv-threads", type=int, default=0, help="0 lets OpenCV decide")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.opencv_threads > 0:
        cv2.setNumThreads(args.opencv_threads)
    cfg = DeblurConfig(
        kernel_size=args.kernel_size,
        gamma_correct=args.gamma,
        xk_iter=args.iterations,
        lambda_dark=args.lambda_dark,
        lambda_grad=args.lambda_grad,
        max_grad_steps=12 if args.fast else None,
        max_dark_steps=5 if args.fast else None,
    )
    image = read_image(args.input)
    start = time.perf_counter()
    result, kernel, interim = deblur_image(image, cfg)
    elapsed = time.perf_counter() - start
    write_image(args.output, result)
    if args.kernel_output:
        vis = kernel / max(float(kernel.max()), 1e-12)
        write_image(args.kernel_output, vis)
    if args.interim_output:
        write_image(args.interim_output, interim)
    print(f"Deblurred {args.input} -> {args.output} in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
