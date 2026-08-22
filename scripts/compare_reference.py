#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_image(path: Path, grayscale: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(path)
    if not grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float64) / 255.0


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean(np.square(a - b)))
    return float("inf") if mse == 0.0 else float(10.0 * np.log10(1.0 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim == 2:
        a, b = a[..., None], b[..., None]
    c1, c2 = 0.01**2, 0.03**2
    scores = []
    for c in range(a.shape[2]):
        x, y = a[..., c], b[..., c]
        mx = cv2.GaussianBlur(x, (11, 11), 1.5)
        my = cv2.GaussianBlur(y, (11, 11), 1.5)
        vx = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mx * mx
        vy = cv2.GaussianBlur(y * y, (11, 11), 1.5) - my * my
        vxy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mx * my
        score = ((2 * mx * my + c1) * (2 * vxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
        scores.append(float(np.mean(score[5:-5, 5:-5])))
    return float(np.mean(scores))


def kernel_agreement(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    candidate /= max(float(candidate.sum()), 1e-12)
    reference /= max(float(reference.sum()), 1e-12)
    return {
        "correlation": float(np.corrcoef(candidate.ravel(), reference.ravel())[0, 1]),
        "l1_distance": float(np.abs(candidate - reference).sum()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Compare Python deblur output to the released MATLAB reference")
    p.add_argument("--reference-result", type=Path, required=True)
    p.add_argument("--candidate-result", type=Path, required=True)
    p.add_argument("--reference-kernel", type=Path)
    p.add_argument("--candidate-kernel", type=Path)
    args = p.parse_args()
    ref = load_image(args.reference_result)
    out = load_image(args.candidate_result)
    report: dict[str, object] = {"psnr_db": psnr(out, ref), "ssim": ssim(out, ref)}
    if args.reference_kernel and args.candidate_kernel:
        report["kernel"] = kernel_agreement(load_image(args.candidate_kernel, True), load_image(args.reference_kernel, True))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
