#!/usr/bin/env python3
"""Compare Python deblurring output with the authors' released MATLAB reference.

The reference image is not ground truth. PSNR/SSIM here measure agreement with the
released MATLAB result for the same blurred input and parameters.
"""
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
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Standard 11x11 Gaussian-window SSIM, averaged over channels."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.ndim == 2:
        a = a[..., None]
        b = b[..., None]

    c1 = 0.01**2
    c2 = 0.03**2
    values: list[float] = []
    for channel in range(a.shape[2]):
        x = a[..., channel]
        y = b[..., channel]
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sigma_x2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
        sigma_y2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy
        score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
        )
        values.append(float(np.mean(score[5:-5, 5:-5])))
    return float(np.mean(values))


def sharpness(image: np.ndarray) -> dict[str, float]:
    if image.ndim == 3:
        gray_u8 = cv2.cvtColor(
            np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8),
            cv2.COLOR_RGB2GRAY,
        )
        gray = gray_u8.astype(np.float64) / 255.0
    else:
        gray = image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return {
        "laplacian_variance": float(laplacian.var()),
        "mean_gradient_magnitude": float(np.mean(np.hypot(gx, gy))),
    }


def kernel_agreement(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    if candidate.shape != reference.shape:
        raise ValueError(f"kernel shape mismatch: {candidate.shape} vs {reference.shape}")
    candidate = candidate / max(float(candidate.sum()), 1e-12)
    reference = reference / max(float(reference.sum()), 1e-12)
    return {
        "correlation": float(np.corrcoef(candidate.ravel(), reference.ravel())[0, 1]),
        "l1_distance": float(np.abs(candidate - reference).sum()),
    }


def compare(
    blurred: Path,
    reference_result: Path,
    candidate_result: Path,
    reference_kernel: Path | None = None,
    candidate_kernel: Path | None = None,
) -> dict[str, object]:
    input_image = load_image(blurred)
    reference = load_image(reference_result)
    candidate = load_image(candidate_result)
    report: dict[str, object] = {
        "reference_note": "Released MATLAB result; not ground truth.",
        "candidate_vs_matlab_reference": {
            "psnr_db": psnr(candidate, reference),
            "ssim": ssim(candidate, reference),
        },
        "blurred_input_vs_matlab_reference": {
            "psnr_db": psnr(input_image, reference),
            "ssim": ssim(input_image, reference),
        },
        "sharpness": {
            "blurred_input": sharpness(input_image),
            "matlab_reference": sharpness(reference),
            "candidate": sharpness(candidate),
        },
    }
    if reference_kernel is not None and candidate_kernel is not None:
        report["kernel_agreement_with_matlab_reference"] = kernel_agreement(
            load_image(candidate_kernel, grayscale=True),
            load_image(reference_kernel, grayscale=True),
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blurred", type=Path, required=True)
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--candidate-result", type=Path, required=True)
    parser.add_argument("--reference-kernel", type=Path)
    parser.add_argument("--candidate-kernel", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = compare(
        args.blurred,
        args.reference_result,
        args.candidate_result,
        args.reference_kernel,
        args.candidate_kernel,
    )
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
