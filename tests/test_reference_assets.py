from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "real_img2"


def _read(path: Path, gray: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    assert image is not None, f"missing regression asset: {path}"
    return image


def test_real_img2_reference_assets_are_complete() -> None:
    expected = [
        EXAMPLE / "input_preview.jpg",
        EXAMPLE / "matlab_reference" / "result_preview.jpg",
        EXAMPLE / "matlab_reference" / "kernel.png",
        EXAMPLE / "python" / "full_result_preview.jpg",
        EXAMPLE / "python" / "full_kernel.png",
        EXAMPLE / "python" / "fast_result_preview.jpg",
        EXAMPLE / "python" / "fast_kernel.png",
        EXAMPLE / "metrics.json",
    ]
    assert all(path.is_file() for path in expected)


def test_reference_images_have_matching_dimensions() -> None:
    blurred = _read(EXAMPLE / "input_preview.jpg")
    matlab = _read(EXAMPLE / "matlab_reference" / "result_preview.jpg")
    full = _read(EXAMPLE / "python" / "full_result_preview.jpg")
    fast = _read(EXAMPLE / "python" / "fast_result_preview.jpg")
    assert blurred.shape == matlab.shape == full.shape == fast.shape == (160, 120, 3)

    matlab_kernel = _read(EXAMPLE / "matlab_reference" / "kernel.png", gray=True)
    full_kernel = _read(EXAMPLE / "python" / "full_kernel.png", gray=True)
    fast_kernel = _read(EXAMPLE / "python" / "fast_kernel.png", gray=True)
    assert matlab_kernel.shape == full_kernel.shape == fast_kernel.shape == (25, 25)


def test_reference_metrics_show_close_matlab_agreement() -> None:
    metrics = json.loads((EXAMPLE / "metrics.json").read_text(encoding="utf-8"))
    agreement = metrics["agreement_with_matlab_reference"]
    kernels = metrics["kernel_agreement_with_matlab_reference"]

    assert agreement["python_full"]["ssim"] > 0.95
    assert agreement["python_fast"]["ssim"] > 0.97
    assert agreement["python_full"]["psnr_db"] > 30.0
    assert agreement["python_fast"]["psnr_db"] > 34.0
    assert agreement["python_fast"]["ssim"] > agreement["blurred_input"]["ssim"]
    assert kernels["python_full"]["correlation"] > 0.85
    assert kernels["python_fast"]["correlation"] > 0.95
