from __future__ import annotations

import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "real_img2"


def _read(path: Path, gray: bool = False):
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    assert image is not None, f"missing regression asset: {path}"
    return image


def test_real_img2_reference_assets_are_complete() -> None:
    expected = [
        EXAMPLE / "input_compare.jpg",
        EXAMPLE / "matlab_reference" / "result_compare.jpg",
        EXAMPLE / "matlab_reference" / "kernel.png",
        EXAMPLE / "python" / "full_result_compare.jpg",
        EXAMPLE / "python" / "full_kernel.png",
        EXAMPLE / "python" / "fast_result_compare.jpg",
        EXAMPLE / "python" / "fast_kernel.png",
        EXAMPLE / "metrics.json",
    ]
    assert all(path.is_file() for path in expected)


def test_reference_preview_dimensions_match() -> None:
    images = [
        _read(EXAMPLE / "input_compare.jpg"),
        _read(EXAMPLE / "matlab_reference" / "result_compare.jpg"),
        _read(EXAMPLE / "python" / "full_result_compare.jpg"),
        _read(EXAMPLE / "python" / "fast_result_compare.jpg"),
    ]
    assert all(image.shape == (120, 90, 3) for image in images)
    kernels = [
        _read(EXAMPLE / "matlab_reference" / "kernel.png", True),
        _read(EXAMPLE / "python" / "full_kernel.png", True),
        _read(EXAMPLE / "python" / "fast_kernel.png", True),
    ]
    assert all(kernel.shape == (25, 25) for kernel in kernels)


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
