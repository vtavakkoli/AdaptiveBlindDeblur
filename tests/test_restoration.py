from __future__ import annotations

import numpy as np

from dark_channel_deblur.config import DeblurConfig
from dark_channel_deblur.optimization import l0_restoration, ringing_artifacts_removal


def _identity_kernel(size: int = 3) -> np.ndarray:
    k = np.zeros((size, size), dtype=np.float32)
    k[size // 2, size // 2] = 1.0
    return k


def test_l0_restoration_identity_kernel_is_stable() -> None:
    rng = np.random.default_rng(19)
    image = rng.random((24, 25), dtype=np.float32)
    cfg = DeblurConfig(kernel_size=3, max_grad_steps=3, fft_workers=1)
    out = l0_restoration(image, _identity_kernel(), 1e-6, cfg)
    assert out.shape == image.shape
    assert np.isfinite(out).all()
    assert float(np.mean(np.abs(out - image))) < 0.08


def test_ringing_removal_rgb_smoke() -> None:
    rng = np.random.default_rng(23)
    image = rng.random((20, 22, 3), dtype=np.float32)
    cfg = DeblurConfig(kernel_size=3, max_grad_steps=2, fft_workers=1)
    out = ringing_artifacts_removal(image, _identity_kernel(), cfg)
    assert out.shape == image.shape
    assert np.isfinite(out).all()
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0
