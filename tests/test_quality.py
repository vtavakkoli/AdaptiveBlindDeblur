from __future__ import annotations

import numpy as np

from dark_channel_deblur.quality import artifact_diagnostics, restoration_score, should_retry_kernel


def _image() -> np.ndarray:
    y, x = np.mgrid[0:64, 0:64]
    base = (0.15 + 0.65 * x / 63.0 + 0.10 * y / 63.0).astype(np.float32)
    image = np.stack([base, base * 0.9, base * 0.8], axis=2)
    image[18:45, 20:44] *= 0.35
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def test_quality_score_penalizes_striping_and_clipping() -> None:
    observed = _image()
    safe = observed.copy()
    striped = observed.copy()
    striped[:, ::2] = np.clip(striped[:, ::2] + 0.45, 0.0, 1.0)

    safe_score, _ = restoration_score(observed, safe, observed)
    bad_score, bad_diag = restoration_score(observed, striped, observed)

    assert bad_score > safe_score
    assert bad_diag.highpass_ratio > 1.0
    assert bad_diag.clipping_growth > 0.0


def test_retry_trigger_detects_fragmented_kernel() -> None:
    observed = _image()
    restored = observed.copy()
    kernel = np.zeros((65, 65), dtype=np.float32)
    kernel[12:15, 12:15] = 1.0
    kernel[30:33, 30:33] = 1.0
    kernel[48:51, 48:51] = 1.0
    kernel /= kernel.sum()

    assert should_retry_kernel(
        observed,
        restored,
        kernel,
        kernel_size=65,
        blind_score=0.01,
    )


def test_artifact_diagnostics_are_neutral_for_identity() -> None:
    observed = _image()
    diag = artifact_diagnostics(observed, observed)
    assert 0.95 <= diag.edge_ratio <= 1.05
    assert diag.clipping_growth == 0.0
