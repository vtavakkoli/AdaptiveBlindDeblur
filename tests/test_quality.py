from __future__ import annotations

import numpy as np

from dark_channel_deblur.quality import (
    ArtifactDiagnostics,
    artifact_diagnostics,
    restoration_score,
    ripple_risk,
    saturation_checkpoint_safe,
    saturation_instability,
    should_retry_kernel,
)


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


def test_long_kernel_ripple_risk_catches_toy_like_failure() -> None:
    diag = ArtifactDiagnostics(
        edge_ratio=2.90,
        noise_ratio=0.80,
        highpass_ratio=2.98,
        clipping_growth=0.033,
    )
    assert ripple_risk(diag, kernel_size=101)


def test_clean_long_motion_is_not_misclassified_as_ripple() -> None:
    # Regression shape modeled on the clean 7_patch_use result: strong useful detail,
    # very low noise, and only modest clipping growth must not trigger a PSF retry.
    diag = ArtifactDiagnostics(
        edge_ratio=1.48,
        noise_ratio=0.05,
        highpass_ratio=2.03,
        clipping_growth=0.010,
    )
    assert not ripple_risk(diag, kernel_size=85)


def test_saturated_instability_requires_joint_edge_and_noise_growth() -> None:
    unstable = ArtifactDiagnostics(2.25, 2.40, 1.75, 0.023)
    clean = ArtifactDiagnostics(1.77, 2.38, 2.52, 0.029)
    assert saturation_instability(unstable)
    assert not saturation_instability(clean)


def test_saturation_checkpoint_budget_accepts_controlled_early_iteration() -> None:
    controlled = ArtifactDiagnostics(1.56, 1.46, 1.23, 0.004)
    assert saturation_checkpoint_safe(controlled)
