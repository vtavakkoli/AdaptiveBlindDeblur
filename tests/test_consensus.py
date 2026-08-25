from __future__ import annotations

import cv2
import numpy as np
import pytest

from dark_channel_deblur import residual_guided_adaptive_consensus_refine
from dark_channel_deblur.quality import restoration_score
from dark_channel_deblur.refinement import reblur_image


def _identity_kernel(size: int = 5) -> np.ndarray:
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, size // 2] = 1.0
    return kernel


def _motion_kernel(size: int = 9) -> np.ndarray:
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    kernel[center, 2 : size - 2] = 1.0
    kernel /= kernel.sum()
    return kernel


def test_rgac_identity_case_preserves_already_consistent_baseline() -> None:
    rng = np.random.default_rng(123)
    observed = rng.random((28, 31, 3), dtype=np.float32)
    baseline = observed.copy()
    kernel = _identity_kernel()

    result, info = residual_guided_adaptive_consensus_refine(
        observed,
        baseline,
        kernel,
        annealed=baseline,
        extreme=baseline,
        workers=1,
        return_diagnostics=True,
    )

    assert result.shape == observed.shape
    assert np.isfinite(result).all()
    assert np.allclose(result, baseline, atol=1e-6)
    assert info.accepted_variant == "baseline_fallback"
    assert 0.0 <= info.psf_confidence <= 1.0
    assert set(info.candidate_scores) == {
        "baseline",
        "conservative",
        "annealed_pnp",
        "extreme_channel",
    }
    assert set(info.mean_weights) == set(info.candidate_scores)
    assert sum(info.mean_weights.values()) == pytest.approx(1.0, abs=2e-4)


def test_rgac_never_accepts_a_worse_reference_free_global_score() -> None:
    clean = np.zeros((42, 46, 3), dtype=np.float32)
    clean[8:34, 10:36] = (0.78, 0.42, 0.18)
    clean[16:26, 17:29] = (0.15, 0.82, 0.92)
    kernel = _motion_kernel()
    observed = cv2.GaussianBlur(clean, (7, 7), 1.2).astype(np.float32)

    # A modest unsharp baseline plus deliberately different detail candidates gives
    # RGAC something real to arbitrate without running expensive PnP in this unit test.
    smooth = cv2.GaussianBlur(observed, (0, 0), 0.9)
    baseline = np.clip(observed + 0.55 * (observed - smooth), 0.0, 1.0).astype(np.float32)
    annealed = cv2.GaussianBlur(baseline, (0, 0), 0.55).astype(np.float32)
    extreme = np.clip(baseline + 0.45 * (baseline - smooth), 0.0, 1.0).astype(np.float32)

    result = residual_guided_adaptive_consensus_refine(
        observed,
        baseline,
        kernel,
        annealed=annealed,
        extreme=extreme,
        workers=1,
    )

    base_reblur = reblur_image(baseline, kernel, workers=1)
    result_reblur = reblur_image(result, kernel, workers=1)
    base_score, _ = restoration_score(observed, baseline, base_reblur)
    result_score, _ = restoration_score(observed, result, result_reblur)

    assert result.shape == observed.shape
    assert np.isfinite(result).all()
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0
    assert float(result_score) <= float(base_score) + 1e-7


def test_rgac_rejects_mismatched_precomputed_candidates() -> None:
    image = np.zeros((20, 22, 3), dtype=np.float32)
    kernel = _identity_kernel()
    with pytest.raises(ValueError, match="annealed candidate"):
        residual_guided_adaptive_consensus_refine(
            image,
            image,
            kernel,
            annealed=np.zeros((10, 10, 3), dtype=np.float32),
            extreme=image,
            workers=1,
        )
