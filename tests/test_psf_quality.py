from __future__ import annotations

import numpy as np

from dark_channel_deblur.psf_quality import psf_plausibility, refine_psf_structure


def _normalize(kernel: np.ndarray) -> np.ndarray:
    kernel = np.maximum(kernel.astype(np.float32), 0.0)
    return kernel / kernel.sum()


def test_spurious_horizontal_branch_is_suppressed() -> None:
    size = 85
    kernel = np.zeros((size, size), dtype=np.float32)

    # Strong curved/mostly vertical motion trajectory.
    for y in range(18, 70):
        x = 43 + int(round(5.0 * np.sin((y - 18) / 52.0 * np.pi)))
        kernel[y, x] += 1.0
        if x + 1 < size:
            kernel[y, x + 1] += 0.45

    # Weak, long horizontal spur similar to the remaining 7_patch_use failure.
    kernel[43, 7:78] += 0.035
    kernel = _normalize(kernel)

    before = psf_plausibility(kernel)
    refined = refine_psf_structure(kernel)
    after = psf_plausibility(refined)

    horizontal_before = float(kernel[43, 7:78].sum())
    horizontal_after = float(refined[43, 7:78].sum())
    assert horizontal_after < horizontal_before * 0.80
    assert after.weak_line_mass <= before.weak_line_mass
    assert np.isclose(refined.sum(), 1.0, atol=1e-6)


def test_detail_preserving_cleanup_keeps_connected_low_amplitude_tail() -> None:
    size = 101
    kernel = np.zeros((size, size), dtype=np.float32)

    # Main trajectory plus a weaker but physically connected curved continuation.
    for index in range(52):
        y = 21 + index
        x = 50 + int(round(11.0 * np.sin(index / 51.0 * 1.35)))
        kernel[y, x] += 1.0
    for index in range(18):
        y = 72 + index
        x = 61 + int(round(0.45 * index))
        if y < size and x < size:
            kernel[y, x] += 0.065
    kernel = _normalize(kernel)

    tail_before = float(kernel[72:90, 60:72].sum())
    refined = refine_psf_structure(kernel)
    tail_after = float(refined[72:90, 60:72].sum())

    assert tail_before > 0
    assert tail_after >= tail_before * 0.65
    assert np.isclose(refined.sum(), 1.0, atol=1e-6)


def test_complex_curved_kernel_is_not_collapsed_to_single_line() -> None:
    size = 65
    kernel = np.zeros((size, size), dtype=np.float32)

    # Curved arc with meaningful secondary thickness, representative of wall-like PSFs.
    for angle in np.linspace(-0.35, 1.30, 70):
        y = int(round(34 + 19 * np.sin(angle)))
        x = int(round(29 + 24 * np.cos(angle)))
        if 1 <= y < size - 1 and 1 <= x < size - 1:
            kernel[y, x] += 1.0
            kernel[y + 1, x] += 0.12
    kernel = _normalize(kernel)

    support_before = int(np.count_nonzero(kernel > kernel.max() * 0.015))
    refined = refine_psf_structure(kernel)
    support_after = int(np.count_nonzero(refined > refined.max() * 0.015))

    assert support_after >= int(support_before * 0.70)
    assert np.isclose(refined.sum(), 1.0, atol=1e-6)


def test_plausibility_score_penalizes_weak_orthogonal_spur() -> None:
    size = 65
    clean = np.zeros((size, size), dtype=np.float32)
    for y in range(12, 55):
        clean[y, 31 + (y // 18)] = 1.0
    clean = _normalize(clean)

    branchy = clean.copy()
    branchy[33, 5:60] += clean.max() * 0.025
    branchy = _normalize(branchy)

    assert psf_plausibility(branchy).score > psf_plausibility(clean).score
