from __future__ import annotations

import numpy as np

from dark_channel_deblur.refinement import reblur_image
from dark_channel_deblur.ugdb import UGDB_VARIANTS, gaussian_linear_update, ugdb_restore


def _delta_kernel(size: int = 5) -> np.ndarray:
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, size // 2] = 1.0
    return kernel


def _motion_kernel(size: int = 9) -> np.ndarray:
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, 2:-2] = 1.0
    kernel /= float(kernel.sum())
    return kernel


def _synthetic_image(size: int = 48) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.float32)
    image[7:22, 6:30, 0] = 0.85
    image[20:39, 16:42, 1] = 0.65
    image[10:36, 32:37, 2] = 1.0
    ramp = np.linspace(0.0, 0.18, size, dtype=np.float32)
    image += ramp[None, :, None]
    return np.clip(image, 0.0, 1.0)


def test_gaussian_linear_update_reduces_to_closed_form_for_identity_psf() -> None:
    rng = np.random.default_rng(4)
    observed = rng.random((24, 25), dtype=np.float32)
    prior = rng.random((24, 25), dtype=np.float32)
    rho = 0.2

    result, observable = gaussian_linear_update(
        observed,
        prior,
        _delta_kernel(),
        rho=rho,
        noise_variance=1e-4,
        nullspace_gate=False,
        workers=1,
    )

    expected = (observed + rho * prior) / (1.0 + rho)
    assert np.allclose(result, expected, atol=2e-5)
    assert observable > 0.99


def test_gaussian_linear_update_validates_uncertainty_shape() -> None:
    observed = np.zeros((16, 16), dtype=np.float32)
    prior = np.zeros_like(observed)
    try:
        gaussian_linear_update(
            observed,
            prior,
            _delta_kernel(),
            rho=0.1,
            kernel_uncertainty=np.zeros((3, 3), dtype=np.float32),
            workers=1,
        )
    except ValueError as exc:
        assert "kernel_uncertainty" in str(exc)
    else:
        raise AssertionError("expected incompatible uncertainty shape to fail")


def test_all_ugdb_variants_return_finite_native_resolution_results() -> None:
    sharp = _synthetic_image()
    kernel = _motion_kernel()
    observed = reblur_image(sharp, kernel, workers=1)

    for variant in UGDB_VARIANTS:
        restored, updated_kernel, diagnostics = ugdb_restore(
            observed,
            observed,
            kernel,
            variant=variant,
            steps=2,
            kernel_hypotheses=3,
            seed=11,
            workers=1,
        )
        assert restored.shape == observed.shape
        assert updated_kernel.shape == kernel.shape
        assert np.isfinite(restored).all()
        assert np.isfinite(updated_kernel).all()
        assert float(restored.min()) >= 0.0
        assert float(restored.max()) <= 1.0
        assert np.all(updated_kernel >= 0.0)
        assert np.isclose(float(updated_kernel.sum()), 1.0, atol=1e-5)
        assert diagnostics.variant == variant
        assert 0.0 <= diagnostics.mean_observable_fraction <= 1.0
        assert diagnostics.mean_kernel_uncertainty >= 0.0
        assert np.isfinite(diagnostics.final_score)


def test_full_variant_is_reproducible_for_fixed_seed() -> None:
    sharp = _synthetic_image(40)
    kernel = _motion_kernel(7)
    observed = reblur_image(sharp, kernel, workers=1)

    first = ugdb_restore(
        observed,
        observed,
        kernel,
        variant="full",
        steps=2,
        kernel_hypotheses=3,
        seed=19,
        workers=1,
    )
    second = ugdb_restore(
        observed,
        observed,
        kernel,
        variant="full",
        steps=2,
        kernel_hypotheses=3,
        seed=19,
        workers=1,
    )

    assert np.allclose(first[0], second[0], atol=1e-7)
    assert np.allclose(first[1], second[1], atol=1e-7)
    assert first[2] == second[2]


def test_unknown_variant_is_rejected() -> None:
    image = np.zeros((16, 16), dtype=np.float32)
    try:
        ugdb_restore(image, image, _delta_kernel(), variant="magic", workers=1)
    except ValueError as exc:
        assert "variant" in str(exc)
    else:
        raise AssertionError("expected unknown UGDB variant to fail")
