from __future__ import annotations

import numpy as np

from dark_channel_deblur.saturation import whyte_deconvolution


def test_whyte_identity_psf_stays_close_to_observation() -> None:
    rng = np.random.default_rng(47)
    image = rng.random((24, 28, 3), dtype=np.float32) * 0.75
    image[8:14, 9:19] = 1.0
    kernel = np.zeros((3, 3), dtype=np.float32)
    kernel[1, 1] = 1.0

    restored = whyte_deconvolution(image, kernel, iterations=2, workers=1)
    assert restored.shape == image.shape
    assert np.isfinite(restored).all()
    # Identity blur has no motion to undo. The smooth saturation model may adjust
    # clipped values slightly, but it must not create a large global change.
    assert float(np.mean(np.abs(restored - image))) < 0.03
