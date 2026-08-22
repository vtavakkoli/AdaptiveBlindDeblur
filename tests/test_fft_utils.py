from __future__ import annotations

import numpy as np

from dark_channel_deblur.fft_utils import otf2psf, psf2otf


def test_psf_otf_roundtrip() -> None:
    rng = np.random.default_rng(11)
    psf = rng.random((5, 7), dtype=np.float32)
    otf = psf2otf(psf, (32, 40), workers=1)
    recovered = otf2psf(otf, psf.shape, workers=1)
    np.testing.assert_allclose(recovered, psf, atol=1e-5)
