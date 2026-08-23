from __future__ import annotations

from dark_channel_deblur.fft_utils import fast_shape


def test_cho_fft_lookup_values_that_differ_from_scipy() -> None:
    assert fast_shape((1, 1), (13, 13)) == (13, 13)
    assert fast_shape((13, 13), (14, 14)) == (26, 26)
    assert fast_shape((52, 52), (14, 14)) == (65, 65)
    assert fast_shape((88, 88), (14, 14)) == (104, 104)
