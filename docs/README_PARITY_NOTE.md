# MATLAB-parity branch note

This branch intentionally favors algorithmic equivalence over the previous speed shortcuts for the baseline benchmark. Once parity is demonstrated, any acceleration should be introduced one transformation at a time with an equivalence regression test so that optimization cannot silently change the dark-channel prior or PSF trajectory again.
