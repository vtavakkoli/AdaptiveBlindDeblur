# Visual parity targets

The MATLAB-parity Docker report must be checked on the following failure classes before merge.

| Case | Previous problem | Expected correction |
|---|---|---|
| `26.png` | Poor PSF/output on saturated night sign | Sequential dark-channel updates + Whyte saturated restoration |
| `7_patch_use.png` | Wrong 85px trajectory | Sequential dark-channel updates, exact pyramid, FFT support, thresholding |
| `IMG_0650_small_patch.png` | Wrong PSF and duplicated edges | Same blind-estimation parity fixes + saturated restoration |
| `IMG_0664_small_patch.png` | Wrong PSF/noisy deconvolution | Same blind-estimation parity fixes + saturated restoration |
| `IMG_4561.JPG` | Dark-scene PSF mismatch | Same blind-estimation parity fixes + saturated restoration |
| `blurry_7.png` | Weak/wrong gradient-only PSF | Exact gradient threshold, kernel initialization/resizing, FFT sizes |
| `postcard.png` | Catastrophically wrong long PSF | Exact dark-channel assignment and multiscale/boundary pipeline |
| `real_leaffiltered.png` | Wrong PSF despite stable output | Exact dark-channel assignment and multiscale/boundary pipeline |

Legacy outputs and kernels are evaluation-only. They are never used to select or construct the current blind estimate.
