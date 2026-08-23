# MATLAB parity audit

This audit compares the current Python implementation with the original MATLAB release function by function. It exists because visual failures in dark/high-contrast motion cases were not explained by the final refinement stage alone.

## Root causes found

1. **Dark-channel assignment was not equivalent.**
   - MATLAB computes `dark_channel` and then calls `assign_dark_channel_to_pixel`.
   - The assignment loops over every overlapping patch sequentially. Each later patch observes changes made by earlier patches.
   - The previous Python optimization found targets independently and applied all selected zero assignments in bulk. That changes the sparse prior itself.
   - The Python implementation now preserves MATLAB column-major tie order and sequential overlapping-patch updates.

2. **Boundary wrapping was approximated.**
   - MATLAB `wrap_boundary_liu` solves a minimum-Laplacian extension with DST Poisson solves.
   - The previous Python code used linear/bilinear interpolation.
   - The Poisson/DST extension is now ported directly.

3. **The image pyramid was different.**
   - MATLAB uses Levin's `downSmpImC`: a specific trimmed Gaussian convolution in `valid` mode followed by bilinear sampling.
   - The previous Python code used OpenCV Gaussian blur plus area resize.
   - The release downsampler is now reproduced.

4. **FFT support sizes were different.**
   - MATLAB uses Cho's `opt_fft_size.m` LUT based on factors 2/3/5/7 with optional 11/13.
   - SciPy `next_fast_len` returns different sizes for many inputs.
   - Release-sized images now use the exact Cho lookup.

5. **Kernel pyramid details differed.**
   - The two-tap initial PSF was one row away from MATLAB's 1-based coordinate conversion.
   - Kernel resizing used centered crop/pad instead of Levin's mass-aware `fixsize` logic.
   - Gradient thresholding approximated `threshold_pxpy_v1` instead of its four-orientation histogram/tail-count rule.
   - These details are now matched.

6. **Saturated-image final restoration was missing.**
   - The MATLAB demo routes saturated cases through `whyte_deconv`/`deconvRL` rather than TV/L0 ringing removal.
   - The uniform-blur Whyte RL path, including smooth saturation response and ringing-prevention masks, is now ported.
   - Benchmark profiles record the original per-image saturation mode.

## Benchmark principle

The MATLAB-parity benchmark disables alternate-kernel and conservative-restoration selection for the baseline. This is intentional: parity must be measured on the translated algorithm itself rather than hidden by a later heuristic. Legacy result/kernel files remain evaluation-only and are not inputs to blind inference.

## Target failure cases

The full report should be inspected particularly closely for:

- `26.png`
- `7_patch_use.png`
- `IMG_0650_small_patch.png`
- `IMG_0664_small_patch.png`
- `IMG_4561.JPG`
- `blurry_7.png`
- `postcard.png`
- `real_leaffiltered.png`

The primary acceptance signal is a substantial improvement in estimated-kernel agreement and visible removal of duplicated-edge/stripe artifacts, while preserving native resolution and independent inference.
