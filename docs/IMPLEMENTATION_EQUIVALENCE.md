# Implementation-equivalence checklist

- [x] Sequential `assign_dark_channel_to_pixel` semantics
- [x] MATLAB column-major minimum tie order
- [x] 35×35 dark-channel support
- [x] Liu minimum-Laplacian boundary wrapping
- [x] Levin `downSmpImC` pyramid construction
- [x] Cho `opt_fft_size` lookup for release-sized images
- [x] Original two-tap PSF initialization coordinates
- [x] Mass-aware `fixsize` after PSF resize
- [x] Four-orientation `threshold_pxpy_v1` histogram rule
- [x] MATLAB RGB-to-gray coefficients
- [x] Whyte saturated-image RL branch and per-image saturation modes
- [ ] Full 23-image regenerated report visually inspected

The final unchecked item is intentionally a merge gate, not an implementation task.
