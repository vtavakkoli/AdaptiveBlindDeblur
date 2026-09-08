# Browser restoration quality

The Browser Lab is a local, classical deblurring tool. It retains the five method families while adding a boundary-corrected solver, explicit output resolution, and a cancellable worker. No image or metadata is uploaded. It uses no learned model, external weights, CDN, or runtime dependency.

## Using the lab

1. Open `docs/index.html` with its CSS and three JavaScript files alongside it, or serve the `docs/` directory over HTTP.
2. Keep **Native resolution** for final output. It preserves the decoded width, height, and alpha channel, up to 12 megapixels. Larger sources require explicitly choosing **Preview**, which limits the longest side to 1400 pixels. Files are limited to 50 MB, 40 megapixels, and 16,384 pixels per side. Nothing is silently reduced.
3. Try **Thorough** automatic estimation. **Quick** searches one estimator family with a shorter refinement schedule. These settings affect automatic kernel estimation, not output dimensions.
4. If the estimated blur is unsuitable, choose **Straight motion** and set its length and angle, or **Out of focus** and set a disk radius. These controls use original-image pixels, including when exporting a preview. Positive motion angles rotate clockwise from horizontal.
5. Increase noise suppression to reduce noise amplification. Run again after changing processing settings.
6. Compare the five results at **100% pixels**. Export the selected result as a lossless PNG, or save the settings, estimated kernels, and diagnostics as JSON.

The PNG is an 8-bit-per-channel browser image. Native resolution does not imply preserving the original ICC profile, EXIF metadata, high bit depth, or file encoding. Browser decoding determines orientation and color conversion. Alpha is preserved at native resolution; transparent regions do not have a separate physical image-formation model.

## Numerical changes

- **Correct boundaries:** reflected extension plus a smooth outer taper replaces zero-filled FFT boundaries. A spectral warm start is corrected with preconditioned conjugate gradients against the actual reflected convolution operator and its adjoint. The adjoint folds contributions from reflected pixels back into the image. Merely flipping a kernel does not implement this operation at image borders.
- **Brightness preservation:** first-difference regularization has zero DC penalty. It does not systematically darken constant regions.
- **Noise-aware regularization:** a robust high-pass noise estimate is measured on a native-resolution sample. It controls regularization together with the noise-suppression setting. The sample is a heuristic and may not represent noise across the whole image.
- **Practical dark-channel estimation:** a separable sliding-window argmin replaces repeated quadratic patch scans. Minima are computed from immutable observations. Automatic estimation searches multiple supports and retains coarse candidates if longer refinement produces a worse score on the same observations.
- **Deterministic PnP:** an annealed bilateral prior replaces random noise injection and indiscriminate Gaussian smoothing. This is not a trained diffusion model.
- **Artifact guards:** candidate selection penalizes added clipping relative to the input and permits useful edge recovery. PnP and extreme-channel refinement can fall back to the baseline. RGAC re-evaluates its contributors against a common PSF before fusion.
- **Native tiling:** overlapping tiles use reflected halos, one noise-derived regularization for the image, and raised-cosine blend weights. Five RGBA results are retained for method switching. Storage grows with image pixels; FFT working arrays are bounded by tile and PSF sizes. Very large images can still require substantial memory and several minutes of CPU time.
- **Responsive execution:** hosted pages run the core in a dedicated Web Worker, transfer pixel buffers, report progress, and terminate the worker on cancellation. Run and render generations prevent superseded jobs from replacing a newer image or method. Browsers that disallow workers for `file://` use cooperative processing in the tab; cancellation takes effect between numerical stages there.

## Diagnostics and limits

All displayed method scores are evaluated on the same diagnostic image (at most 320 pixels on its longest side) and common estimated PSF. The JSON report records that resolution. Reblur error measures agreement with the blur model. Edge and high-pass ratios indicate detail/noise changes. Added clipping is the increase in the fraction of pixels with any channel near a limit.

These are heuristics, not ground-truth quality, confidence percentages, or proof that the selected PSF is correct. A wrong kernel can still receive a low score. Automatic estimation cannot guarantee recovery from spatially varying blur, rolling shutter, strong saturation, severe noise, or missing information. Inspect the output and try manual controls when the estimated kernel is unsuitable.

The browser implementation is separate from the Python/Docker research pipeline. This change does not alter dataset assets, benchmark profiles, Python algorithms, or the official native-resolution benchmark protocol.

## Reproducible checks

```bash
node --test tests/browser-quality.cjs
python -m pytest -q tests/test_browser_lab_syntax.py tests/test_report_assets.py
```

The Node suite exercises the actual numerical core and worker message protocol. It checks the reflected forward operator against an independently implemented spatial convolution, DC preservation, recovery from independently blurred synthetic images, clipping diagnostics, dimensions and alpha, overlap coverage, dark-channel minima, input limits, transferable results, and cancellation.

On the deterministic 96 × 80 fixture in that suite, with a **known** kernel and additive seeded noise, the corrected channel solver improved PSNR over its blurred input by:

| Blur | PSNR improvement |
| --- | ---: |
| Horizontal motion, 9 px | 8.81 dB |
| Motion at 32°, 13 px | 6.89 dB |
| Defocus disk, radius 3 px | 5.31 dB |

These small synthetic checks isolate the restoration solver. They are not an automatic blind-deblurring benchmark, a comparison against other software, or a state-of-the-art claim. Run the checked-in suite to reproduce the values. Browser visual testing and the full Python/Docker benchmark are separate validations.
