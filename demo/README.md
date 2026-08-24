# Browser Lab

`index.html` is a single-file, dependency-free interactive deblurring playground.

## Run

Open `demo/index.html` directly in a modern browser. No web server, package install, CDN, or backend is required.

The page provides:

- drag-and-drop image upload;
- before/after comparison slider;
- motion-kernel length and angle controls;
- lightweight gradient-based angle estimation;
- iterative data-consistency restoration;
- smoothness regularization;
- optional dark/bright-extrema detail protection;
- artifact-safe update attenuation;
- PSF visualization;
- runtime / edge-gain / residual diagnostics;
- PNG export;
- presets for balanced, long-motion, night/saturated, and fine-detail use.

## Important scope note

The Browser Lab is an **interactive approximation** designed for immediate experimentation. It is not a JavaScript reimplementation of the full multi-scale blind PSF estimator in the Python package.

For the authoritative full-quality algorithm and native-resolution benchmark, use:

```bash
docker compose run --rm test
```

or the `dark-channel-deblur` Python CLI.

The browser demo defaults to a 512-pixel working dimension for responsiveness. Users may explicitly choose original-size processing from the UI. The repository benchmark itself never resizes evaluation images.
