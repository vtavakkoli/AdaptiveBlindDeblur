# debluring — high-performance C++ blind image deblurring

A modern C++20 implementation of the dark-channel-prior blind deblurring method from:

> Jinshan Pan, Deqing Sun, Hanspeter Pfister, and Ming-Hsuan Yang, **Blind Image Deblurring Using Dark Channel Prior**, CVPR 2016.

The supplied MATLAB reference code is useful for research, but it spends substantial time in repeated interpreted loops, dense dark-channel scans, and FFT setup. This implementation keeps the same multiscale blind-deconvolution structure and L0/dark-channel priors while restructuring the hot paths for native execution.

## Performance-oriented design

- **FFTW3 real-to-complex 2D FFTs** with cached plans and threaded execution.
- **O(N) dark-channel arg-min filter** using monotonic deques instead of scanning every 35×35 patch.
- **OpenMP** over image channels, spectral arithmetic, morphology passes, and expensive pixel loops.
- **Half-spectrum storage** for real FFTs, reducing spectral memory and arithmetic.
- **Optimized FFT sizes** composed of small prime factors (2/3/5/7/11/13).
- **LTO/IPO + `-O3`** in Release builds; optional `-march=native` for machine-specific binaries.
- No MATLAB runtime and no heavyweight GUI dependency.

The implementation uses reflected FFT padding rather than the original MATLAB Poisson boundary wrapper. This is deliberate: it removes a large preprocessing cost while retaining stable edge behavior for the iterative solver.

## Build

Dependencies: CMake 3.20+, C++20 compiler, FFTW3, libpng, libjpeg. OpenMP is used automatically when available.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDEBLUR_NATIVE_ARCH=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## CLI

```bash
./build/deblur input.png output.png \
  --kernel-size 25 \
  --kernel-out kernel.png \
  --interim-out interim.png
```

The important parameters retain the MATLAB meaning:

| Option | Meaning | Default |
|---|---|---:|
| `--kernel-size` | odd blur-kernel width/height | 25 |
| `--iterations` | latent/kernel alternations per scale | 5 |
| `--gamma` | input gamma correction | 1.0 |
| `--lambda-dark` | L0 dark-channel weight | 0.004 |
| `--lambda-grad` | L0 gradient weight | 0.004 |
| `--lambda-tv` | final TV reconstruction weight | 0.003 |
| `--lambda-l0` | final L0 reconstruction weight | 0.0005 |
| `--ringing-weight` | ringing suppression strength | 1.0 |

For quick experiments, `--fast` uses two kernel/image iterations per scale and a lower continuation ceiling.

## Docker Compose validation

The Compose `test` service builds the same Release configuration and executes the unit/integration test suite:

```bash
docker compose up --build --abort-on-container-exit test
```

To process an image, place it at `data/input.png`, then run:

```bash
docker compose run --rm deblur
```

Outputs are written to `data/output.png` and `data/kernel.png`.

## Tests

The test suite checks:

- kernel positivity and normalization,
- FFT convolution alignment and energy conservation,
- PNG encode/decode round-trip,
- non-blind reconstruction improvement on a deterministic synthetic blur,
- end-to-end blind-kernel estimation smoke behavior (finite normalized kernel and valid output).

## Notes on the reference algorithm

Blind deblurring is non-convex. As noted by the original authors, difficult images may need a different kernel size or gamma setting. For saturated/strongly clipped images, the original package optionally invokes Whyte et al.'s specialized non-uniform deconvolution; this C++ pipeline instead uses the same fast TV/L0 final reconstruction path for all inputs.

## Citation

If this implementation is used to generate results for an academic publication, cite the CVPR 2016 paper above. The uploaded MATLAB package also requests citation of that work.
