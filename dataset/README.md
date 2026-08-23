# Full `dataset/image` benchmark

The repository already contains **23 source images** under `dataset/image`, originating from the supplied CVPR 2016 dark-channel deblurring release.

`docker compose run --rm test` validates and benchmarks **every supported file in that folder**. The original committed source files are never modified. For practical CI runtime, each image is resized only in memory to a maximum side length of 192 pixels before the three-method comparison is run.

For each source image the benchmark performs one blind dark-channel PSF estimation and then evaluates:

- the dark-channel baseline restoration;
- the new annealed Gaussian plug-and-play refinement;
- the new extreme-channel guided refinement.

Both new refinements reuse the same baseline PSF so the report compares restoration priors under the same estimated blur model. Outputs, metrics and the final visual comparison are generated under `results/`.

Other folders under `dataset/` are preserved as historical/benchmark assets but are not part of this 23-image Docker smoke/research report.
