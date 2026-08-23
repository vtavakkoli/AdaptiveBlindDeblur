# Full image-folder benchmark

`docker compose run --rm test` benchmarks every supported image from the `image/` folder in the authors' official CVPR 2016 dark-channel deblurring code release.

`prepare_dataset.py` downloads the official `cvpr16_deblurring_code_v1.zip` linked from Jinshan Pan's project page, extracts all **23** supported images into `dataset/image/`, and creates bounded **192 px** lossless working copies for CI. The original stems are preserved. The source images themselves are not duplicated in Git.

The resize is only for practical, deterministic CI runtime; it is not presented as the paper's full-resolution quantitative benchmark. Generated outputs and the HTML/JSON comparison are written to `results/`.

The benchmark runs the baseline DCP restoration plus both research refinements for every input. The two refinements reuse the baseline's estimated PSF so their added restoration prior is compared under the same blur model.
