# CI image dataset

`docker compose run --rm test` benchmarks every image extracted into `dataset/image/`.

The repository stores `dataset_images_ci.zip`, a compact 192 px preview set derived from every image in the `image/` folder of the supplied CVPR 2016 dark-channel deblurring release. Keeping the CI copies compact makes the full multi-method benchmark reproducible on GitHub Actions and laptops. The original filenames are preserved.

The Docker test extracts the archive automatically before running the benchmark. Generated benchmark outputs are written to `results/`.
