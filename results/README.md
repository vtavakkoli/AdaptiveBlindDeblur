# Generated comparison results

Run:

```bash
docker compose run --rm test
```

The Docker test writes a fresh deblurring benchmark here, including `report.html`, `report.json`, the new Python result/interim/kernel, and copies of the MATLAB and previous Python comparison images used by the report.

Generated files are ignored by Git. `report.html` is also uploaded by GitHub Actions as the `deblurring-comparison-report` artifact.
