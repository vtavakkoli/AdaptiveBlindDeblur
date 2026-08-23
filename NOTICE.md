# Notice and provenance

This repository contains an independently maintained Python deblurring implementation together with source images and legacy result/reference assets that predate the current codebase.

## Current implementation

The current software is an experimental blind-deblurring framework with adaptive PSF estimation and two guarded restoration refinements. It is not presented as a reproduction or port of a specific published paper.

## Dataset and legacy assets

Files under `dataset/`, `examples/`, and legacy result folders may originate from earlier research/demo packages or previous experiments.

They are retained for regression, comparison, and development convenience. Their inclusion does not assert new ownership and does not automatically grant redistribution rights.

The current benchmark treats `dataset/results/` as **evaluation-only**. Legacy result pixels and legacy kernel values are not supplied to the current restoration algorithms.

## Licensing status

No repository-wide `LICENSE` file is currently present. The repository therefore should not be interpreted as granting a general open-source license by default.

Before public redistribution or third-party reuse, the repository owner should select an appropriate license for original code and separately confirm that bundled dataset/reference assets may be redistributed under compatible terms.

A future license for the Python source does not automatically change the licensing status of pre-existing dataset or legacy assets.

## Citation

Machine-readable citation metadata for the **current software repository** is provided in `CITATION.cff`.
