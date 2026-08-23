# Notice and provenance

This repository combines original Python implementation work with research methodology and data/reference assets originating from prior academic code releases.

## Method provenance

The baseline implementation is a Python reimplementation of the method described by:

Jinshan Pan, Deqing Sun, Hanspeter Pfister, and Ming-Hsuan Yang, **Blind Image Deblurring Using Dark Channel Prior**, CVPR 2016.

The repository-specific Extreme-Channel Guided refinement is motivated by later work using dark and bright image extrema. The Annealed Gaussian PnP refinement is a new weight-free experimental variant implemented in this repository.

## Dataset/reference provenance

Files under `dataset/`, `examples/`, and historical result folders may originate from or be derived from the supplied research-code package used to reproduce the original method.

Their inclusion in this repository does not assert new ownership or relicense third-party research assets.

## Licensing status

No repository-wide `LICENSE` file is currently present. That means this repository should **not** be interpreted as granting a general open-source license by default.

Before public redistribution or third-party reuse, the repository owner should select an appropriate license for original code and separately confirm that bundled dataset/reference assets may be redistributed under compatible terms.

Do not assume that a future license for the Python source automatically changes the licensing terms of third-party research assets.

## Citation

Academic use of the dark-channel method should cite the original CVPR 2016 paper. Machine-readable citation information is provided in `CITATION.cff`.
