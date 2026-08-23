from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DeblurConfig:
    """Configuration for blind PSF estimation and final deconvolution."""

    kernel_size: int = 25
    lambda_dark: float = 4e-3
    lambda_grad: float = 4e-3
    gamma_correct: float = 1.0
    xk_iter: int = 5
    k_thresh: float = 20.0
    prescale: float = 1.0

    # Final restoration regularization.
    lambda_tv: float = 3e-3
    lambda_l0: float = 5e-4
    weight_ring: float = 1.0
    saturated: bool = False
    saturation_iterations: int = 50

    # Sparse local-extrema / gradient optimization settings.
    dark_patch_size: int = 35
    kappa: float = 2.0
    beta_max_grad: float = 1e5
    beta_max_pixel: float = 8.0

    # Robust inference is optional and is kept separate from MATLAB-equivalence
    # fixes. Benchmark profiles can disable these heuristics when measuring parity.
    robust_selection: bool = True
    retry_gradient_only: bool = True
    conservative_restoration: bool = True

    # Optional caps are intended only for previews and small unit tests.
    max_grad_steps: int | None = None
    max_dark_steps: int | None = None

    # scipy.fft workers. -1 means all available CPU workers.
    fft_workers: int = -1

    def validate(self) -> None:
        if self.kernel_size < 3 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        if self.dark_patch_size < 3 or self.dark_patch_size % 2 == 0:
            raise ValueError("dark_patch_size must be an odd integer >= 3")
        if self.gamma_correct <= 0:
            raise ValueError("gamma_correct must be > 0")
        if self.xk_iter < 1:
            raise ValueError("xk_iter must be >= 1")
        if self.kappa <= 1:
            raise ValueError("kappa must be > 1")
        if self.prescale <= 0:
            raise ValueError("prescale must be > 0")
        if self.saturation_iterations < 1:
            raise ValueError("saturation_iterations must be >= 1")
