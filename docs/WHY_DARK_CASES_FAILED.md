# Why the dark/high-contrast cases failed

The local-minimum map was not the main problem. OpenCV erosion reproduces the scalar dark-channel minimum accurately. The failure came from how those minima were projected back into the latent image.

The MATLAB release is **order dependent**: for every overlapping 35×35 window it checks the current, already-modified patch and changes only the original `J_idx` location when the current minimum differs from the refined dark-channel value. Later windows therefore observe earlier changes.

The previous Python optimization treated windows independently and applied all selected writes in bulk. On difficult examples this modified several times more latent pixels during each dark-channel projection. In dark scenes, where many local minima satisfy the sparse-prior threshold, that difference is especially large. The altered latent image then drives the PSF normal equation toward the wrong trajectory, and the final non-blind deconvolution amplifies that PSF error into duplicated edges and stripes.

This is an implementation-equivalence issue, not a limitation of Python or NumPy. The parity branch restores the sequential update semantics and the other release-specific numerical steps that feed the PSF estimator.
