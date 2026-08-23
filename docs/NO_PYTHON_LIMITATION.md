# No Python limitation

No stage identified in the MATLAB release requires MATLAB-specific mathematics. The dark-channel prior, conjugate-gradient PSF solve, DST Poisson boundary extension, TV/L0 optimization, and Whyte saturated Richardson–Lucy update all have direct NumPy/SciPy/OpenCV implementations. The observed mismatch came from non-equivalent substitutions and optimizations, not from a missing Python capability.
