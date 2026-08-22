#pragma once

#include "deblur/image.hpp"

#include <string>
#include <vector>

namespace deblur {

struct Kernel {
    int width = 0;
    int height = 0;
    std::vector<double> values;

    Kernel() = default;
    Kernel(int h, int w, double value = 0.0)
        : width(w), height(h), values(static_cast<std::size_t>(h) * w, value) {}

    double& at(int y, int x) { return values[static_cast<std::size_t>(y) * width + x]; }
    const double& at(int y, int x) const { return values[static_cast<std::size_t>(y) * width + x]; }
};

struct Options {
    int kernelSize = 25;
    int iterationsPerScale = 5;
    double gamma = 1.0;
    double kernelThresholdDivisor = 20.0;

    double lambdaDark = 4e-3;
    double lambdaGrad = 4e-3;
    double lambdaTv = 3e-3;
    double lambdaL0 = 5e-4;
    double ringingWeight = 1.0;

    int darkPatchSize = 35;
    double l0Kappa = 2.0;
    double betaMax = 1e5;
    int cgIterations = 20;
    double cgTolerance = 1e-5;
    bool verbose = true;
};

struct Result {
    Kernel kernel;
    Image interim;
    Image latent;
};

Result deblur(const Image& blurred, const Options& options = {});
Image deconvolveNonBlind(const Image& blurred, const Kernel& kernel, const Options& options = {});
Image convolve(const Image& image, const Kernel& kernel);
Kernel normalizeKernel(Kernel kernel);
Image kernelToImage(const Kernel& kernel, bool normalizeForDisplay = true);
double meanSquaredError(const Image& a, const Image& b);
double psnr(const Image& reference, const Image& candidate, double peak = 1.0);

} // namespace deblur
