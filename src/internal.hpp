#pragma once

#include "deblur/deblur.hpp"
#include "fft.hpp"

#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

namespace deblur::impl {

constexpr double kEps = 1e-12;

struct Gray {
    int h = 0;
    int w = 0;
    std::vector<double> v;

    Gray() = default;
    Gray(int hh, int ww, double value = 0.0)
        : h(hh), w(ww), v(static_cast<std::size_t>(hh) * ww, value) {}

    double& at(int y, int x) { return v[static_cast<std::size_t>(y) * w + x]; }
    const double& at(int y, int x) const { return v[static_cast<std::size_t>(y) * w + x]; }
};

struct DarkProjection {
    Gray dark;
    std::vector<int> argmin;
    int paddedWidth = 0;
    int radius = 0;
};

Gray imageChannel(const Image& img, int c);
Image grayToImage(const Gray& g);
void setChannel(Image& img, int c, const Gray& g);
int reflect101(int i, int n);
Gray padForFft(const Gray& src, int kh, int kw);
Gray crop(const Gray& src, int h, int w);
Gray resizeBilinear(const Gray& src, int nh, int nw);
Gray gaussianThenDownsample(const Gray& src, double scale);
std::pair<Gray, Gray> forwardGradValid(const Gray& s);
void periodicGradient(const Gray& s, Gray& h, Gray& v);
Gray divergence(const Gray& h, const Gray& v);
std::vector<double> gradDenominator(int h, int w);
double otsuThresholdSquared(const Gray& s);
DarkProjection darkChannelArgmin(const Gray& s, int patch);
Gray projectDarkZeros(const Gray& s, const DarkProjection& dp, double threshold);

Gray l0Restore(const Gray& input, const Kernel& kernel, double lambda, double kappa, double betaMax, bool alreadyPadded = false);
Gray l0DarkRestore(const Gray& input, const Kernel& kernel, double lambdaDark, double lambdaGrad,
                   double kappa, double betaMax, int darkPatch);
std::tuple<Gray, Gray, double> thresholdGradients(const Gray& latent, int psfSize, double threshold, bool estimate);
Kernel estimatePsf(const Gray& bx, const Gray& by, const Gray& lx, const Gray& ly,
                   double weight, int kh, int kw, const Options& opts);
void pruneKernel(Kernel& k);
void adjustKernelCenter(Kernel& k);
Kernel resizeKernel(const Kernel& src, int size);
Kernel initKernel(int n);
std::pair<Kernel, Gray> blindKernel(const Gray& input, Options opts);

Gray tvDeconvolve(const Gray& b, const Kernel& k, double lambda);
Image bilateralDiff(const Image& src, double sigmaSpace, double sigmaRange);

} // namespace deblur::impl
