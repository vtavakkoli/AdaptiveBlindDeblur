#include "internal.hpp"

#include <algorithm>
#include <cmath>

namespace deblur::impl {
using detail::Spectrum;

Gray tvDeconvolve(const Gray& b, const Kernel& k, double lambda) {
    constexpr double betaMin = 0.001;
    double beta = 1.0 / std::max(lambda, 1e-12);
    Gray image = b, ix, iy;
    periodicGradient(image, ix, iy);
    auto otf = detail::psfToOtf(k.values, k.height, k.width, b.h, b.w);
    auto fb = detail::fft2(b.v, b.h, b.w);
    auto nomin = detail::spectralMultiply(fb, otf, false, true);
    auto den1 = detail::spectralPower(otf);
    auto den2 = gradDenominator(b.h, b.w);

    while (beta > betaMin) {
        const double gamma = 1.0 / (2.0 * beta);
        const double threshold = beta * lambda;
        Gray wx(ix.h, ix.w), wy(iy.h, iy.w);
#pragma omp parallel for if(ix.h * ix.w > 50000)
        for (long long i = 0; i < static_cast<long long>(ix.v.size()); ++i) {
            const auto z = static_cast<std::size_t>(i);
            auto shrink = [&](double v) {
                const double a = std::abs(v) - threshold;
                return a > 0 ? std::copysign(a, v) : 0.0;
            };
            wx.v[z] = shrink(ix.v[z]);
            wy.v[z] = shrink(iy.v[z]);
        }
        auto fd = detail::fft2(divergence(wx, wy).v, b.h, b.w);
        Spectrum num = nomin;
        std::vector<double> den(den1.size());
        for (std::size_t i = 0; i < den.size(); ++i) {
            num.data[i] += gamma * fd.data[i];
            den[i] = den1[i] + gamma * den2[i];
        }
        image.v = detail::ifft2(detail::spectralDivideReal(num, den));
        periodicGradient(image, ix, iy);
        beta /= 2.0;
    }
    return image;
}

Image bilateralDiff(const Image& src, double sigmaSpace, double sigmaRange) {
    const int r = static_cast<int>(std::ceil(3.0 * sigmaSpace));
    const double invS = 1.0 / (2 * sigmaSpace * sigmaSpace);
    const double invR = 1.0 / (2 * sigmaRange * sigmaRange);
    Image out(src.height, src.width, src.channels);
#pragma omp parallel for schedule(dynamic) if(src.height * src.width > 10000)
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            std::vector<double> sums(src.channels, 0.0);
            double weights = 0.0;
            for (int yy = -r; yy <= r; ++yy) {
                for (int xx = -r; xx <= r; ++xx) {
                    const int sy = std::clamp(y + yy, 0, src.height - 1);
                    const int sx = std::clamp(x + xx, 0, src.width - 1);
                    double feature = 0.0;
                    for (int c = 0; c < src.channels; ++c) {
                        const double d = src.at(y,x,c) - src.at(sy,sx,c);
                        feature += d * d;
                    }
                    const double w = std::exp(-(xx*xx + yy*yy) * invS - feature * invR);
                    weights += w;
                    for (int c = 0; c < src.channels; ++c) sums[c] += w * src.at(sy,sx,c);
                }
            }
            for (int c = 0; c < src.channels; ++c) out.at(y,x,c) = sums[c] / std::max(weights, kEps);
        }
    }
    return out;
}

} // namespace deblur::impl

namespace deblur {

Image convolve(const Image& image, const Kernel& kernel) {
    if (image.empty() || kernel.values.empty()) throw std::invalid_argument("convolve requires non-empty image/kernel");
    Image out(image.height, image.width, image.channels);
    const auto otf = detail::psfToOtf(kernel.values, kernel.height, kernel.width, image.height, image.width);
#pragma omp parallel for if(image.channels > 1)
    for (int c = 0; c < image.channels; ++c) {
        impl::Gray g = impl::imageChannel(image, c);
        auto f = detail::fft2(g.v, g.h, g.w);
        auto conv = detail::spectralMultiply(f, otf);
        impl::Gray r(g.h, g.w); r.v = detail::ifft2(conv);
        impl::setChannel(out, c, r);
    }
    return out;
}

Image deconvolveNonBlind(const Image& blurred, const Kernel& kernel, const Options& options) {
    if (blurred.empty()) throw std::invalid_argument("deconvolveNonBlind: empty image");
    Image tv(blurred.height, blurred.width, blurred.channels);
#pragma omp parallel for if(blurred.channels > 1)
    for (int c = 0; c < blurred.channels; ++c) {
        impl::Gray b = impl::imageChannel(blurred, c);
        impl::Gray p = impl::padForFft(b, kernel.height, kernel.width);
        impl::setChannel(tv, c, impl::crop(impl::tvDeconvolve(p, kernel, options.lambdaTv), b.h, b.w));
    }
    if (options.ringingWeight == 0.0) return clampImage(tv);

    Image l0(blurred.height, blurred.width, blurred.channels);
#pragma omp parallel for if(blurred.channels > 1)
    for (int c = 0; c < blurred.channels; ++c) {
        impl::Gray b = impl::imageChannel(blurred, c);
        impl::Gray p = impl::padForFft(b, kernel.height, kernel.width);
        impl::setChannel(l0, c, impl::crop(impl::l0Restore(p, kernel, options.lambdaL0,
                                                         options.l0Kappa, options.betaMax, true), b.h, b.w));
    }
    Image diff = tv;
    for (std::size_t i = 0; i < diff.pixels.size(); ++i) diff.pixels[i] -= l0.pixels[i];
    Image smooth = impl::bilateralDiff(diff, 3.0, 0.1);
    Image result = tv;
    for (std::size_t i = 0; i < result.pixels.size(); ++i) result.pixels[i] -= options.ringingWeight * smooth.pixels[i];
    return clampImage(result);
}

} // namespace deblur
