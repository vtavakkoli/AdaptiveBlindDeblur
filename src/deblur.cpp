#include "internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace deblur {

Kernel normalizeKernel(Kernel kernel) {
    for (double& v : kernel.values) if (!std::isfinite(v) || v < 0.0) v = 0.0;
    const double sum = std::accumulate(kernel.values.begin(), kernel.values.end(), 0.0);
    if (sum <= impl::kEps) {
        std::fill(kernel.values.begin(), kernel.values.end(), 0.0);
        if (!kernel.values.empty()) kernel.at(kernel.height / 2, kernel.width / 2) = 1.0;
        return kernel;
    }
    for (double& v : kernel.values) v /= sum;
    return kernel;
}

Result deblur(const Image& blurred, const Options& options) {
    if (blurred.empty()) throw std::invalid_argument("deblur: empty image");
    if (options.kernelSize < 3 || options.kernelSize % 2 == 0)
        throw std::invalid_argument("kernelSize must be odd and >= 3");
    if (options.darkPatchSize < 1 || options.darkPatchSize % 2 == 0)
        throw std::invalid_argument("darkPatchSize must be positive and odd");

    Image grayImage = toGrayscale(blurred);
    auto [kernel, interim] = impl::blindKernel(impl::imageChannel(grayImage, 0), options);
    Image latent = deconvolveNonBlind(blurred, kernel, options);
    return {std::move(kernel), impl::grayToImage(interim), std::move(latent)};
}

Image kernelToImage(const Kernel& kernel, bool normalizeForDisplay) {
    Image out(kernel.height, kernel.width, 1);
    if (kernel.values.empty()) return out;
    double lo = 0.0, hi = 1.0;
    if (normalizeForDisplay) {
        auto [mn, mx] = std::minmax_element(kernel.values.begin(), kernel.values.end());
        lo = *mn; hi = *mx;
    }
    const double d = std::max(hi - lo, impl::kEps);
    for (std::size_t i = 0; i < kernel.values.size(); ++i)
        out.pixels[i] = normalizeForDisplay ? (kernel.values[i] - lo) / d : kernel.values[i];
    return out;
}

double meanSquaredError(const Image& a, const Image& b) {
    if (a.width != b.width || a.height != b.height || a.channels != b.channels)
        throw std::invalid_argument("MSE image size mismatch");
    double sum = 0.0;
#pragma omp parallel for reduction(+:sum) if(a.pixels.size() > 50000)
    for (long long i = 0; i < static_cast<long long>(a.pixels.size()); ++i) {
        const double d = a.pixels[static_cast<std::size_t>(i)] - b.pixels[static_cast<std::size_t>(i)];
        sum += d * d;
    }
    return sum / std::max<std::size_t>(1, a.pixels.size());
}

double psnr(const Image& reference, const Image& candidate, double peak) {
    const double mse = meanSquaredError(reference, candidate);
    if (mse <= impl::kEps) return std::numeric_limits<double>::infinity();
    return 10.0 * std::log10(peak * peak / mse);
}

} // namespace deblur
