#include "deblur/deblur.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numeric>

namespace {
int failures = 0;
#define CHECK(expr) do { if (!(expr)) { std::cerr << "FAIL " << __FILE__ << ':' << __LINE__ << "  " #expr "\n"; ++failures; } } while (0)

bool finiteImage(const deblur::Image& i) {
    return std::all_of(i.pixels.begin(), i.pixels.end(), [](double v) { return std::isfinite(v); });
}

bool finiteKernel(const deblur::Kernel& k) {
    return std::all_of(k.values.begin(), k.values.end(), [](double v) { return std::isfinite(v) && v >= 0.0; });
}

deblur::Image synthetic(int h, int w) {
    deblur::Image x(h, w, 1);
    for (int y = 0; y < h; ++y) {
        for (int xx = 0; xx < w; ++xx) {
            double v = 0.12;
            if (xx > w / 5 && xx < 4 * w / 5 && y > h / 5 && y < 4 * h / 5) v = 0.82;
            if ((xx / 6 + y / 6) % 2 == 0) v += 0.08;
            x.at(y, xx) = std::clamp(v, 0.0, 1.0);
        }
    }
    return x;
}

deblur::Kernel motion5() {
    deblur::Kernel k(5, 5);
    k.at(2, 1) = 0.2;
    k.at(2, 2) = 0.6;
    k.at(2, 3) = 0.2;
    return deblur::normalizeKernel(std::move(k));
}

void testKernelNormalization() {
    deblur::Kernel k(3, 3);
    k.at(1, 1) = 2;
    k.at(1, 2) = 1;
    k.at(0, 0) = -3;
    k = deblur::normalizeKernel(k);
    CHECK(std::abs(std::accumulate(k.values.begin(), k.values.end(), 0.0) - 1.0) < 1e-12);
    CHECK(k.at(0, 0) == 0.0);
    CHECK(finiteKernel(k));
}

void testConvolutionImpulse() {
    deblur::Image x(17, 17, 1);
    x.at(8, 8) = 1.0;
    auto y = deblur::convolve(x, motion5());
    CHECK(std::abs(std::accumulate(y.pixels.begin(), y.pixels.end(), 0.0) - 1.0) < 1e-9);
    CHECK(y.at(8, 8) > 0.59);
    CHECK(y.at(8, 7) > 0.19);
    CHECK(y.at(8, 9) > 0.19);
}

void testImageIo() {
    auto x = synthetic(24, 28);
    auto path = std::filesystem::temp_directory_path() / "deblur_roundtrip.png";
    deblur::saveImage(path.string(), x);
    auto y = deblur::loadImage(path.string());
    CHECK(y.width == x.width && y.height == x.height && y.channels == 1);
    CHECK(deblur::meanSquaredError(x, y) < 2e-5);
    std::filesystem::remove(path);
}

void testNonBlindImproves() {
    auto sharp = synthetic(64, 64);
    auto k = motion5();
    auto blur = deblur::convolve(sharp, k);
    deblur::Options o;
    o.lambdaTv = 0.003;
    o.lambdaL0 = 0.0005;
    o.ringingWeight = 0.0;
    o.betaMax = 4096;
    o.verbose = false;
    auto restored = deblur::deconvolveNonBlind(blur, k, o);
    CHECK(finiteImage(restored));
    const double before = deblur::meanSquaredError(sharp, blur);
    const double after = deblur::meanSquaredError(sharp, restored);
    std::cerr << "nonblind mse before=" << before << " after=" << after << "\n";
    CHECK(after < before);
}

void testBlindSmoke() {
    auto sharp = synthetic(48, 48);
    auto blur = deblur::convolve(sharp, motion5());
    deblur::Options o;
    o.kernelSize = 5;
    o.iterationsPerScale = 1;
    o.lambdaDark = 0.0;
    o.lambdaGrad = 0.003;
    o.lambdaTv = 0.003;
    o.lambdaL0 = 0.0005;
    o.ringingWeight = 0.0;
    o.betaMax = 256;
    o.cgIterations = 8;
    o.verbose = false;
    auto r = deblur::deblur(blur, o);
    CHECK(r.kernel.width == 5 && r.kernel.height == 5);
    CHECK(finiteKernel(r.kernel));
    CHECK(std::abs(std::accumulate(r.kernel.values.begin(), r.kernel.values.end(), 0.0) - 1.0) < 1e-9);
    CHECK(r.latent.width == blur.width && r.latent.height == blur.height);
    CHECK(finiteImage(r.latent));
}
} // namespace

int main() {
    testKernelNormalization();
    testConvolutionImpulse();
    testImageIo();
    testNonBlindImproves();
    testBlindSmoke();
    if (failures) {
        std::cerr << failures << " test assertion(s) failed\n";
        return 1;
    }
    std::cout << "all tests passed\n";
    return 0;
}
