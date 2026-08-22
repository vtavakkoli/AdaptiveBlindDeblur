#include "internal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>

namespace deblur::impl {
using detail::Spectrum;

Gray l0Restore(const Gray& input, const Kernel& kernel, double lambda, double kappa,
               double betaMax, bool alreadyPadded) {
    const int oh = input.h, ow = input.w;
    Gray s = alreadyPadded ? input : padForFft(input, kernel.height, kernel.width);
    auto ker = detail::psfToOtf(kernel.values, kernel.height, kernel.width, s.h, s.w);
    auto fs0 = detail::fft2(s.v, s.h, s.w);
    auto norm1 = detail::spectralMultiply(fs0, ker, false, true);
    auto denKer = detail::spectralPower(ker);
    auto denGrad = gradDenominator(s.h, s.w);

    double beta = std::max(2.0 * lambda, 1e-8);
    while (beta < betaMax) {
        Gray gx, gy;
        periodicGradient(s, gx, gy);
        const double cutoff = lambda / beta;
#pragma omp parallel for if(s.h * s.w > 50000)
        for (long long i = 0; i < static_cast<long long>(gx.v.size()); ++i) {
            const auto k = static_cast<std::size_t>(i);
            if (gx.v[k] * gx.v[k] + gy.v[k] * gy.v[k] < cutoff) gx.v[k] = gy.v[k] = 0.0;
        }
        auto div = divergence(gx, gy);
        auto fdiv = detail::fft2(div.v, s.h, s.w);
        Spectrum num = norm1;
        std::vector<double> den(denKer.size());
#pragma omp parallel for if(den.size() > 50000)
        for (long long i = 0; i < static_cast<long long>(den.size()); ++i) {
            const auto k = static_cast<std::size_t>(i);
            num.data[k] += beta * fdiv.data[k];
            den[k] = denKer[k] + beta * denGrad[k];
        }
        s.v = detail::ifft2(detail::spectralDivideReal(num, den));
        beta *= kappa;
        if (lambda == 0.0) break;
    }
    return alreadyPadded ? s : crop(s, oh, ow);
}

Gray l0DarkRestore(const Gray& input, const Kernel& kernel, double lambdaDark, double lambdaGrad,
                   double kappa, double betaMax, int darkPatch) {
    Gray s = input;
    auto ker = detail::psfToOtf(kernel.values, kernel.height, kernel.width, s.h, s.w);
    auto fs0 = detail::fft2(s.v, s.h, s.w);
    auto norm1 = detail::spectralMultiply(fs0, ker, false, true);
    auto denKer = detail::spectralPower(ker);
    auto denGrad = gradDenominator(s.h, s.w);

    double pixelBeta = lambdaDark / otsuThresholdSquared(s);
    while (pixelBeta < 8.0) {
        auto dp = darkChannelArgmin(s, darkPatch);
        Gray u = projectDarkZeros(s, dp, lambdaDark / std::max(pixelBeta, kEps));
        double beta = std::max(2.0 * lambdaGrad, 1e-8);
        while (beta < betaMax) {
            Gray gx, gy;
            periodicGradient(s, gx, gy);
            const double cutoff = lambdaGrad / beta;
#pragma omp parallel for if(s.h * s.w > 50000)
            for (long long i = 0; i < static_cast<long long>(gx.v.size()); ++i) {
                const auto k = static_cast<std::size_t>(i);
                if (gx.v[k] * gx.v[k] + gy.v[k] * gy.v[k] < cutoff) gx.v[k] = gy.v[k] = 0.0;
            }
            auto fdiv = detail::fft2(divergence(gx, gy).v, s.h, s.w);
            auto fu = detail::fft2(u.v, s.h, s.w);
            Spectrum num = norm1;
            std::vector<double> den(denKer.size());
#pragma omp parallel for if(den.size() > 50000)
            for (long long i = 0; i < static_cast<long long>(den.size()); ++i) {
                const auto k = static_cast<std::size_t>(i);
                num.data[k] += beta * fdiv.data[k] + pixelBeta * fu.data[k];
                den[k] = denKer[k] + beta * denGrad[k] + pixelBeta;
            }
            s.v = detail::ifft2(detail::spectralDivideReal(num, den));
            beta *= kappa;
            if (lambdaGrad == 0.0) break;
        }
        pixelBeta *= kappa;
    }
    return s;
}

std::tuple<Gray, Gray, double> thresholdGradients(const Gray& latent, int psfSize,
                                                   double threshold, bool estimate) {
    auto [gx, gy] = forwardGradValid(latent);
    const double pi = std::acos(-1.0);
    if (estimate) {
        std::array<std::vector<double>, 4> buckets;
        for (std::size_t i = 0; i < gx.v.size(); ++i) {
            const double x = gx.v[i], y = gy.v[i], pm = x * x + y * y;
            const double safeX = std::abs(x) < 1e-20 ? std::copysign(1e-20, x == 0.0 ? 1.0 : x) : x;
            const double angle = std::atan(y / safeX);
            int b = -1;
            if (angle >= 0 && angle < pi / 4) b = 0;
            else if (angle >= pi / 4 && angle <= pi / 2) b = 1;
            else if (angle >= -pi / 4 && angle < 0) b = 2;
            else if (angle >= -pi / 2 && angle < -pi / 4) b = 3;
            if (b >= 0) buckets[static_cast<std::size_t>(b)].push_back(pm);
        }
        const std::size_t need = static_cast<std::size_t>(std::max(psfSize * 20, 10));
        double minKth = std::numeric_limits<double>::infinity();
        for (auto& b : buckets) {
            if (b.size() < need) { minKth = 0.0; break; }
            auto nth = b.end() - static_cast<std::ptrdiff_t>(need);
            std::nth_element(b.begin(), nth, b.end());
            minKth = std::min(minKth, *nth);
        }
        if (!std::isfinite(minKth) || minKth <= 0.0) {
            std::vector<double> pm(gx.v.size());
            for (std::size_t i = 0; i < pm.size(); ++i) pm[i] = gx.v[i] * gx.v[i] + gy.v[i] * gy.v[i];
            const std::size_t idx = pm.size() * 9 / 10;
            std::nth_element(pm.begin(), pm.begin() + static_cast<std::ptrdiff_t>(idx), pm.end());
            minKth = pm[idx];
        }
        constexpr double step = 0.00006;
        threshold = std::max(step, std::floor(minKth / step) * step);
    }

    auto countKept = [&](double th) {
        std::size_t kept = 0;
        for (std::size_t i = 0; i < gx.v.size(); ++i)
            if (gx.v[i] * gx.v[i] + gy.v[i] * gy.v[i] >= th) ++kept;
        return kept;
    };
    while (countKept(threshold) == 0 && threshold > 1e-12) threshold *= 0.81;
    for (std::size_t i = 0; i < gx.v.size(); ++i) {
        if (gx.v[i] * gx.v[i] + gy.v[i] * gy.v[i] < threshold) gx.v[i] = gy.v[i] = 0.0;
    }
    if (!estimate) threshold /= 1.1;
    return {std::move(gx), std::move(gy), threshold};
}

Kernel estimatePsf(const Gray& bx, const Gray& by, const Gray& lx, const Gray& ly,
                   double weight, int kh, int kw, const Options& opts) {
    auto lxf = detail::fft2(lx.v, lx.h, lx.w);
    auto lyf = detail::fft2(ly.v, ly.h, ly.w);
    auto bxf = detail::fft2(bx.v, bx.h, bx.w);
    auto byf = detail::fft2(by.v, by.h, by.w);
    auto bspec = detail::spectralAdd(detail::spectralMultiply(lxf, bxf, true, false),
                                     detail::spectralMultiply(lyf, byf, true, false));
    std::vector<double> m(lxf.data.size());
    for (std::size_t i = 0; i < m.size(); ++i) m[i] = std::norm(lxf.data[i]) + std::norm(lyf.data[i]);

    Kernel x(kh, kw, 1.0 / static_cast<double>(kh * kw));
    auto applyA = [&](const Kernel& in) {
        auto xf = detail::psfToOtf(in.values, kh, kw, lx.h, lx.w);
        for (std::size_t i = 0; i < xf.data.size(); ++i) xf.data[i] *= m[i];
        Kernel y(kh, kw);
        y.values = detail::otfToPsf(xf, kh, kw);
        for (std::size_t i = 0; i < y.values.size(); ++i) y.values[i] += weight * in.values[i];
        return y;
    };

    Kernel b(kh, kw);
    b.values = detail::otfToPsf(bspec, kh, kw);
    Kernel ax = applyA(x), r(kh, kw), p(kh, kw);
    for (std::size_t i = 0; i < r.values.size(); ++i) r.values[i] = p.values[i] = b.values[i] - ax.values[i];
    auto dot = [](const Kernel& a, const Kernel& c) {
        double s = 0.0;
        for (std::size_t i = 0; i < a.values.size(); ++i) s += a.values[i] * c.values[i];
        return s;
    };
    double rsold = dot(r, r);
    for (int it = 0; it < opts.cgIterations && std::sqrt(rsold) >= opts.cgTolerance; ++it) {
        Kernel ap = applyA(p);
        const double denom = dot(p, ap);
        if (std::abs(denom) < kEps) break;
        const double alpha = rsold / denom;
        for (std::size_t i = 0; i < x.values.size(); ++i) {
            x.values[i] += alpha * p.values[i];
            r.values[i] -= alpha * ap.values[i];
        }
        const double rsnew = dot(r, r);
        if (std::sqrt(rsnew) < opts.cgTolerance) break;
        const double beta = rsnew / std::max(rsold, kEps);
        for (std::size_t i = 0; i < p.values.size(); ++i) p.values[i] = r.values[i] + beta * p.values[i];
        rsold = rsnew;
    }
    const double maxv = *std::max_element(x.values.begin(), x.values.end());
    for (double& v : x.values) if (v < maxv * 0.05) v = 0.0;
    return normalizeKernel(std::move(x));
}

void pruneKernel(Kernel& k) {
    const int h = k.height, w = k.width;
    std::vector<unsigned char> seen(k.values.size(), 0);
    constexpr int dy[8] = {-1,-1,-1,0,0,1,1,1};
    constexpr int dx[8] = {-1,0,1,-1,1,-1,0,1};
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const std::size_t start = static_cast<std::size_t>(y) * w + x;
            if (seen[start] || k.values[start] <= 0.0) continue;
            std::queue<std::pair<int,int>> q;
            std::vector<std::size_t> comp;
            q.push({y,x}); seen[start] = 1;
            double sum = 0.0;
            while (!q.empty()) {
                auto [cy,cx] = q.front(); q.pop();
                const std::size_t id = static_cast<std::size_t>(cy) * w + cx;
                comp.push_back(id); sum += k.values[id];
                for (int d = 0; d < 8; ++d) {
                    const int ny = cy + dy[d], nx = cx + dx[d];
                    if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
                    const auto ni = static_cast<std::size_t>(ny) * w + nx;
                    if (!seen[ni] && k.values[ni] > 0) { seen[ni] = 1; q.push({ny,nx}); }
                }
            }
            if (sum < 0.1) for (auto id : comp) k.values[id] = 0.0;
        }
    }
    k = normalizeKernel(std::move(k));
}

void adjustKernelCenter(Kernel& k) {
    const double sum = std::accumulate(k.values.begin(), k.values.end(), 0.0);
    if (sum <= kEps) return;
    double cx = 0.0, cy = 0.0;
    for (int y = 0; y < k.height; ++y)
        for (int x = 0; x < k.width; ++x) { const double v = k.at(y,x); cx += v*x; cy += v*y; }
    cx /= sum; cy /= sum;
    const int shiftX = static_cast<int>(std::llround(k.width / 2 - cx));
    const int shiftY = static_cast<int>(std::llround(k.height / 2 - cy));
    Kernel out(k.height, k.width);
    for (int y = 0; y < k.height; ++y) {
        for (int x = 0; x < k.width; ++x) {
            const int ny = y + shiftY, nx = x + shiftX;
            if (ny >= 0 && ny < k.height && nx >= 0 && nx < k.width) out.at(ny,nx) = k.at(y,x);
        }
    }
    k = normalizeKernel(std::move(out));
}

Kernel resizeKernel(const Kernel& src, int size) {
    Gray g(src.height, src.width); g.v = src.values;
    Gray r = resizeBilinear(g, size, size);
    Kernel out(size, size); out.values = std::move(r.v);
    for (double& v : out.values) v = std::max(0.0, v);
    return normalizeKernel(std::move(out));
}

Kernel initKernel(int n) {
    Kernel k(n, n);
    const int r = std::max(0, n / 2 - 1), c = std::max(0, n / 2 - 1);
    k.at(r, c) = 0.5;
    k.at(r, std::min(c + 1, n - 1)) += 0.5;
    return k;
}

std::pair<Kernel, Gray> blindKernel(const Gray& input, Options opts) {
    Gray y = input;
    if (std::abs(opts.gamma - 1.0) > 1e-12)
        for (double& v : y.v) v = std::pow(std::max(v, 0.0), opts.gamma);

    const double ret = std::sqrt(0.5);
    const int maxitr = std::max(static_cast<int>(std::floor(std::log(5.0 / opts.kernelSize) / std::log(ret))), 0);
    const int scales = maxitr + 1;
    std::vector<int> sizes(scales);
    for (int i = 0; i < scales; ++i) {
        int n = static_cast<int>(std::ceil(opts.kernelSize * std::pow(ret, i)));
        if (n % 2 == 0) ++n;
        sizes[i] = n;
    }

    Kernel ks;
    Gray interim;
    double threshold = 0.0;
    bool thresholdKnown = false;
    for (int s = scales - 1; s >= 0; --s) {
        ks = s == scales - 1 ? initKernel(sizes[s]) : resizeKernel(ks, sizes[s]);
        Gray ys = gaussianThenDownsample(y, std::pow(ret, s));
        if (opts.verbose)
            std::cerr << "scale " << (scales - s) << '/' << scales << " kernel=" << ks.width << 'x' << ks.height
                      << " image=" << ys.w << 'x' << ys.h << '\n';

        auto pad = padForFft(ys, ks.height, ks.width);
        auto [bx, by] = forwardGradValid(crop(pad, ys.h, ys.w));
        if (!thresholdKnown) {
            auto tmp = thresholdGradients(ys, ks.width, 0.0, true);
            threshold = std::get<2>(tmp);
            thresholdKnown = true;
        }
        for (int iter = 0; iter < opts.iterationsPerScale; ++iter) {
            Gray latent;
            if (opts.lambdaDark != 0.0) {
                latent = crop(l0DarkRestore(pad, ks, opts.lambdaDark, opts.lambdaGrad,
                                            opts.l0Kappa, opts.betaMax, opts.darkPatchSize), ys.h, ys.w);
            } else {
                latent = l0Restore(ys, ks, opts.lambdaGrad, opts.l0Kappa, opts.betaMax, false);
            }
            auto [lx, ly, newThreshold] = thresholdGradients(latent, ks.width, threshold, false);
            threshold = newThreshold;
            ks = estimatePsf(bx, by, lx, ly, 2.0, ks.height, ks.width, opts);
            pruneKernel(ks);
            interim = latent;
            if (opts.lambdaDark != 0.0) opts.lambdaDark = std::max(opts.lambdaDark / 1.1, 1e-4);
            if (opts.lambdaGrad != 0.0) opts.lambdaGrad = std::max(opts.lambdaGrad / 1.1, 1e-4);
            if (opts.verbose)
                std::cerr << "  iteration " << (iter + 1) << '/' << opts.iterationsPerScale
                          << " threshold=" << threshold << '\n';
        }
        adjustKernelCenter(ks);
    }
    if (opts.kernelThresholdDivisor > 0) {
        const double mx = *std::max_element(ks.values.begin(), ks.values.end());
        for (double& v : ks.values) if (v < mx / opts.kernelThresholdDivisor) v = 0.0;
        ks = normalizeKernel(std::move(ks));
    }
    return {std::move(ks), std::move(interim)};
}

} // namespace deblur::impl
