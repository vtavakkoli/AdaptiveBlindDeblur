#include "internal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <limits>

namespace deblur::impl {

Gray imageChannel(const Image& img, int c) {
    Gray g(img.height, img.width);
    for (int y = 0; y < img.height; ++y)
        for (int x = 0; x < img.width; ++x) g.at(y, x) = img.at(y, x, c);
    return g;
}

Image grayToImage(const Gray& g) {
    Image out(g.h, g.w, 1);
    out.pixels = g.v;
    return out;
}

void setChannel(Image& img, int c, const Gray& g) {
    for (int y = 0; y < img.height; ++y)
        for (int x = 0; x < img.width; ++x) img.at(y, x, c) = g.at(y, x);
}

int reflect101(int i, int n) {
    if (n <= 1) return 0;
    while (i < 0 || i >= n) {
        if (i < 0) i = -i;
        if (i >= n) i = 2 * n - 2 - i;
    }
    return i;
}

Gray padForFft(const Gray& src, int kh, int kw) {
    const int h = detail::optimalFftSize(src.h + kh - 1);
    const int w = detail::optimalFftSize(src.w + kw - 1);
    Gray out(h, w);
#pragma omp parallel for if(h * w > 50000)
    for (int y = 0; y < h; ++y) {
        const int sy = reflect101(y, src.h);
        for (int x = 0; x < w; ++x) out.at(y, x) = src.at(sy, reflect101(x, src.w));
    }
    return out;
}

Gray crop(const Gray& src, int h, int w) {
    Gray out(h, w);
    for (int y = 0; y < h; ++y) {
        std::copy_n(src.v.begin() + static_cast<std::size_t>(y) * src.w,
                    w, out.v.begin() + static_cast<std::size_t>(y) * w);
    }
    return out;
}

Gray resizeBilinear(const Gray& src, int nh, int nw) {
    if (nh == src.h && nw == src.w) return src;
    Gray out(nh, nw);
    const double sy = nh > 1 ? static_cast<double>(src.h - 1) / (nh - 1) : 0.0;
    const double sx = nw > 1 ? static_cast<double>(src.w - 1) / (nw - 1) : 0.0;
#pragma omp parallel for if(nh * nw > 50000)
    for (int y = 0; y < nh; ++y) {
        const double fy = y * sy;
        const int y0 = static_cast<int>(fy), y1 = std::min(y0 + 1, src.h - 1);
        const double ay = fy - y0;
        for (int x = 0; x < nw; ++x) {
            const double fx = x * sx;
            const int x0 = static_cast<int>(fx), x1 = std::min(x0 + 1, src.w - 1);
            const double ax = fx - x0;
            out.at(y, x) = (1 - ay) * ((1 - ax) * src.at(y0, x0) + ax * src.at(y0, x1))
                         + ay * ((1 - ax) * src.at(y1, x0) + ax * src.at(y1, x1));
        }
    }
    return out;
}

Gray gaussianThenDownsample(const Gray& src, double scale) {
    if (std::abs(scale - 1.0) < 1e-12) return src;
    const double sig = scale / std::acos(-1.0);
    std::vector<double> full(101), csum(101);
    double sum = 0.0;
    for (int i = -50; i <= 50; ++i) {
        const double g0 = i * 2.0 * std::acos(-1.0);
        full[static_cast<std::size_t>(i + 50)] = std::exp(-0.5 * g0 * g0 * sig * sig);
        sum += full[static_cast<std::size_t>(i + 50)];
    }
    for (double& v : full) v /= sum;
    double running = 0.0;
    for (int i = 0; i < 101; ++i) {
        running += full[static_cast<std::size_t>(i)];
        csum[static_cast<std::size_t>(i)] = running;
    }
    int lo = 0;
    while (lo < 50 && std::min(csum[lo], csum[100 - lo]) <= 0.05) ++lo;
    const int hi = 100 - lo;
    std::vector<double> k(full.begin() + lo, full.begin() + hi + 1);
    const int r = static_cast<int>(k.size()) / 2;
    Gray tmp(src.h, src.w), filtered(src.h, src.w);
#pragma omp parallel for if(src.h * src.w > 50000)
    for (int y = 0; y < src.h; ++y) {
        for (int x = 0; x < src.w; ++x) {
            double s = 0.0;
            for (int j = -r; j <= r; ++j) s += k[static_cast<std::size_t>(j + r)] * src.at(y, reflect101(x + j, src.w));
            tmp.at(y, x) = s;
        }
    }
#pragma omp parallel for if(src.h * src.w > 50000)
    for (int y = 0; y < src.h; ++y) {
        for (int x = 0; x < src.w; ++x) {
            double s = 0.0;
            for (int j = -r; j <= r; ++j) s += k[static_cast<std::size_t>(j + r)] * tmp.at(reflect101(y + j, src.h), x);
            filtered.at(y, x) = s;
        }
    }
    const int nh = std::max(2, static_cast<int>(std::floor((src.h - 1) * scale)) + 1);
    const int nw = std::max(2, static_cast<int>(std::floor((src.w - 1) * scale)) + 1);
    return resizeBilinear(filtered, nh, nw);
}

std::pair<Gray, Gray> forwardGradValid(const Gray& s) {
    if (s.h < 2 || s.w < 2) throw std::invalid_argument("Image too small for gradient");
    Gray gx(s.h - 1, s.w - 1), gy(s.h - 1, s.w - 1);
#pragma omp parallel for if(gx.h * gx.w > 50000)
    for (int y = 0; y < gx.h; ++y) {
        for (int x = 0; x < gx.w; ++x) {
            gx.at(y, x) = s.at(y, x + 1) - s.at(y, x);
            gy.at(y, x) = s.at(y + 1, x) - s.at(y, x);
        }
    }
    return {std::move(gx), std::move(gy)};
}

void periodicGradient(const Gray& s, Gray& h, Gray& v) {
    h = Gray(s.h, s.w);
    v = Gray(s.h, s.w);
#pragma omp parallel for if(s.h * s.w > 50000)
    for (int y = 0; y < s.h; ++y) {
        for (int x = 0; x < s.w; ++x) {
            h.at(y, x) = s.at(y, (x + 1) % s.w) - s.at(y, x);
            v.at(y, x) = s.at((y + 1) % s.h, x) - s.at(y, x);
        }
    }
}

Gray divergence(const Gray& h, const Gray& v) {
    Gray d(h.h, h.w);
#pragma omp parallel for if(h.h * h.w > 50000)
    for (int y = 0; y < h.h; ++y) {
        for (int x = 0; x < h.w; ++x) {
            d.at(y, x) = h.at(y, (x - 1 + h.w) % h.w) - h.at(y, x)
                       + v.at((y - 1 + h.h) % h.h, x) - v.at(y, x);
        }
    }
    return d;
}

std::vector<double> gradDenominator(int h, int w) {
    const std::vector<double> f{1.0, -1.0};
    auto ox = detail::psfToOtf(f, 1, 2, h, w);
    auto oy = detail::psfToOtf(f, 2, 1, h, w);
    auto px = detail::spectralPower(ox);
    auto py = detail::spectralPower(oy);
    for (std::size_t i = 0; i < px.size(); ++i) px[i] += py[i];
    return px;
}

double otsuThresholdSquared(const Gray& s) {
    std::array<double, 256> hist{};
    for (double v : s.v) {
        const double q = std::clamp(v * v, 0.0, 1.0);
        ++hist[static_cast<std::size_t>(std::min(255, static_cast<int>(q * 255.0)))];
    }
    const double total = static_cast<double>(s.v.size());
    double sumAll = 0.0;
    for (int i = 0; i < 256; ++i) sumAll += i * hist[static_cast<std::size_t>(i)];
    double sumB = 0.0, wB = 0.0, best = -1.0;
    int bestT = 0;
    for (int t = 0; t < 256; ++t) {
        wB += hist[static_cast<std::size_t>(t)];
        if (wB <= 0.0) continue;
        const double wF = total - wB;
        if (wF <= 0.0) break;
        sumB += t * hist[static_cast<std::size_t>(t)];
        const double mB = sumB / wB, mF = (sumAll - sumB) / wF;
        const double between = wB * wF * (mB - mF) * (mB - mF);
        if (between > best) { best = between; bestT = t; }
    }
    return std::max(bestT / 255.0, 1e-6);
}

DarkProjection darkChannelArgmin(const Gray& s, int patch) {
    if (patch % 2 == 0 || patch < 1) throw std::invalid_argument("darkPatchSize must be positive and odd");
    const int r = patch / 2, hp = s.h + 2 * r, wp = s.w + 2 * r;
    Gray padded(hp, wp);
    for (int y = 0; y < hp; ++y)
        for (int x = 0; x < wp; ++x)
            padded.at(y, x) = s.at(std::clamp(y - r, 0, s.h - 1), std::clamp(x - r, 0, s.w - 1));

    Gray hmin(hp, s.w);
    std::vector<int> hsrc(static_cast<std::size_t>(hp) * s.w);
#pragma omp parallel for if(hp * s.w > 50000)
    for (int y = 0; y < hp; ++y) {
        std::deque<int> dq;
        for (int x = 0; x < wp; ++x) {
            while (!dq.empty() && padded.at(y, dq.back()) >= padded.at(y, x)) dq.pop_back();
            dq.push_back(x);
            while (!dq.empty() && dq.front() <= x - patch) dq.pop_front();
            if (x >= patch - 1) {
                const int ox = x - patch + 1;
                if (ox < s.w) {
                    hmin.at(y, ox) = padded.at(y, dq.front());
                    hsrc[static_cast<std::size_t>(y) * s.w + ox] = dq.front();
                }
            }
        }
    }

    DarkProjection out{Gray(s.h, s.w), std::vector<int>(static_cast<std::size_t>(s.h) * s.w), wp, r};
#pragma omp parallel for if(s.h * s.w > 50000)
    for (int x = 0; x < s.w; ++x) {
        std::deque<int> dq;
        for (int y = 0; y < hp; ++y) {
            while (!dq.empty() && hmin.at(dq.back(), x) >= hmin.at(y, x)) dq.pop_back();
            dq.push_back(y);
            while (!dq.empty() && dq.front() <= y - patch) dq.pop_front();
            if (y >= patch - 1) {
                const int oy = y - patch + 1;
                if (oy < s.h) {
                    const int sy = dq.front();
                    const int sx = hsrc[static_cast<std::size_t>(sy) * s.w + x];
                    out.dark.at(oy, x) = hmin.at(sy, x);
                    out.argmin[static_cast<std::size_t>(oy) * s.w + x] = sy * wp + sx;
                }
            }
        }
    }
    return out;
}

Gray projectDarkZeros(const Gray& s, const DarkProjection& dp, double threshold) {
    Gray out = s;
    const int r = dp.radius;
    for (int y = 0; y < s.h; ++y) {
        for (int x = 0; x < s.w; ++x) {
            const double d = dp.dark.at(y, x);
            if (d * d >= threshold) continue;
            const int idx = dp.argmin[static_cast<std::size_t>(y) * s.w + x];
            const int py = idx / dp.paddedWidth, px = idx % dp.paddedWidth;
            const int sy = py - r, sx = px - r;
            if (sy >= r && sy < s.h - r && sx >= r && sx < s.w - r) out.at(sy, sx) = 0.0;
        }
    }
    return out;
}

} // namespace deblur::impl
