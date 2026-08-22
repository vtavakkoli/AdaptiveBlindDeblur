#include "fft.hpp"
#include "fftw_compat.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace deblur::detail {
namespace {

struct Plan {
    int h;
    int w;
    int cols;
    double* real = nullptr;
    fftw_complex* complex = nullptr;
    fftw_plan forward = nullptr;
    fftw_plan inverse = nullptr;
    std::mutex mutex;

    Plan(int height, int width) : h(height), w(width), cols(width / 2 + 1) {
        real = static_cast<double*>(fftw_malloc(sizeof(double) * static_cast<std::size_t>(h) * w));
        complex = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * static_cast<std::size_t>(h) * cols));
        if (!real || !complex) throw std::bad_alloc();
        forward = fftw_plan_dft_r2c_2d(h, w, real, complex, FFTW_ESTIMATE);
        inverse = fftw_plan_dft_c2r_2d(h, w, complex, real, FFTW_ESTIMATE);
        if (!forward || !inverse) throw std::runtime_error("FFTW plan creation failed");
    }

    ~Plan() {
        if (forward) fftw_destroy_plan(forward);
        if (inverse) fftw_destroy_plan(inverse);
        if (complex) fftw_free(complex);
        if (real) fftw_free(real);
    }
};

std::shared_ptr<Plan> getPlan(int h, int w) {
    static std::once_flag threadInit;
    std::call_once(threadInit, [] {
        if (fftw_init_threads()) {
            unsigned n = std::max(1u, std::thread::hardware_concurrency());
            if (const char* env = std::getenv("DEBLUR_FFT_THREADS")) {
                try { n = static_cast<unsigned>(std::max(1, std::stoi(env))); } catch (...) {}
            } else {
                n = std::min(n, 8u);
            }
            fftw_plan_with_nthreads(static_cast<int>(n));
        }
    });

    static std::mutex cacheMutex;
    static std::unordered_map<unsigned long long, std::shared_ptr<Plan>> cache;
    const auto key = (static_cast<unsigned long long>(static_cast<unsigned>(h)) << 32U) |
                     static_cast<unsigned>(w);
    std::lock_guard lock(cacheMutex);
    if (auto it = cache.find(key); it != cache.end()) return it->second;
    auto p = std::make_shared<Plan>(h, w);
    cache.emplace(key, p);
    return p;
}

void validateSame(const Spectrum& a, const Spectrum& b) {
    if (a.height != b.height || a.width != b.width) throw std::invalid_argument("Spectrum size mismatch");
}

bool isFast(int n) {
    if (n <= 0) return false;
    for (int p : {2, 3, 5, 7, 11, 13}) while (n % p == 0) n /= p;
    return n == 1;
}

} // namespace

Spectrum fft2(const std::vector<double>& input, int height, int width) {
    if (input.size() != static_cast<std::size_t>(height) * width) throw std::invalid_argument("fft2 input size mismatch");
    auto plan = getPlan(height, width);
    Spectrum out(height, width);
    std::lock_guard lock(plan->mutex);
    std::memcpy(plan->real, input.data(), sizeof(double) * input.size());
    fftw_execute(plan->forward);
    for (std::size_t i = 0; i < out.data.size(); ++i) {
        out.data[i] = {plan->complex[i][0], plan->complex[i][1]};
    }
    return out;
}

std::vector<double> ifft2(const Spectrum& input) {
    auto plan = getPlan(input.height, input.width);
    std::vector<double> out(static_cast<std::size_t>(input.height) * input.width);
    std::lock_guard lock(plan->mutex);
    for (std::size_t i = 0; i < input.data.size(); ++i) {
        plan->complex[i][0] = input.data[i].real();
        plan->complex[i][1] = input.data[i].imag();
    }
    fftw_execute(plan->inverse);
    const double scale = 1.0 / static_cast<double>(input.height * input.width);
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = plan->real[i] * scale;
    return out;
}

Spectrum psfToOtf(const std::vector<double>& psf, int ph, int pw, int h, int w) {
    if (psf.size() != static_cast<std::size_t>(ph) * pw || ph > h || pw > w) {
        throw std::invalid_argument("Invalid PSF/OTF dimensions");
    }
    std::vector<double> padded(static_cast<std::size_t>(h) * w, 0.0);
    const int cy = ph / 2;
    const int cx = pw / 2;
    for (int y = 0; y < ph; ++y) {
        const int yy = (y - cy + h) % h;
        for (int x = 0; x < pw; ++x) {
            const int xx = (x - cx + w) % w;
            padded[static_cast<std::size_t>(yy) * w + xx] = psf[static_cast<std::size_t>(y) * pw + x];
        }
    }
    return fft2(padded, h, w);
}

std::vector<double> otfToPsf(const Spectrum& otf, int ph, int pw) {
    auto spatial = ifft2(otf);
    std::vector<double> psf(static_cast<std::size_t>(ph) * pw);
    const int cy = ph / 2;
    const int cx = pw / 2;
    for (int y = 0; y < ph; ++y) {
        const int yy = (y - cy + otf.height) % otf.height;
        for (int x = 0; x < pw; ++x) {
            const int xx = (x - cx + otf.width) % otf.width;
            psf[static_cast<std::size_t>(y) * pw + x] = spatial[static_cast<std::size_t>(yy) * otf.width + xx];
        }
    }
    return psf;
}

std::vector<double> spectralPower(const Spectrum& s) {
    std::vector<double> out(s.data.size());
    for (std::size_t i = 0; i < s.data.size(); ++i) out[i] = std::norm(s.data[i]);
    return out;
}

Spectrum spectralMultiply(const Spectrum& a, const Spectrum& b, bool conjugateA, bool conjugateB) {
    validateSame(a, b);
    Spectrum out(a.height, a.width);
#pragma omp parallel for if(a.data.size() > 50000)
    for (long long i = 0; i < static_cast<long long>(a.data.size()); ++i) {
        auto av = conjugateA ? std::conj(a.data[static_cast<std::size_t>(i)]) : a.data[static_cast<std::size_t>(i)];
        auto bv = conjugateB ? std::conj(b.data[static_cast<std::size_t>(i)]) : b.data[static_cast<std::size_t>(i)];
        out.data[static_cast<std::size_t>(i)] = av * bv;
    }
    return out;
}

Spectrum spectralAdd(const Spectrum& a, const Spectrum& b) {
    validateSame(a, b);
    Spectrum out(a.height, a.width);
#pragma omp parallel for if(a.data.size() > 50000)
    for (long long i = 0; i < static_cast<long long>(a.data.size()); ++i) out.data[static_cast<std::size_t>(i)] = a.data[static_cast<std::size_t>(i)] + b.data[static_cast<std::size_t>(i)];
    return out;
}

Spectrum spectralScale(const Spectrum& a, double scale) {
    Spectrum out = a;
#pragma omp parallel for if(a.data.size() > 50000)
    for (long long i = 0; i < static_cast<long long>(out.data.size()); ++i) out.data[static_cast<std::size_t>(i)] *= scale;
    return out;
}

Spectrum spectralDivideReal(const Spectrum& a, const std::vector<double>& denominator, double epsilon) {
    if (a.data.size() != denominator.size()) throw std::invalid_argument("spectralDivideReal size mismatch");
    Spectrum out = a;
#pragma omp parallel for if(a.data.size() > 50000)
    for (long long i = 0; i < static_cast<long long>(out.data.size()); ++i) {
        const double d = std::max(std::abs(denominator[static_cast<std::size_t>(i)]), epsilon);
        out.data[static_cast<std::size_t>(i)] /= d;
    }
    return out;
}

int optimalFftSize(int n) {
    n = std::max(n, 1);
    while (!isFast(n)) ++n;
    return n;
}

} // namespace deblur::detail
