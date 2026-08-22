#pragma once

#include <complex>
#include <vector>

namespace deblur::detail {

struct Spectrum {
    int height = 0;
    int width = 0;
    int cols = 0;
    std::vector<std::complex<double>> data;

    Spectrum() = default;
    Spectrum(int h, int w)
        : height(h), width(w), cols(w / 2 + 1), data(static_cast<std::size_t>(h) * cols) {}

    std::complex<double>& at(int y, int x) { return data[static_cast<std::size_t>(y) * cols + x]; }
    const std::complex<double>& at(int y, int x) const { return data[static_cast<std::size_t>(y) * cols + x]; }
};

Spectrum fft2(const std::vector<double>& input, int height, int width);
std::vector<double> ifft2(const Spectrum& input);
Spectrum psfToOtf(const std::vector<double>& psf, int ph, int pw, int h, int w);
std::vector<double> otfToPsf(const Spectrum& otf, int ph, int pw);
std::vector<double> spectralPower(const Spectrum& s);
Spectrum spectralMultiply(const Spectrum& a, const Spectrum& b, bool conjugateA = false, bool conjugateB = false);
Spectrum spectralAdd(const Spectrum& a, const Spectrum& b);
Spectrum spectralScale(const Spectrum& a, double scale);
Spectrum spectralDivideReal(const Spectrum& a, const std::vector<double>& denominator, double epsilon = 1e-12);
int optimalFftSize(int n);

} // namespace deblur::detail
