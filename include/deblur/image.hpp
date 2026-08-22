#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace deblur {

struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<double> pixels;

    Image() = default;
    Image(int h, int w, int c, double value = 0.0)
        : width(w), height(h), channels(c), pixels(static_cast<std::size_t>(h) * w * c, value) {}

    [[nodiscard]] bool empty() const noexcept { return width <= 0 || height <= 0 || channels <= 0 || pixels.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return pixels.size(); }

    double& at(int y, int x, int c = 0) {
        return pixels[(static_cast<std::size_t>(y) * width + x) * channels + c];
    }
    const double& at(int y, int x, int c = 0) const {
        return pixels[(static_cast<std::size_t>(y) * width + x) * channels + c];
    }
};

Image loadImage(const std::string& path);
void saveImage(const std::string& path, const Image& image, int jpegQuality = 95);
Image toGrayscale(const Image& image);
Image clampImage(const Image& image, double lo = 0.0, double hi = 1.0);

} // namespace deblur
