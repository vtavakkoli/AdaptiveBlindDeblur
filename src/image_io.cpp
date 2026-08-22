#include "deblur/image.hpp"

#include <png.h>
#include <jpeglib.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace deblur {
namespace {

std::string lowerExt(const std::string& path) {
    auto ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

unsigned char toByte(double v) {
    v = std::clamp(v, 0.0, 1.0);
    return static_cast<unsigned char>(v * 255.0 + 0.5);
}

Image loadPng(const std::string& path) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("Cannot open PNG: " + path);
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop info = png_create_info_struct(png);
    if (!png || !info) { std::fclose(fp); throw std::runtime_error("PNG initialization failed"); }
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        std::fclose(fp);
        throw std::runtime_error("PNG decode failed: " + path);
    }
    png_init_io(png, fp);
    png_read_info(png, info);
    int w = static_cast<int>(png_get_image_width(png, info));
    int h = static_cast<int>(png_get_image_height(png, info));
    int color = png_get_color_type(png, info);
    int depth = png_get_bit_depth(png, info);
    if (depth == 16) png_set_strip_16(png);
    if (color == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color == PNG_COLOR_TYPE_GRAY && depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (color & PNG_COLOR_MASK_ALPHA) png_set_strip_alpha(png);
    png_read_update_info(png, info);
    const int channels = png_get_channels(png, info);
    const std::size_t rowBytes = png_get_rowbytes(png, info);
    std::vector<unsigned char> raw(static_cast<std::size_t>(h) * rowBytes);
    std::vector<png_bytep> rows(h);
    for (int y = 0; y < h; ++y) rows[y] = raw.data() + static_cast<std::size_t>(y) * rowBytes;
    png_read_image(png, rows.data());
    png_destroy_read_struct(&png, &info, nullptr);
    std::fclose(fp);
    Image img(h, w, channels);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < channels; ++c) img.at(y, x, c) = rows[y][x * channels + c] / 255.0;
        }
    }
    return img;
}

void savePng(const std::string& path, const Image& image) {
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) throw std::runtime_error("Cannot write PNG: " + path);
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop info = png_create_info_struct(png);
    if (!png || !info) { std::fclose(fp); throw std::runtime_error("PNG initialization failed"); }
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        std::fclose(fp);
        throw std::runtime_error("PNG encode failed: " + path);
    }
    png_init_io(png, fp);
    const int channels = image.channels == 1 ? 1 : 3;
    const int color = channels == 1 ? PNG_COLOR_TYPE_GRAY : PNG_COLOR_TYPE_RGB;
    png_set_IHDR(png, info, image.width, image.height, 8, color, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);
    std::vector<unsigned char> row(static_cast<std::size_t>(image.width) * channels);
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            for (int c = 0; c < channels; ++c) {
                const int src = image.channels == 1 ? 0 : std::min(c, image.channels - 1);
                row[static_cast<std::size_t>(x) * channels + c] = toByte(image.at(y, x, src));
            }
        }
        png_write_row(png, row.data());
    }
    png_write_end(png, nullptr);
    png_destroy_write_struct(&png, &info);
    std::fclose(fp);
}

Image loadJpeg(const std::string& path) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("Cannot open JPEG: " + path);
    jpeg_decompress_struct cinfo{};
    jpeg_error_mgr jerr{};
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, fp);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);
    const int w = static_cast<int>(cinfo.output_width);
    const int h = static_cast<int>(cinfo.output_height);
    const int ch = static_cast<int>(cinfo.output_components);
    Image image(h, w, ch);
    std::vector<unsigned char> row(static_cast<std::size_t>(w) * ch);
    while (cinfo.output_scanline < cinfo.output_height) {
        JSAMPROW ptr = row.data();
        const int y = static_cast<int>(cinfo.output_scanline);
        jpeg_read_scanlines(&cinfo, &ptr, 1);
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < ch; ++c) image.at(y, x, c) = row[static_cast<std::size_t>(x) * ch + c] / 255.0;
        }
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    std::fclose(fp);
    return image;
}

void saveJpeg(const std::string& path, const Image& image, int quality) {
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) throw std::runtime_error("Cannot write JPEG: " + path);
    jpeg_compress_struct cinfo{};
    jpeg_error_mgr jerr{};
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, fp);
    cinfo.image_width = image.width;
    cinfo.image_height = image.height;
    cinfo.input_components = image.channels == 1 ? 1 : 3;
    cinfo.in_color_space = cinfo.input_components == 1 ? JCS_GRAYSCALE : JCS_RGB;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, std::clamp(quality, 1, 100), TRUE);
    jpeg_start_compress(&cinfo, TRUE);
    std::vector<unsigned char> row(static_cast<std::size_t>(image.width) * cinfo.input_components);
    while (cinfo.next_scanline < cinfo.image_height) {
        const int y = static_cast<int>(cinfo.next_scanline);
        for (int x = 0; x < image.width; ++x) {
            for (int c = 0; c < cinfo.input_components; ++c) {
                const int src = image.channels == 1 ? 0 : std::min(c, image.channels - 1);
                row[static_cast<std::size_t>(x) * cinfo.input_components + c] = toByte(image.at(y, x, src));
            }
        }
        JSAMPROW ptr = row.data();
        jpeg_write_scanlines(&cinfo, &ptr, 1);
    }
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    std::fclose(fp);
}

Image loadPnm(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open PNM: " + path);
    std::string magic;
    f >> magic;
    if (magic != "P5" && magic != "P6") throw std::runtime_error("Only binary P5/P6 PNM is supported");
    auto skipComments = [&] {
        while (f >> std::ws && f.peek() == '#') f.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    };
    skipComments();
    int w, h, maxv;
    f >> w;
    skipComments();
    f >> h;
    skipComments();
    f >> maxv;
    f.get();
    if (maxv != 255) throw std::runtime_error("PNM max value must be 255");
    const int ch = magic == "P5" ? 1 : 3;
    std::vector<unsigned char> raw(static_cast<std::size_t>(w) * h * ch);
    f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
    Image img(h, w, ch);
    for (std::size_t i = 0; i < raw.size(); ++i) img.pixels[i] = raw[i] / 255.0;
    return img;
}

void savePnm(const std::string& path, const Image& image) {
    const int ch = image.channels == 1 ? 1 : 3;
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write PNM: " + path);
    f << (ch == 1 ? "P5\n" : "P6\n") << image.width << ' ' << image.height << "\n255\n";
    std::vector<unsigned char> raw(static_cast<std::size_t>(image.width) * image.height * ch);
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            for (int c = 0; c < ch; ++c) {
                const int src = image.channels == 1 ? 0 : std::min(c, image.channels - 1);
                raw[(static_cast<std::size_t>(y) * image.width + x) * ch + c] = toByte(image.at(y, x, src));
            }
        }
    }
    f.write(reinterpret_cast<const char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
}

} // namespace

Image loadImage(const std::string& path) {
    const auto ext = lowerExt(path);
    if (ext == ".png") return loadPng(path);
    if (ext == ".jpg" || ext == ".jpeg") return loadJpeg(path);
    if (ext == ".pgm" || ext == ".ppm" || ext == ".pnm") return loadPnm(path);
    throw std::runtime_error("Unsupported image format: " + ext + " (use PNG, JPEG, PGM or PPM)");
}

void saveImage(const std::string& path, const Image& image, int jpegQuality) {
    if (image.empty()) throw std::invalid_argument("Cannot save empty image");
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    const auto ext = lowerExt(path);
    if (ext == ".png") return savePng(path, image);
    if (ext == ".jpg" || ext == ".jpeg") return saveJpeg(path, image, jpegQuality);
    if (ext == ".pgm" || ext == ".ppm" || ext == ".pnm") return savePnm(path, image);
    throw std::runtime_error("Unsupported image format: " + ext + " (use PNG, JPEG, PGM or PPM)");
}

Image toGrayscale(const Image& image) {
    if (image.channels == 1) return image;
    Image gray(image.height, image.width, 1);
#pragma omp parallel for if(image.height * image.width > 50000)
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            if (image.channels >= 3) gray.at(y, x) = 0.2989 * image.at(y, x, 0) + 0.5870 * image.at(y, x, 1) + 0.1140 * image.at(y, x, 2);
            else gray.at(y, x) = image.at(y, x, 0);
        }
    }
    return gray;
}

Image clampImage(const Image& image, double lo, double hi) {
    Image out = image;
#pragma omp parallel for if(out.pixels.size() > 50000)
    for (long long i = 0; i < static_cast<long long>(out.pixels.size()); ++i) {
        out.pixels[static_cast<std::size_t>(i)] = std::clamp(out.pixels[static_cast<std::size_t>(i)], lo, hi);
    }
    return out;
}

} // namespace deblur
