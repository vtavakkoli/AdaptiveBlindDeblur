#include "deblur/deblur.hpp"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void usage(const char* exe) {
    std::cout << "High-performance CVPR 2016 dark-channel blind deblurring\n\n"
              << "Usage:\n  " << exe << " <input> <output> [options]\n\n"
              << "Options:\n"
              << "  --kernel-size N       Odd blur-kernel size (default 25)\n"
              << "  --iterations N        Kernel/image alternations per scale (default 5)\n"
              << "  --gamma X             Input gamma correction (default 1.0)\n"
              << "  --lambda-dark X       Dark-channel L0 weight (default 0.004)\n"
              << "  --lambda-grad X       Gradient L0 weight (default 0.004)\n"
              << "  --lambda-tv X         Final TV weight (default 0.003)\n"
              << "  --lambda-l0 X         Final L0 weight (default 0.0005)\n"
              << "  --ringing-weight X    Ringing suppression strength (default 1.0)\n"
              << "  --kernel-out PATH     Save estimated kernel visualization\n"
              << "  --interim-out PATH    Save intermediate latent image\n"
              << "  --quiet                Suppress progress output\n"
              << "  --fast                 Faster preview: 2 iterations, betaMax=4096\n"
              << "  -h, --help             Show this help\n";
}

double parseDouble(const char* s, const char* name) {
    char* end = nullptr;
    const double v = std::strtod(s, &end);
    if (!end || *end != '\0') throw std::runtime_error(std::string("Invalid ") + name + ": " + s);
    return v;
}

int parseInt(const char* s, const char* name) {
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (!end || *end != '\0') throw std::runtime_error(std::string("Invalid ") + name + ": " + s);
    return static_cast<int>(v);
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 1 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
            usage(argv[0]);
            return 0;
        }
        if (argc < 3) {
            usage(argv[0]);
            return 2;
        }

        const std::string input = argv[1];
        const std::string output = argv[2];
        std::string kernelOut, interimOut;
        deblur::Options o;

        for (int i = 3; i < argc; ++i) {
            const std::string a = argv[i];
            auto next = [&](const char* name) {
                if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value for ") + name);
                return argv[++i];
            };
            if (a == "--kernel-size") o.kernelSize = parseInt(next("--kernel-size"), "kernel size");
            else if (a == "--iterations") o.iterationsPerScale = parseInt(next("--iterations"), "iterations");
            else if (a == "--gamma") o.gamma = parseDouble(next("--gamma"), "gamma");
            else if (a == "--lambda-dark") o.lambdaDark = parseDouble(next("--lambda-dark"), "lambda-dark");
            else if (a == "--lambda-grad") o.lambdaGrad = parseDouble(next("--lambda-grad"), "lambda-grad");
            else if (a == "--lambda-tv") o.lambdaTv = parseDouble(next("--lambda-tv"), "lambda-tv");
            else if (a == "--lambda-l0") o.lambdaL0 = parseDouble(next("--lambda-l0"), "lambda-l0");
            else if (a == "--ringing-weight") o.ringingWeight = parseDouble(next("--ringing-weight"), "ringing-weight");
            else if (a == "--kernel-out") kernelOut = next("--kernel-out");
            else if (a == "--interim-out") interimOut = next("--interim-out");
            else if (a == "--quiet") o.verbose = false;
            else if (a == "--fast") { o.iterationsPerScale = 2; o.betaMax = 4096; }
            else throw std::runtime_error("Unknown option: " + a);
        }

        auto img = deblur::loadImage(input);
        const auto t0 = std::chrono::steady_clock::now();
        auto result = deblur::deblur(img, o);
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        deblur::saveImage(output, result.latent);
        if (!kernelOut.empty()) deblur::saveImage(kernelOut, deblur::kernelToImage(result.kernel));
        if (!interimOut.empty()) deblur::saveImage(interimOut, deblur::clampImage(result.interim));

        std::cerr << std::fixed << std::setprecision(3)
                  << "done in " << elapsed << " s; kernel sum=1; output=" << output << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
