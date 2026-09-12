// Standalone diagnostic tool: reports where ORB::detectAndCompute spends its
// time, stage by stage (pyramid construction, FAST detection, Harris
// scoring, orientation, pattern init, Gaussian blur, descriptor computation).
// Zero external dependencies (unlike tools/compare_orb.cpp, this needs no
// OpenCV) - built by default alongside orb_test.

#include "orb.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::string imagePath = argc > 1 ? argv[1] : "pinball.jpg";
    int repeats = argc > 2 ? std::atoi(argv[2]) : 10;

    int width, height, channels;
    unsigned char* imgData = stbi_load(imagePath.c_str(), &width, &height, &channels, 1);
    if (!imgData) {
        std::cerr << "Error loading image: " << imagePath << std::endl;
        return 1;
    }

    Image8U image(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            image.at(y, x) = imgData[y * width + x];
        }
    }
    stbi_image_free(imgData);

    std::cout << "Image: " << imagePath << " (" << width << "x" << height << ")\n";

    // Same defaults as main.cpp / compare_orb.cpp, for consistency across tools.
    ORB orb(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);

    // Warm-up (discarded), then timed repeats, accumulating per-stage totals
    // by name so results don't depend on push_back order.
    std::vector<KeyPoint> keypoints;
    Image8U descriptors;
    std::vector<StageTiming> warmup;
    orb.detectAndCompute(image, keypoints, descriptors, &warmup);

    std::vector<std::string> stageNames;
    std::vector<double> stageTotalMs;
    double totalMs = 0.0;

    for (int i = 0; i < repeats; ++i) {
        std::vector<StageTiming> stageTimings;
        auto t0 = std::chrono::steady_clock::now();
        orb.detectAndCompute(image, keypoints, descriptors, &stageTimings);
        auto t1 = std::chrono::steady_clock::now();
        totalMs += std::chrono::duration<double, std::milli>(t1 - t0).count();

        for (const auto& stage : stageTimings) {
            auto it = std::find(stageNames.begin(), stageNames.end(), stage.name);
            if (it == stageNames.end()) {
                stageNames.push_back(stage.name);
                stageTotalMs.push_back(stage.ms);
            } else {
                stageTotalMs[static_cast<size_t>(it - stageNames.begin())] += stage.ms;
            }
        }
    }

    std::cout << "Keypoints: " << keypoints.size() << "\n";
    std::cout << "Total (measured end-to-end): " << std::fixed << std::setprecision(2)
              << (totalMs / repeats) << " ms/run (" << repeats << " runs)\n\n";

    double stageSumMs = std::accumulate(stageTotalMs.begin(), stageTotalMs.end(), 0.0);
    std::cout << "Stage breakdown (mean per run, " << repeats << " runs):\n";
    for (size_t i = 0; i < stageNames.size(); ++i) {
        double meanMs = stageTotalMs[i] / repeats;
        double pct = stageSumMs > 0.0 ? 100.0 * stageTotalMs[i] / stageSumMs : 0.0;
        std::cout << "  " << std::left << std::setw(16) << stageNames[i] << std::right
                  << std::setw(10) << meanMs << " ms  (" << std::setw(5) << pct << "%)\n";
    }

    return 0;
}
