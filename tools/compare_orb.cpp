// Standalone diagnostic tool: compares this repo's ORB against OpenCV's
// cv::ORB on keypoint count, spatial agreement, descriptor similarity, and
// timing. Not part of the default build (see BUILD_OPENCV_COMPARISON in
// CMakeLists.txt) and not linked into orb_lib/orb_test/orb_tests.
//
// Design: docs/design/compare-orb-design.md

#include "orb.hpp"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <array>
#include <bitset>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

// Framework-agnostic representation both implementations get converted into.
struct CommonKeypoint {
    float x, y;
    float angle;
    float response;
    int octave;
};

struct ImplementationResult {
    std::string name;
    std::vector<CommonKeypoint> keypoints;
    std::vector<std::array<uint8_t, 32>> descriptors; // index-aligned with keypoints
    double meanMs = 0.0;
    double stddevMs = 0.0;
};

// One shared parameter set used to construct both ORB instances, so any
// divergence in the results reflects algorithm behavior, not configuration.
struct OrbParams {
    int nfeatures = 500;
    float scaleFactor = 1.2f;
    int nlevels = 8;
    int edgeThreshold = 31;
    int firstLevel = 0;
    int wta_k = 2;
    int patchSize = 31;
    int fastThreshold = 20;
};

struct ComparisonReport {
    size_t countA = 0;
    size_t countB = 0;
    size_t spatialMatches = 0;
    double meanHamming = 0.0;
    double medianHamming = 0.0;
    double maxHamming = 0.0;
    double meanAngleDiffDeg = 0.0;
    // Breakdown by whether the matched pair shares the same pyramid octave:
    // orientation is computed on that octave's own (differently-scaled) image,
    // so an octave disagreement is expected to inflate the angle difference
    // even when both implementations found "the same" corner spatially.
    size_t octaveAgreeCount = 0;
    double meanAngleDiffOctaveAgreeDeg = 0.0;
    double meanAngleDiffOctaveDisagreeDeg = 0.0;
    double meanHammingOctaveAgree = 0.0;
    double meanHammingOctaveDisagree = 0.0;
};

// Smallest angular distance between two angles in degrees, in [0, 180].
// A plain |a - b| is wrong near the 0/360 wraparound (e.g. 359 vs 9 is 10
// degrees apart, not 350).
double circularAngleDiffDeg(float a, float b) {
    double diff = std::fabs(static_cast<double>(a) - static_cast<double>(b));
    return diff > 180.0 ? 360.0 - diff : diff;
}

// Summary of a response-score distribution across one implementation's full
// keypoint set. Harris/FAST response is scored on a different scale between
// the two implementations, so this is reported per-implementation rather
// than as a paired difference (unlike angle/Hamming above).
struct ResponseStats {
    double mean = 0.0;
    double median = 0.0;
    double min = 0.0;
    double max = 0.0;
};

ResponseStats computeResponseStats(const ImplementationResult& result) {
    ResponseStats stats;
    if (result.keypoints.empty()) return stats;

    std::vector<double> responses;
    responses.reserve(result.keypoints.size());
    for (const auto& kp : result.keypoints) responses.push_back(kp.response);

    stats.mean = std::accumulate(responses.begin(), responses.end(), 0.0) / responses.size();
    std::sort(responses.begin(), responses.end());
    stats.median = responses[responses.size() / 2];
    stats.min = responses.front();
    stats.max = responses.back();
    return stats;
}

// Times `fn` once (discarded, warm-up) then `repeats` more times, returning
// {mean, stddev} in milliseconds.
template <typename Fn>
std::pair<double, double> timeRepeats(Fn&& fn, int repeats) {
    fn();

    std::vector<double> samples(repeats);
    for (int i = 0; i < repeats; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        fn();
        auto t1 = std::chrono::steady_clock::now();
        samples[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    double variance = 0.0;
    for (double s : samples) variance += (s - mean) * (s - mean);
    variance /= samples.size();

    return {mean, std::sqrt(variance)};
}

int hammingDistance(const std::array<uint8_t, 32>& a, const std::array<uint8_t, 32>& b) {
    int dist = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dist += std::bitset<8>(static_cast<uint8_t>(a[i] ^ b[i])).count();
    }
    return dist;
}

ImplementationResult runMine(const Image8U& img, const OrbParams& p, int repeats) {
    ORB orb(p.nfeatures, p.scaleFactor, p.nlevels, p.edgeThreshold, p.firstLevel,
            p.wta_k, ORB::HARRIS_SCORE, p.patchSize, p.fastThreshold);

    std::vector<KeyPoint> keypoints;
    Image8U descriptors;

    auto timing = timeRepeats([&]() { orb.detectAndCompute(img, keypoints, descriptors); }, repeats);

    ImplementationResult result;
    result.name = "mine";
    result.meanMs = timing.first;
    result.stddevMs = timing.second;

    result.keypoints.reserve(keypoints.size());
    for (const auto& kp : keypoints) {
        result.keypoints.push_back({kp.pt.x, kp.pt.y, kp.angle, kp.response, kp.octave});
    }

    result.descriptors.resize(keypoints.size());
    for (size_t i = 0; i < keypoints.size(); ++i) {
        const uint8_t* row = descriptors.ptr(static_cast<int>(i));
        std::copy(row, row + 32, result.descriptors[i].begin());
    }

    return result;
}

ImplementationResult runOpenCv(const cv::Mat& img, const OrbParams& p, int repeats) {
    cv::Ptr<cv::ORB> orb = cv::ORB::create(p.nfeatures, p.scaleFactor, p.nlevels, p.edgeThreshold,
                                            p.firstLevel, p.wta_k, cv::ORB::HARRIS_SCORE,
                                            p.patchSize, p.fastThreshold);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    auto timing = timeRepeats(
        [&]() { orb->detectAndCompute(img, cv::noArray(), keypoints, descriptors); }, repeats);

    ImplementationResult result;
    result.name = "opencv";
    result.meanMs = timing.first;
    result.stddevMs = timing.second;

    result.keypoints.reserve(keypoints.size());
    for (const auto& kp : keypoints) {
        result.keypoints.push_back({kp.pt.x, kp.pt.y, kp.angle, kp.response, kp.octave});
    }

    result.descriptors.resize(keypoints.size());
    for (size_t i = 0; i < keypoints.size(); ++i) {
        const uint8_t* row = descriptors.ptr(static_cast<int>(i));
        std::copy(row, row + 32, result.descriptors[i].begin());
    }

    return result;
}

// Greedy nearest-neighbor spatial matching (not a mutual/one-to-one
// assignment) — adequate for a diagnostic; see docs/design/compare-orb-design.md.
ComparisonReport compareImplementations(const ImplementationResult& a, const ImplementationResult& b,
                                         float spatialRadiusPx = 3.0f) {
    ComparisonReport report;
    report.countA = a.keypoints.size();
    report.countB = b.keypoints.size();

    std::vector<double> hammingDistances;
    std::vector<double> hammingOctaveAgree;
    std::vector<double> hammingOctaveDisagree;
    std::vector<double> angleDiffs;
    std::vector<double> angleDiffsOctaveAgree;
    std::vector<double> angleDiffsOctaveDisagree;

    for (size_t i = 0; i < a.keypoints.size(); ++i) {
        const auto& kpA = a.keypoints[i];
        int bestJ = -1;
        float bestDist = spatialRadiusPx;

        for (size_t j = 0; j < b.keypoints.size(); ++j) {
            const auto& kpB = b.keypoints[j];
            float dx = kpA.x - kpB.x;
            float dy = kpA.y - kpB.y;
            float dist = std::sqrt(dx * dx + dy * dy);
            if (dist <= bestDist) {
                bestDist = dist;
                bestJ = static_cast<int>(j);
            }
        }

        if (bestJ >= 0) {
            const auto& kpB = b.keypoints[static_cast<size_t>(bestJ)];
            ++report.spatialMatches;
            double hamming = hammingDistance(a.descriptors[i], b.descriptors[static_cast<size_t>(bestJ)]);
            hammingDistances.push_back(hamming);

            double angleDiff = circularAngleDiffDeg(kpA.angle, kpB.angle);
            angleDiffs.push_back(angleDiff);
            if (kpA.octave == kpB.octave) {
                ++report.octaveAgreeCount;
                angleDiffsOctaveAgree.push_back(angleDiff);
                hammingOctaveAgree.push_back(hamming);
            } else {
                angleDiffsOctaveDisagree.push_back(angleDiff);
                hammingOctaveDisagree.push_back(hamming);
            }
        }
    }

    if (!hammingDistances.empty()) {
        report.meanHamming = std::accumulate(hammingDistances.begin(), hammingDistances.end(), 0.0) /
                              hammingDistances.size();
        std::vector<double> sorted = hammingDistances;
        std::sort(sorted.begin(), sorted.end());
        report.medianHamming = sorted[sorted.size() / 2];
        report.maxHamming = *std::max_element(hammingDistances.begin(), hammingDistances.end());
        report.meanAngleDiffDeg = std::accumulate(angleDiffs.begin(), angleDiffs.end(), 0.0) / angleDiffs.size();
    }
    if (!angleDiffsOctaveAgree.empty()) {
        report.meanAngleDiffOctaveAgreeDeg =
            std::accumulate(angleDiffsOctaveAgree.begin(), angleDiffsOctaveAgree.end(), 0.0) /
            angleDiffsOctaveAgree.size();
    }
    if (!angleDiffsOctaveDisagree.empty()) {
        report.meanAngleDiffOctaveDisagreeDeg =
            std::accumulate(angleDiffsOctaveDisagree.begin(), angleDiffsOctaveDisagree.end(), 0.0) /
            angleDiffsOctaveDisagree.size();
    }
    if (!hammingOctaveAgree.empty()) {
        report.meanHammingOctaveAgree =
            std::accumulate(hammingOctaveAgree.begin(), hammingOctaveAgree.end(), 0.0) / hammingOctaveAgree.size();
    }
    if (!hammingOctaveDisagree.empty()) {
        report.meanHammingOctaveDisagree = std::accumulate(hammingOctaveDisagree.begin(),
                                                             hammingOctaveDisagree.end(), 0.0) /
                                            hammingOctaveDisagree.size();
    }

    return report;
}

} // namespace

int main(int argc, char** argv) {
    std::string imagePath = argc > 1 ? argv[1] : "pinball.jpg";
    int repeats = argc > 2 ? std::atoi(argv[2]) : 10;

    int width, height, channels;
    unsigned char* imgData = stbi_load(imagePath.c_str(), &width, &height, &channels, 1);
    if (!imgData) {
        std::cerr << "Error loading image: " << imagePath << std::endl;
        return 1;
    }

    Image8U grayImage(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            grayImage.at(y, x) = imgData[y * width + x];
        }
    }
    stbi_image_free(imgData);

    // Both implementations run on the same pixel buffer, so any divergence
    // reflects the ORB algorithm, not different image decoders.
    cv::Mat cvImage(grayImage.rows, grayImage.cols, CV_8UC1, grayImage.data.data(), grayImage.step);
    cv::Mat cvImageOwned = cvImage.clone();

    OrbParams params;

    std::cout << "Image: " << imagePath << " (" << width << "x" << height << ")\n";
    std::cout << "Params: nfeatures=" << params.nfeatures << " scaleFactor=" << params.scaleFactor
              << " nlevels=" << params.nlevels << " edgeThreshold=" << params.edgeThreshold
              << " wta_k=" << params.wta_k << " patchSize=" << params.patchSize
              << " fastThreshold=" << params.fastThreshold << "\n\n";

    ImplementationResult mine = runMine(grayImage, params, repeats);
    ImplementationResult opencv = runOpenCv(cvImageOwned, params, repeats);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "mine:   " << mine.keypoints.size() << " keypoints, mean " << mine.meanMs
              << " ms (stddev " << mine.stddevMs << " ms, " << repeats << " runs)\n";
    std::cout << "opencv: " << opencv.keypoints.size() << " keypoints, mean " << opencv.meanMs
              << " ms (stddev " << opencv.stddevMs << " ms, " << repeats << " runs)\n\n";

    // Response is scored on a different scale by each implementation, so it's
    // reported as a per-implementation distribution rather than a paired diff.
    ResponseStats mineResponse = computeResponseStats(mine);
    ResponseStats opencvResponse = computeResponseStats(opencv);
    std::cout << std::scientific << std::setprecision(3);
    std::cout << "Response distribution (mine):   mean " << mineResponse.mean << ", median "
              << mineResponse.median << ", min " << mineResponse.min << ", max " << mineResponse.max << "\n";
    std::cout << "Response distribution (opencv): mean " << opencvResponse.mean << ", median "
              << opencvResponse.median << ", min " << opencvResponse.min << ", max " << opencvResponse.max
              << "\n\n";
    std::cout << std::fixed << std::setprecision(2);

    ComparisonReport report = compareImplementations(mine, opencv);
    double matchPct = mine.keypoints.empty()
                           ? 0.0
                           : 100.0 * static_cast<double>(report.spatialMatches) / mine.keypoints.size();

    std::cout << "Spatial match (mine -> opencv, radius <= 3px): " << report.spatialMatches << "/"
              << mine.keypoints.size() << " (" << matchPct << "%)\n";
    std::cout << "Descriptor Hamming distance (matched pairs, out of 256 bits): mean "
              << report.meanHamming << ", median " << report.medianHamming << ", max "
              << report.maxHamming << "\n";
    std::cout << "Angle difference (matched pairs, circular): mean " << report.meanAngleDiffDeg << " deg\n";
    std::cout << "  same octave (" << report.octaveAgreeCount << "/" << report.spatialMatches
              << "): mean angle diff " << report.meanAngleDiffOctaveAgreeDeg << " deg, mean Hamming "
              << report.meanHammingOctaveAgree << "\n";
    std::cout << "  different octave (" << (report.spatialMatches - report.octaveAgreeCount) << "/"
              << report.spatialMatches << "): mean angle diff " << report.meanAngleDiffOctaveDisagreeDeg
              << " deg, mean Hamming " << report.meanHammingOctaveDisagree << "\n";

    return 0;
}
