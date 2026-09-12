#include "orb.hpp"
#include "fast_detector.hpp"
#include "image_utils.hpp"
#include "orb_pattern.hpp" // Contains bit_pattern_31_
#include <chrono>
#include <cmath>
#include <algorithm>
#include <random> // Required for initializeOrbPattern

namespace {
// Accumulates elapsed time into `totalMs` for the duration of its scope.
// Used to time interleaved sub-steps inside a per-pyramid-level loop, where
// a single before/after timer around the whole loop can't separate them.
class ScopedAccumulator {
public:
    ScopedAccumulator(double* totalMs) : totalMs_(totalMs), start_(std::chrono::steady_clock::now()) {}
    ~ScopedAccumulator() {
        if (totalMs_) {
            *totalMs_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count();
        }
    }
private:
    double* totalMs_;
    std::chrono::steady_clock::time_point start_;
};
} // namespace

static void initializeOrbPattern(const Point2i* pattern0, std::vector<Point2i>& pattern, int ntuples, int tupleSize, int poolSize) {
    std::mt19937 rng(0x12345678); // Fixed seed to ensure consistent descriptors
    std::uniform_int_distribution<int> dist(0, poolSize - 1);
    
    int i, k, k1;
    pattern.resize(ntuples * tupleSize);

    for (i = 0; i < ntuples; i++) {
        for (k = 0; k < tupleSize; k++) {
            for (;;) {
                int idx = dist(rng);
                Point2i pt = pattern0[idx];
                for (k1 = 0; k1 < k; k1++) {
                    if (pattern[tupleSize * i + k1] == pt)
                        break;
                }
                if (k1 == k) {
                    pattern[tupleSize * i + k] = pt;
                    break;
                }
            }
        }
    }
}

ORB::ORB(int nfeatures, float scaleFactor, int nlevels, int edgeThreshold, 
         int firstLevel, int wta_k, ScoreType scoreType, int patchSize, int fastThreshold)
    : nfeatures(nfeatures), scaleFactor(scaleFactor), nlevels(nlevels),
      edgeThreshold(edgeThreshold), firstLevel(firstLevel), wta_k(wta_k),
      scoreType(scoreType), patchSize(patchSize), fastThreshold(fastThreshold) 
{}

float ORB::getScale(int level) const {
    return std::pow(scaleFactor, static_cast<float>(level - firstLevel));
}

void ORB::detectAndCompute(const Image8U& image, std::vector<KeyPoint>& keypoints, Image8U& descriptors,
                            std::vector<StageTiming>* stageTimings) {
    if (image.cols == 0 || image.rows == 0) return;

    keypoints.clear();

    // 1. Build the Image Pyramid
    double pyramidMs = 0.0;
    std::vector<Image8U> imagePyramid(nlevels);
    std::vector<float> layerScale(nlevels);
    std::vector<Rect> layerInfo(nlevels); // Used to simulate OpenCV's offset structure if needed by core functions

    {
        ScopedAccumulator timer(&pyramidMs);
        imagePyramid[0] = image; // Base level
        layerScale[0] = 1.0f;
        layerInfo[0] = {0, 0, image.cols, image.rows};

        for (int level = 1; level < nlevels; ++level) {
            float scale = getScale(level);
            layerScale[level] = scale;
            float inv_scale = 1.0f / scale;

            int scaled_width = static_cast<int>(std::round(image.cols * inv_scale));
            int scaled_height = static_cast<int>(std::round(image.rows * inv_scale));

            resizeBilinear(imagePyramid[level - 1], imagePyramid[level], scaled_width, scaled_height);
            layerInfo[level] = {0, 0, scaled_width, scaled_height};
        }
    }

    // 2. Detect Keypoints across all pyramid levels
    double fastDetectionMs = 0.0;
    double harrisScoringMs = 0.0;
    std::vector<KeyPoint> allKeypoints;
    float factor = 1.0f / scaleFactor;
    float desiredFeaturesPerScale = nfeatures * (1.0f - factor) / (1.0f - std::pow(factor, static_cast<float>(nlevels)));
    int sumFeatures = 0;

    for (int level = 0; level < nlevels; ++level) {
        int featuresNum = (level == nlevels - 1) ? std::max(nfeatures - sumFeatures, 0) : static_cast<int>(std::round(desiredFeaturesPerScale));
        sumFeatures += featuresNum;
        desiredFeaturesPerScale *= factor;

        std::vector<KeyPoint> levelKeypoints;
        std::vector<KeyPoint> filteredKeypoints;
        {
            ScopedAccumulator timer(&fastDetectionMs);
            detectFAST(imagePyramid[level], levelKeypoints, fastThreshold, true);

            // Filter by image border
            int border = edgeThreshold;
            for (const auto& kp : levelKeypoints) {
                if (kp.pt.x >= border && kp.pt.x < imagePyramid[level].cols - border &&
                    kp.pt.y >= border && kp.pt.y < imagePyramid[level].rows - border) {

                    KeyPoint newKp = kp;
                    newKp.octave = level;
                    newKp.size = patchSize * layerScale[level];
                    filteredKeypoints.push_back(newKp);
                }
            }
        }

        // Apply Harris scoring to get the best 'featuresNum' points
        if (scoreType == HARRIS_SCORE && !filteredKeypoints.empty()) {
            ScopedAccumulator timer(&harrisScoringMs);
            HarrisResponses(imagePyramid[level], layerInfo, filteredKeypoints, 7, 0.04f);

            // Sort descending by response
            std::sort(filteredKeypoints.begin(), filteredKeypoints.end(),
                      [](const KeyPoint& a, const KeyPoint& b) { return a.response > b.response; });

            // Retain top features
            if (filteredKeypoints.size() > static_cast<size_t>(featuresNum)) {
                filteredKeypoints.resize(featuresNum);
            }
        }

        allKeypoints.insert(allKeypoints.end(), filteredKeypoints.begin(), filteredKeypoints.end());
    }

    if (stageTimings) {
        stageTimings->push_back({"pyramid", pyramidMs});
        stageTimings->push_back({"fast_detection", fastDetectionMs});
        stageTimings->push_back({"harris_scoring", harrisScoringMs});
    }

    if (allKeypoints.empty()) return;

    // 3. Compute Angles (ICAngles)
    double orientationMs = 0.0;
    int halfPatchSize = patchSize / 2;
    std::vector<int> umax(halfPatchSize + 2);
    {
        ScopedAccumulator timer(&orientationMs);
        int vmax = static_cast<int>(std::floor(halfPatchSize * std::sqrt(2.0f) / 2.0f + 1));
        int vmin = static_cast<int>(std::ceil(halfPatchSize * std::sqrt(2.0f) / 2.0f));

        for (int v = 0; v <= vmax; ++v) {
            umax[v] = static_cast<int>(std::round(std::sqrt(halfPatchSize * halfPatchSize - v * v)));
        }
        for (int v = halfPatchSize, v0 = 0; v >= vmin; --v) {
            while (umax[v0] == umax[v0 + 1]) ++v0;
            umax[v] = v0;
            ++v0;
        }

        // We process angles level by level to pass the correct image
        for (int level = 0; level < nlevels; ++level) {
            std::vector<KeyPoint> levelKpts;
            std::vector<int> originalIndices;

            for (size_t i = 0; i < allKeypoints.size(); ++i) {
                if (allKeypoints[i].octave == level) {
                    levelKpts.push_back(allKeypoints[i]);
                    originalIndices.push_back(i);
                }
            }

            if (!levelKpts.empty()) {
                ICAngles(imagePyramid[level], layerInfo, levelKpts, umax, halfPatchSize);
                for (size_t i = 0; i < levelKpts.size(); ++i) {
                    allKeypoints[originalIndices[i]].angle = levelKpts[i].angle;
                }
            }
        }
    }
    if (stageTimings) stageTimings->push_back({"orientation", orientationMs});

    // 4. Prepare pattern from bit_pattern_31_
    double patternInitMs = 0.0;
    std::vector<Point2i> pattern;
    const int npoints = 512;
    std::vector<Point2i> patternbuf(npoints);
    const Point2i* pattern0 = reinterpret_cast<const Point2i*>(bit_pattern_31_);

    {
        ScopedAccumulator timer(&patternInitMs);
        if (wta_k == 2) {
            pattern.assign(pattern0, pattern0 + npoints);
        } else if (wta_k == 3 || wta_k == 4) {
            // dsize is 32 bytes, which means 256 bits (or tuples).
            // 256 tuples * wta_k points per tuple.
            int ntuples = 32 * 8;
            initializeOrbPattern(pattern0, pattern, ntuples, wta_k, npoints);
        }
    }
    if (stageTimings) stageTimings->push_back({"pattern_init", patternInitMs});

    if (wta_k != 2 && wta_k != 3 && wta_k != 4) {
        // Fallback for unsupported wta_k
        return;
    }

    // 5. Compute Descriptors
    double blurMs = 0.0;
    double descriptorsMs = 0.0;
    int dsize = 32; // 32 bytes = 256 bits
    descriptors = Image8U(dsize, allKeypoints.size());

    for (int level = 0; level < nlevels; ++level) {
        {
            // Apply Gaussian Blur before computing descriptors for stability
            ScopedAccumulator timer(&blurMs);
            gaussianBlur7x7(imagePyramid[level], imagePyramid[level]);
        }

        std::vector<KeyPoint> levelKpts;
        std::vector<int> originalIndices;
        
        for (size_t i = 0; i < allKeypoints.size(); ++i) {
            if (allKeypoints[i].octave == level) {
                levelKpts.push_back(allKeypoints[i]);
                originalIndices.push_back(i);
            }
        }

        if (!levelKpts.empty()) {
            ScopedAccumulator timer(&descriptorsMs);
            Image8U levelDescriptors(dsize, levelKpts.size());
            computeOrbDescriptors(imagePyramid[level], layerInfo, layerScale, levelKpts, levelDescriptors, pattern, dsize, wta_k);

            // Copy back to main descriptor matrix
            for (size_t i = 0; i < levelKpts.size(); ++i) {
                std::copy(levelDescriptors.ptr(i), levelDescriptors.ptr(i) + dsize, descriptors.ptr(originalIndices[i]));
            }
        }
    }
    if (stageTimings) {
        stageTimings->push_back({"gaussian_blur", blurMs});
        stageTimings->push_back({"descriptors", descriptorsMs});
    }

    // Finally, scale keypoints back to the original image resolution
    double finalizeMs = 0.0;
    {
        ScopedAccumulator timer(&finalizeMs);
        for (auto& kp : allKeypoints) {
            kp.pt *= layerScale[kp.octave];
        }
    }
    if (stageTimings) stageTimings->push_back({"finalize", finalizeMs});

    keypoints = allKeypoints;
}