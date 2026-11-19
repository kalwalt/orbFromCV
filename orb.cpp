#include "orb.hpp"
#include "fast_detector.hpp"
#include "image_utils.hpp"
#include "orb_pattern.hpp" // Contains bit_pattern_31_
#include <chrono>
#include <cmath>
#include <stdexcept>
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

// Harris block size used for scoring, as in OpenCV's ORB.
static const int HARRIS_BLOCK_SIZE = 7;

// How far the (unrotated or rotated) BRIEF pattern can reach from a keypoint:
// ceil of the largest point norm in bit_pattern_31_ (19 px). The table is fixed
// at 31 px, so this does not shrink with a smaller patchSize.
static int patternReach() {
    static const int reach = [] {
        const Point2i* p = reinterpret_cast<const Point2i*>(bit_pattern_31_);
        float maxNorm2 = 0.f;
        for (int i = 0; i < 512; ++i) {
            float n2 = static_cast<float>(p[i].x * p[i].x + p[i].y * p[i].y);
            maxNorm2 = std::max(maxNorm2, n2);
        }
        return static_cast<int>(std::ceil(std::sqrt(maxNorm2)));
    }();
    return reach;
}

// KeyPointsFilter::retainBest, as in OpenCV: keep the n best by response.
// Points tied with the n-th response are kept too, so the result can exceed n
// (FAST scores are small integers, so ties at the cut are common).
static void retainBest(std::vector<KeyPoint>& keypoints, int n) {
    if (n < 0 || keypoints.size() <= static_cast<size_t>(n)) return;
    if (n == 0) { keypoints.clear(); return; }
    auto greater = [](const KeyPoint& a, const KeyPoint& b) { return a.response > b.response; };
    std::nth_element(keypoints.begin(), keypoints.begin() + n - 1, keypoints.end(), greater);
    const float ambiguous = keypoints[n - 1].response;
    auto newEnd = std::partition(keypoints.begin() + n, keypoints.end(),
                                 [ambiguous](const KeyPoint& k) { return k.response >= ambiguous; });
    keypoints.resize(static_cast<size_t>(newEnd - keypoints.begin()));
}

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
{
    // Reject unsupported parameters here, before any detection work, instead
    // of discovering them after the pyramid, FAST and Harris have run (#22,
    // #23). OpenCV asserts wta_k the same way in detectAndCompute.
    if (nfeatures < 0) throw std::invalid_argument("ORB: nfeatures must be >= 0");
    if (!(scaleFactor > 1.0f)) throw std::invalid_argument("ORB: scaleFactor must be > 1");
    if (nlevels < 1) throw std::invalid_argument("ORB: nlevels must be >= 1");
    if (edgeThreshold < 0) throw std::invalid_argument("ORB: edgeThreshold must be >= 0");
    // Level 0 is always the input image at scale 1 and the other levels are
    // scaled by scaleFactor^level; a firstLevel above 0 would duplicate level 0
    // and shift every size by one step, so it is not supported (#23).
    if (firstLevel != 0) throw std::invalid_argument("ORB: only firstLevel == 0 is supported");
    if (wta_k != 2 && wta_k != 3 && wta_k != 4) throw std::invalid_argument("ORB: wta_k must be 2, 3 or 4");
    if (scoreType != HARRIS_SCORE && scoreType != FAST_SCORE) throw std::invalid_argument("ORB: unknown scoreType");
    if (patchSize <= 0) throw std::invalid_argument("ORB: patchSize must be > 0");
    if (fastThreshold <= 0) throw std::invalid_argument("ORB: fastThreshold must be > 0");
}

float ORB::getScale(int level) const {
    return std::pow(scaleFactor, static_cast<float>(level - firstLevel));
}

void ORB::detectAndCompute(const Image8U& image, std::vector<KeyPoint>& keypoints, Image8U& descriptors,
                            std::vector<StageTiming>* stageTimings) {
    // Every call fully defines its outputs, whatever the path taken (issue
    // #22): clearing here means an early return cannot leave the caller
    // holding keypoints or a descriptor matrix from a previous call.
    keypoints.clear();
    descriptors = Image8U();

    if (image.cols == 0 || image.rows == 0) return;

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

            // Filter by image border. OpenCV pads every level with a reflected
            // border of max(edgeThreshold, ceil(halfPatchSize * sqrt(2)),
            // HARRIS_BLOCK_SIZE / 2) pixels and samples into the padding; this
            // port has no padding, so the same margin must be enforced on the
            // keypoints themselves or the rotated BRIEF pattern (19 px, fixed
            // table), ICAngles (patchSize / 2) and the Harris block (3 px) read
            // outside the level. Differs from OpenCV only for edgeThreshold < 22
            // with the default patchSize (issue #35).
            const int descPatchSize = static_cast<int>(std::ceil((patchSize / 2) * std::sqrt(2.0)));
            const int border = std::max(std::max(edgeThreshold, patternReach()),
                                        std::max(descPatchSize, HARRIS_BLOCK_SIZE / 2));
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

        // Keep the best 'featuresNum' points of this level, as OpenCV does.
        if (scoreType == HARRIS_SCORE && !filteredKeypoints.empty()) {
            ScopedAccumulator timer(&harrisScoringMs);
            // Keep only the best 2 * featuresNum corners by FAST score before
            // Harris, then the best featuresNum by Harris (#24).
            retainBest(filteredKeypoints, 2 * featuresNum);
            HarrisResponses(imagePyramid[level], layerInfo, filteredKeypoints, HARRIS_BLOCK_SIZE, 0.04f);
            retainBest(filteredKeypoints, featuresNum);
        } else {
            // FAST_SCORE: ranked by the FAST score itself (#21).
            retainBest(filteredKeypoints, featuresNum);
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

    // Scale keypoints to level-0 (original image) coordinates now, as OpenCV
    // does before computing descriptors: computeOrbDescriptors maps them back
    // into their level with 1 / layerScale[octave]. HarrisResponses and
    // ICAngles above read level coordinates, so this must happen between the
    // two stages (issue #34).
    {
        for (auto& kp : allKeypoints) {
            kp.pt *= layerScale[kp.octave];
        }
    }

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
        } else { // wta_k == 3 or 4, validated on entry
            // Each tuple of wta_k points yields 2 bits, so a 32-byte descriptor
            // reads 4 tuples per byte: 128 tuples, as OpenCV's
            // descriptorSize() * 4. The tuples are drawn in order, so the bytes
            // are the same as with the 256 that used to be generated (#25).
            const int ntuples = 32 * 4;
            initializeOrbPattern(pattern0, pattern, ntuples, wta_k, npoints);
        }
    }
    if (stageTimings) stageTimings->push_back({"pattern_init", patternInitMs});

    // 5. Compute Descriptors
    double blurMs = 0.0;
    double descriptorsMs = 0.0;
    int dsize = 32; // 32 bytes = 256 bits
    descriptors = Image8U(dsize, allKeypoints.size());

    for (int level = 0; level < nlevels; ++level) {
        std::vector<KeyPoint> levelKpts;
        std::vector<int> originalIndices;

        for (size_t i = 0; i < allKeypoints.size(); ++i) {
            if (allKeypoints[i].octave == level) {
                levelKpts.push_back(allKeypoints[i]);
                originalIndices.push_back(i);
            }
        }

        // The blurred level is read only by the descriptors (the pyramid was
        // built from the unblurred levels), so a level without keypoints
        // needs no blur (#25).
        if (levelKpts.empty()) continue;

        {
            // Apply Gaussian Blur before computing descriptors for stability
            ScopedAccumulator timer(&blurMs);
            gaussianBlur7x7(imagePyramid[level], imagePyramid[level]);
        }

        {
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

    keypoints = allKeypoints;
}