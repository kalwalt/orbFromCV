// The detectAndCompute pipeline structure follows OpenCV's
// ORB_Impl::detectAndCompute (BSD-3-Clause, Willow Garage) - see NOTICE at
// the repo root for full attribution.
#ifndef ORB_HPP
#define ORB_HPP

#include "orb_core.hpp"
#include <memory>
#include <string>
#include <vector>

class ThreadPool;

// Elapsed time (in milliseconds) for one named stage of detectAndCompute.
// Populated only when a non-null vector is passed in; purely additive
// instrumentation that doesn't affect detection/descriptor results.
struct StageTiming {
    std::string name;
    double ms;
};

// Standalone ORB feature detector and descriptor extractor
class ORB {
public:
    enum ScoreType { HARRIS_SCORE = 0, FAST_SCORE = 1 };

    // nfeatures: at most this many keypoints, as in OpenCV: per level the best
    // 2 * n by FAST score are kept for Harris scoring and then the best n by
    // Harris; points tied with the n-th response are kept too, so the total
    // can exceed nfeatures on images with many identical corners.
    // edgeThreshold: keypoints closer than this to a level's border are dropped.
    // Unlike OpenCV, pyramid levels are not padded, so the effective border is
    // at least max(ceil(patchSize / 2 * sqrt(2)), 19): the reach of the
    // orientation patch and of the fixed 31 px BRIEF pattern. With the default
    // patchSize values below 22 behave as 22.
    // firstLevel: only 0 is supported (level 0 is always the input image at
    // scale 1); OpenCV's upscaled levels below the input are not implemented.
    // Throws std::invalid_argument for nfeatures < 0, scaleFactor <= 1,
    // nlevels < 1, edgeThreshold < 0, firstLevel != 0, wta_k not in {2, 3, 4},
    // an unknown scoreType, patchSize <= 0 or fastThreshold <= 0.
    ORB(int nfeatures = 500, float scaleFactor = 1.2f, int nlevels = 8,
        int edgeThreshold = 31, int firstLevel = 0, int wta_k = 2,
        ScoreType scoreType = HARRIS_SCORE, int patchSize = 31, int fastThreshold = 20);

    // Detects keypoints and computes descriptors
    // image: Input 8-bit grayscale image, as a non-owning view (pointer, cols,
    //        rows, step >= cols). The buffer is only read; level 0 of the
    //        pyramid is the view itself, so nothing is copied. An Image8U
    //        converts implicitly. An empty view (cols or rows 0) clears the
    //        outputs and returns; a malformed one (null data with a non-zero
    //        size, negative size, step < cols) throws std::invalid_argument.
    // keypoints: Extracted keypoints
    // descriptors: Output descriptors (each row corresponds to a keypoint)
    // stageTimings: if non-null, appended with per-stage elapsed time
    void detectAndCompute(const ImageView8U& image, std::vector<KeyPoint>& keypoints, Image8U& descriptors,
                          std::vector<StageTiming>* stageTimings = nullptr);

    // Number of threads detectAndCompute may use for the pyramid resize, the
    // FAST detection and the blur (issue #19); default 1. n <= 0 counts as 1.
    // Returns the effective count: on a wasm32 build without pthreads it is
    // always 1. Output is identical for every thread count: the kernels split
    // rows into bands and merge them in order. Copies of an ORB share its pool.
    int setNumThreads(int n);
    int numThreads() const { return numThreads_; }

private:
    int nfeatures;
    float scaleFactor;
    int nlevels;
    int edgeThreshold;
    int firstLevel;
    int wta_k;
    ScoreType scoreType;
    int patchSize;
    int fastThreshold;

    int numThreads_ = 1;
    std::shared_ptr<ThreadPool> pool_;   // numThreads_ - 1 workers, null for 1 thread

    // Helper to compute scale for a specific pyramid level
    float getScale(int level) const;
};

#endif // ORB_HPP