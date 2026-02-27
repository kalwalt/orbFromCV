#ifndef ORB_HPP
#define ORB_HPP

#include "orb_core.hpp"
#include <vector>

// Standalone ORB feature detector and descriptor extractor
class ORB {
public:
    enum ScoreType { HARRIS_SCORE = 0, FAST_SCORE = 1 };

    ORB(int nfeatures = 500, float scaleFactor = 1.2f, int nlevels = 8, 
        int edgeThreshold = 31, int firstLevel = 0, int wta_k = 2, 
        ScoreType scoreType = HARRIS_SCORE, int patchSize = 31, int fastThreshold = 20);

    // Detects keypoints and computes descriptors
    // image: Input 8-bit grayscale image
    // keypoints: Extracted keypoints
    // descriptors: Output descriptors (each row corresponds to a keypoint)
    void detectAndCompute(const Image8U& image, std::vector<KeyPoint>& keypoints, Image8U& descriptors);

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

    // Helper to compute scale for a specific pyramid level
    float getScale(int level) const;
};

#endif // ORB_HPP