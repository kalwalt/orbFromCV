#ifndef FAST_DETECTOR_HPP
#define FAST_DETECTOR_HPP

#include "orb_core.hpp"
#include <vector>

// Fast feature detection implementation (FAST-9)
// nonmaxSuppression: if true, applies non-maximal suppression to cluster corners
void detectFAST(const Image8U& img, std::vector<KeyPoint>& keypoints, int threshold, bool nonmaxSuppression = true);

#endif // FAST_DETECTOR_HPP