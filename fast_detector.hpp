// Adapted from OpenCV's FAST feature detector (BSD-3-Clause, Willow Garage) -
// see NOTICE at the repo root for full attribution.
#ifndef FAST_DETECTOR_HPP
#define FAST_DETECTOR_HPP

#include "orb_core.hpp"
#include <vector>

class ThreadPool;

// Fast feature detection implementation (FAST-9)
// nonmaxSuppression: if true, applies non-maximal suppression to cluster corners
// useSimd: run the 16-pixel SSE2/NEON/WASM SIMD128 segment test when the
// library was built with one (orbSimdActive()); false forces the scalar path.
// Both return the same keypoints, responses and order.
// pool: when non-null, the rows are split into bands run on its threads and
// the raw corners merged in band order, so keypoints and order are unchanged.
void detectFAST(const ImageView8U& img, std::vector<KeyPoint>& keypoints, int threshold,
                bool nonmaxSuppression = true, bool useSimd = orbSimdActive(), ThreadPool* pool = nullptr);

#endif // FAST_DETECTOR_HPP