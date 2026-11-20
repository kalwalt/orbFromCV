// HarrisResponses, ICAngles, and computeOrbDescriptors are adapted from
// OpenCV's modules/features2d/src/orb.cpp (BSD-3-Clause, Willow Garage) -
// see NOTICE at the repo root for full attribution.
#ifndef ORB_CORE_HPP
#define ORB_CORE_HPP

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

// Basic geometry and feature structures replacing OpenCV types
struct Point2i {
    int x, y;
    bool operator==(const Point2i& other) const { return x == other.x && y == other.y; }
};

struct Point2f {
    float x, y;
    Point2f& operator*=(float scale) { x *= scale; y *= scale; return *this; }
};

struct Rect {
    int x, y, width, height;
};

struct KeyPoint {
    Point2f pt;
    float size;
    float angle;
    float response;
    int octave;
    int class_id;
};

// Non-owning, read-only view of an 8-bit single-channel image: a pointer to
// the first pixel plus cols, rows and step (bytes between consecutive rows,
// >= cols). Lets a caller hand an existing buffer (a decoded image, a camera
// frame, a sub-image of a larger one) to the pipeline without copying it
// (issue #28). The caller keeps the buffer alive for the duration of the call.
struct ImageView8U {
    const uint8_t* data;
    int cols, rows, step;

    ImageView8U() : data(nullptr), cols(0), rows(0), step(0) {}
    ImageView8U(const uint8_t* d, int c, int r, int s = -1)
        : data(d), cols(c), rows(r), step(s < 0 ? c : s) {}

    const uint8_t* ptr(int y = 0) const { return data + static_cast<size_t>(y) * step; }
    const uint8_t& at(int y, int x) const { return data[static_cast<size_t>(y) * step + x]; }
    bool empty() const { return cols == 0 || rows == 0; }
};

// Minimalistic 8-bit single-channel image container replacing cv::Mat
struct Image8U {
    int cols, rows, step;
    std::vector<uint8_t> data;

    Image8U(int c = 0, int r = 0) : cols(c), rows(r), step(c) {
        data.resize(cols * rows, 0);
    }

    const uint8_t* ptr(int y = 0) const { return data.data() + y * step; }
    uint8_t* ptr(int y = 0) { return data.data() + y * step; }
    const uint8_t& at(int y, int x) const { return data[y * step + x]; }
    uint8_t& at(int y, int x) { return data[y * step + x]; }

    // Every read-only image parameter in the library takes an ImageView8U, so
    // an owned image converts implicitly.
    operator ImageView8U() const { return ImageView8U(data.data(), cols, rows, step); }
};

// Core ORB Functions
void HarrisResponses(const ImageView8U& img, const std::vector<Rect>& layerinfo,
                     std::vector<KeyPoint>& pts, int blockSize, float harris_k);

void ICAngles(const ImageView8U& img, const std::vector<Rect>& layerinfo,
              std::vector<KeyPoint>& pts, const std::vector<int>& u_max, int half_k);

void computeOrbDescriptors(const ImageView8U& imagePyramid, const std::vector<Rect>& layerInfo,
                           const std::vector<float>& layerScale, std::vector<KeyPoint>& keypoints,
                           Image8U& descriptors, const std::vector<Point2i>& _pattern, int dsize, int wta_k);

// True when the library was built with ORB_ENABLE_SIMD=ON and the compiler
// targets SSE2, NEON or WASM SIMD128, i.e. the vector kernels are in use.
bool orbSimdActive();

#endif // ORB_CORE_HPP