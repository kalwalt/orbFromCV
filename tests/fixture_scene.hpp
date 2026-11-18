#ifndef FIXTURE_SCENE_HPP
#define FIXTURE_SCENE_HPP

#include "orb_core.hpp"
#include <cstdint>

// The input image of the OpenCV ground-truth fixtures (issue #32).
//
// Integer arithmetic only, so every compiler and platform produces the same
// bytes. It is dumped ONCE to tests/fixtures/opencv/scene-640x480.pgm by the
// dump_scene target and committed; the Python generator reads that file, so
// both sides see identical pixels without re-implementing this function. A
// test checks that the committed PGM still equals cornerScene(640, 480).
//
// Content: a textured background (per-pixel hash noise over 8x8 cells of
// varying brightness) with a few rectangles of inverted / darkened / xor-ed
// texture, so FAST finds corners on every pyramid level and at every
// orientation, including right at the rectangle edges.
inline uint32_t fixtureHash32(uint32_t x) { // "lowbias32"
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

inline uint8_t fixtureScenePixel(int x, int y, int width) {
    uint32_t coarse = fixtureHash32((static_cast<uint32_t>(y >> 3) * 256u + static_cast<uint32_t>(x >> 3)) ^ 0x9e3779b9U) & 0xffu;
    uint32_t fine = fixtureHash32((static_cast<uint32_t>(y) * static_cast<uint32_t>(width) + static_cast<uint32_t>(x)) ^ 0x85ebca6bU) & 0x3fu;
    uint32_t v = ((coarse * 3u) >> 2) + fine; // 0..254
    if (x >= 60 && x < 260 && y >= 70 && y < 230) v = 255u - v;
    if (x >= 330 && x < 580 && y >= 140 && y < 330) v = (v >> 1) + 96u;
    if (x >= 150 && x < 480 && y >= 340 && y < 450) v ^= 0x5au;
    if (x >= 20 && x < 120 && y >= 300 && y < 460) v = (v * 3u) >> 2;
    return static_cast<uint8_t>(v & 0xffu);
}

inline Image8U cornerScene(int width, int height) {
    Image8U img(width, height);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            img.at(y, x) = fixtureScenePixel(x, y, width);
    return img;
}

#endif // FIXTURE_SCENE_HPP
