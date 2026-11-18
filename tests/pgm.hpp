#ifndef PGM_HPP
#define PGM_HPP

#include "orb_core.hpp"
#include <cstdio>
#include <string>

// Minimal binary PGM (P5, maxval 255) reader for the committed fixtures.
// Returns an empty image (cols == 0) on any error.
inline Image8U readPGM(const std::string& path) {
    Image8U img;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return img;
    char magic[3] = {0, 0, 0};
    int width = 0, height = 0, maxval = 0;
    if (std::fscanf(f, "%2s %d %d %d", magic, &width, &height, &maxval) != 4 ||
        magic[0] != 'P' || magic[1] != '5' || maxval != 255 || width <= 0 || height <= 0) {
        std::fclose(f);
        return img;
    }
    std::fgetc(f); // the single whitespace byte after maxval
    img = Image8U(width, height);
    size_t n = std::fread(img.data.data(), 1, img.data.size(), f);
    std::fclose(f);
    if (n != img.data.size()) return Image8U();
    return img;
}

#endif // PGM_HPP
