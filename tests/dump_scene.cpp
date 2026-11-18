// Writes the fixture input image as a binary PGM (P5), see fixture_scene.hpp.
//
//   dump_scene <out.pgm>
//
// Run once; the output is committed as tests/fixtures/opencv/scene-640x480.pgm
// and read by scripts/gen-opencv-fixtures.py.
#include "fixture_scene.hpp"
#include <cstdio>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: dump_scene <out.pgm>\n");
        return 2;
    }
    const int width = 640, height = 480;
    Image8U img = cornerScene(width, height);
    FILE* f = std::fopen(argv[1], "wb");
    if (!f) {
        std::perror(argv[1]);
        return 1;
    }
    std::fprintf(f, "P5\n%d %d\n255\n", width, height);
    std::fwrite(img.data.data(), 1, img.data.size(), f);
    std::fclose(f);
    std::printf("wrote %s (%dx%d)\n", argv[1], width, height);
    return 0;
}
