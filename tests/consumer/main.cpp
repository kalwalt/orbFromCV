// Smoke test for the installed package: include the public header through
// the imported target, run detectAndCompute on a view of a synthetic buffer
// and fail if nothing comes out.
#include <orb.hpp>

#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    const int W = 160, H = 120, STEP = 192; // padded rows, as a camera frame might have
    std::vector<uint8_t> frame(static_cast<size_t>(STEP) * H, 30);
    for (int y = 10; y < H; y += 20)
        for (int x = 10; x < W; x += 20)
            for (int dy = -3; dy <= 3; ++dy)
                for (int dx = -3; dx <= 3; ++dx) frame[static_cast<size_t>(y + dy) * STEP + x + dx] = 220;

    ORB orb(100, 1.2f, 3, 16, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    std::vector<KeyPoint> keypoints;
    Image8U descriptors;
    orb.detectAndCompute(ImageView8U(frame.data(), W, H, STEP), keypoints, descriptors);

    std::printf("StandaloneORB consumer: %zu keypoints, %d x %d descriptors\n",
                keypoints.size(), descriptors.rows, descriptors.cols);
    return (keypoints.empty() || descriptors.rows != static_cast<int>(keypoints.size())) ? 1 : 0;
}
