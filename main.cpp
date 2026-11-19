#include <iostream>
#include <vector>

// Define the implementation of stb_image only in this file
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "orb.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }

    const char* imagePath = argv[1];

    // 1. Load the image using stb_image
    int width, height, channels;
    // Force loading as 1 channel (grayscale) because ORB works on grayscale images
    unsigned char* img_data = stbi_load(imagePath, &width, &height, &channels, 1);

    if (!img_data) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        return 1;
    }

    std::cout << "Image loaded: " << width << "x" << height << " pixels." << std::endl;

    // 2. Wrap the stb_image buffer in a non-owning view: ORB reads it in
    // place, so no copy is needed (the buffer stays alive until step 5).
    ImageView8U grayImage(img_data, width, height);

    // 3. ORB initialization
    // Default parameters: 500 keypoints, scaleFactor 1.2, 8 levels
    ORB orb(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);

    std::vector<KeyPoint> keypoints;
    Image8U descriptors;

    std::cout << "Extracting keypoints and descriptors..." << std::endl;

    // 4. Run the algorithm
    orb.detectAndCompute(grayImage, keypoints, descriptors);
    stbi_image_free(img_data);

    // 5. Print the results
    std::cout << "Extraction complete." << std::endl;
    std::cout << "Keypoints found: " << keypoints.size() << std::endl;

    if (!keypoints.empty()) {
        std::cout << "\nFirst keypoint:" << std::endl;
        std::cout << " - Coordinates (x, y): (" << keypoints[0].pt.x << ", " << keypoints[0].pt.y << ")" << std::endl;
        std::cout << " - Response (Harris/FAST): " << keypoints[0].response << std::endl;
        std::cout << " - Angle: " << keypoints[0].angle << " degrees" << std::endl;
        std::cout << " - Pyramid level (octave): " << keypoints[0].octave << std::endl;

        std::cout << " - Descriptor (first 8 bytes, hex): ";
        const uint8_t* desc_ptr = descriptors.ptr(0);
        for (int i = 0; i < 8; ++i) {
            printf("%02X ", desc_ptr[i]);
        }
        std::cout << "..." << std::endl;
    }

    return 0;
}