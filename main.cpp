#include <iostream>
#include <vector>

// Define the implementation of stb_image only in this file
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "orb.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <percorso_immagine>" << std::endl;
        return 1;
    }

    const char* imagePath = argv[1];

    // 1. Load the image using stb_image
    int width, height, channels;
    // Force loading as 1 channel (grayscale) because ORB works on grayscale images
    unsigned char* img_data = stbi_load(imagePath, &width, &height, &channels, 1);

    if (!img_data) {
        std::cerr << "Errore nel caricamento dell'immagine: " << imagePath << std::endl;
        return 1;
    }

    std::cout << "Immagine caricata: " << width << "x" << height << " pixel." << std::endl;

    // 2. Conversion to our Image8U format
    Image8U grayImage(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            grayImage.at(y, x) = img_data[y * width + x];
        }
    }

    // Free stb_image memory, we don't need it anymore
    stbi_image_free(img_data);

    // 3. ORB initialization
    // Default parameters: 500 keypoints, scaleFactor 1.2, 8 levels
    ORB orb(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);

    std::vector<KeyPoint> keypoints;
    Image8U descriptors;

    std::cout << "Estrazione keypoints e descrittori in corso..." << std::endl;

    // 4. Run the algorithm
    orb.detectAndCompute(grayImage, keypoints, descriptors);

    // 5. Print the results
    std::cout << "Estrazione completata!" << std::endl;
    std::cout << "Keypoints trovati: " << keypoints.size() << std::endl;

    if (!keypoints.empty()) {
        std::cout << "\nDettagli del primo Keypoint:" << std::endl;
        std::cout << " - Coordinate (x, y): (" << keypoints[0].pt.x << ", " << keypoints[0].pt.y << ")" << std::endl;
        std::cout << " - Risposta (Harris/FAST): " << keypoints[0].response << std::endl;
        std::cout << " - Angolo: " << keypoints[0].angle << " gradi" << std::endl;
        std::cout << " - Livello piramide (Octave): " << keypoints[0].octave << std::endl;

        std::cout << " - Descrittore (primi 8 byte in esadecimale): ";
        const uint8_t* desc_ptr = descriptors.ptr(0);
        for (int i = 0; i < 8; ++i) {
            printf("%02X ", desc_ptr[i]);
        }
        std::cout << "..." << std::endl;
    }

    return 0;
}