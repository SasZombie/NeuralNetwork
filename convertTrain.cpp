#include <iostream>
#include <filesystem>
#include <string>
#include "stb_image.h"
#include "stb_image_write.h"

namespace fs = std::filesystem;

void convertJpgToPng(const std::string& sourcePath, const std::string& destPath) {
    int ind = 0;
    for (const auto& entry : fs::directory_iterator(sourcePath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
            // Load JPG image using stb_image
            int width, height, channels;
            unsigned char* data = stbi_load(entry.path().string().c_str(), &width, &height, &channels, 0);

            if (data != nullptr) {
                // Create the destination path with a PNG extension
                std::string destFilePath = destPath + "/" + entry.path().stem().string() + ".png";

                // Save the loaded image as PNG using stb_image_write
                stbi_write_png(destFilePath.c_str(), width, height, channels, data, width * channels);

                // Free the loaded image data
                stbi_image_free(data);

                std::cout << "Converted: " << entry.path().filename() << std::endl;
            } else {
                std::cerr << "Failed to load image: " << entry.path().filename() << std::endl;
            }
            ++ind;
            if(ind == 20)
                break;
        }
    }
}

int main() {
    std::string sourceFolder = "Minst/trainingSet/trainingSet/0";
    std::string destFolder = "MinstConvert/0";

    // Create the destination folder if it doesn't exist
    if (!fs::exists(destFolder)) {
        fs::create_directory(destFolder);
    }

    convertJpgToPng(sourceFolder, destFolder);

    return 0;
}
