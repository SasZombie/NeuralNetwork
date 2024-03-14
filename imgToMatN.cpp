#include <iostream>
#include <fstream>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "NeuralNetwork.hpp"

constexpr size_t samples = 10;
constexpr size_t factor = 28 * 28;
constexpr size_t picturePerSample = 20;

nn::Mat PicToMat(const std::string& name, float number)
{

    int imgWidth, imgHeight, imgComp;
    
    uint8_t *data = reinterpret_cast<uint8_t*>(stbi_load(name.c_str(), &imgWidth, &imgHeight, &imgComp, 0));
   
    nn::Mat t(1, factor + samples);

    t.fill(0);
    int indl = 0;
    for(size_t i = 0; i < static_cast<size_t>(imgHeight); ++i)
    {
        for(size_t j = 0; j < static_cast<size_t>(imgWidth); ++j)
        {
            indl = indl + 1;
            size_t index = i * imgWidth + j;

            t.setAt(0, index, data[index]/255.f);
        }
    }
    
    t.setAt(0, factor + number, 1);

    return t;

}

nn::Mat wholeFolder(const std::filesystem::path& dirPath, size_t number)
{
    nn::Mat bigMat{picturePerSample, factor + samples};
    int i = 0;

    for (const auto& file : std::filesystem::recursive_directory_iterator(dirPath))
    {

        nn::Mat inter = PicToMat(file.path(), number);
        bigMat = bigMat + inter;
        ++i;
        if(i == picturePerSample)
            break;
    }

    return bigMat;
}

int main(int argc, const char **argv)
{
    if(argc < 1)
    {
        std::cerr << "No name for mat provided";
        return 1;
    }


    nn::Mat t(picturePerSample * samples, factor + samples); 

    std::filesystem::path bigForlder = "MinstConvert/MinstTrain/";

    for(size_t k = 0; k < samples; ++k)
    {
        std::string currentFolder = bigForlder;

        currentFolder.append(std::to_string(k));
        t = t + wholeFolder(currentFolder, k);
    }

    std::string outPath = argv[1];
    outPath.append(".mat");

    std::ofstream f(outPath, std::ios::binary);
    
    if(!f.is_open())
    {
        std::cerr << "Cannot open file\n";
        return 1;
    }
    t.save(f);

    f.close();
}
