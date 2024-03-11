#include <iostream>
#include <fstream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "NeuralNetwork.hpp"


nn::Mat PicToMat(const std::string& name, float number)
{
    
}

int main(int argc, const char **argv)
{
    if(argc < 4)
    {
        std::cerr << "No image input\nUsage: image.png\n";
        return 1;
    }

    int imgWidth, imgHeight, imgComp;
    int imgWidth2, imgHeight2, imgComp2;

    
    uint8_t *data = reinterpret_cast<uint8_t*>(stbi_load(argv[1], &imgWidth, &imgHeight, &imgComp, 0));
    uint8_t *data2 = reinterpret_cast<uint8_t*>(stbi_load(argv[2], &imgWidth2, &imgHeight2, &imgComp2, 0));


    if(data == nullptr)
    {
        std::cerr << "Cannot load image!\n";
        return 1;
    }

    if(imgComp != 1)
    {
        std::cerr << "Image is not supported\n";
        return 1;
    }

    const size_t factor = imgHeight * imgWidth;
    nn::Mat t(2, factor + 2);


    for(size_t k = 0; k < 2; ++k)
    {
        for(size_t i = 0; i < static_cast<size_t>(imgHeight); ++i)
        {
            for(size_t j = 0; j < static_cast<size_t>(imgWidth); ++j)
            {
                size_t index = i * imgWidth + j;
                if(k == 0)
                    t.setAt(k, index, data[index]/255.f);
                else
                    t.setAt(k, index, data2[index]/255.f);
            }
        }

        t.setAt(k, factor + 1, k);
    }
    

    std::string outPath = argv[3];
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
