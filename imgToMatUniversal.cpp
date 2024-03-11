#include <iostream>
#include <fstream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"
#include "NeuralNetwork.hpp"

int main(int argc, const char **argv)
{
    if(argc < 3)
    {
        std::cerr << "No image input\nUsage: image.png\n";
        return 1;
    }

    int imgWidth, imgHeight, imgComp;
    
    unsigned char *data = reinterpret_cast<unsigned char*>(stbi_load(argv[1], &imgWidth, &imgHeight, &imgComp, 0));

    if(data == nullptr)
    {
        std::cerr << "Cannot load image!\n";
        return 1;
    }
 
    constexpr size_t factor = 2;
      

    nn::Mat t(imgWidth * imgHeight * imgComp, factor + imgComp);

    for(size_t i = 0; i < static_cast<size_t>(imgHeight); ++i)
    {
        for(size_t j = 0; j < static_cast<size_t>(imgWidth); ++j)
        {
            int index = (i * imgWidth + j) * imgComp;

            t.setAt(index, 0, static_cast<float>(j)/(imgWidth - 1));
            t.setAt(index, 1, static_cast<float>(i)/(imgHeight - 1));
            t.setAt(index, 2, data[index]/255.f);

            for(size_t k = 0; k < static_cast<size_t>(imgComp); ++k)
            {
                t.setAt(index, k + 3, data[index + (k + 1)]/255.f);
            }
            
        }
    }
    
    std::string outPath = argv[2];
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
