#include <iostream>
#include <fstream>
#include <filesystem>

#include "NeuralNetwork.hpp"

constexpr size_t samples = 10;
constexpr size_t factor = 28 * 28;
constexpr size_t picturePerSample = 1;

int main(int argc, const char **argv)
{
    if(argc < 1)
    {
        std::cerr << "No name for mat provided";
        return 1;
    }


    constexpr size_t sizeTest = 700;
    constexpr size_t rows = 1;
    nn::Mat t{rows, sizeTest};

    for(size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < sizeTest; ++j)
        {
            std::cout << j << '\n';
            t.setAt(i, j, 0);
        }
    }

    for(size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < sizeTest; ++j)
        {
            std::cout << t.getAt(i, j) << '\n';
        }
    }

    std::cout << t.getCols() * t.getRows();
    
    std::string outPath = argv[1];
    outPath.append(".mat");

    std::ofstream f(outPath, std::ios::binary);
    
    std::cout << "Hello\n";
    if(!f.is_open())
    {
        std::cerr << "Cannot open file\n";
        return 1;
    }
    t.save(f);

    f.close();
}
