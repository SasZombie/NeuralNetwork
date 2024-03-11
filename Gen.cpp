#include "NeuralNetwork.hpp"
#include "iostream"


float td[] = {
    0, 0, 0,
    1, 0, 1,
    0, 1, 1,
    1, 1, 0
};
int main()
{
    size_t arch[] = {2, 2, 1};

    size_t stride = sizeof(arch)/sizeof(arch[0]);

    size_t n = sizeof(td)/sizeof(td[0])/stride;

    nn::Mat ti(n, 2, stride, td);

    nn::Mat to(n, 1, stride, td+2);

    ti.append(to);

    std::ofstream saveFile{"save.mat", std::ios::binary};

    if(!saveFile.is_open())
    {
        std::cerr << "Could not open file";
        return 1;
    }

    ti.save(saveFile);
    saveFile.close();

    std::ifstream readFile{"save.mat", std::ios::binary};
    nn::Mat testMat;

    testMat.load(readFile);
    
    readFile.close();
}