#include <iostream>
#include "NeuralNetwork.hpp"

float dat[] = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,
    10, 11, 12,
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,
    10, 11, 12,
    7, 8, 9
};

float getRandom1() noexcept
{
    std::random_device rd;

    std::default_random_engine engine(rd());

    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    return distribution(engine);
}

int main()
{
    nn::Mat td(9, 3, dat);

    size_t arch[] = {2, 2, 1};

    nn::NN nn(arch, 3);

    nn::NN ln;
    std::ofstream f("test", std::ios::binary | std::ios::app);
    nn.rand();
    nn.save(f);

    f.close();
    nn.print();

    std::ifstream g("test", std::ios::binary);

    ln.load(g);

    std::cout << "---------------------------------------\n";

    ln.print();
}