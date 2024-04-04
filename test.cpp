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
    nn::Mat td(2, 3, dat);
    nn::Mat td2(2, 3, dat + 6);

    
    td.append(td2);

    td.print();
    
}