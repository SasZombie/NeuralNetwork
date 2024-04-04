#include <iostream>
#include "NeuralNetwork.hpp"

float dat[] = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,
    10, 11, 12,
    13, 14, 15,
    16, 17, 18,
    19, 20, 21,
    22, 23, 24,
    25, 26, 27,
    28, 29, 30
};

float getRandom1() noexcept
{
    std::random_device rd;

    std::default_random_engine engine(rd());

    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    return distribution(engine);
}

bool isEqual(float a, float b, float epsilon)
{
    if (fabs(a - b) < epsilon)
    {
        return true; 
    } 

    return fabs(a) < epsilon && fabs(b) < epsilon;

}

int main()
{
    std::cout << isEqual(0, 0, 0.0f);
    
}