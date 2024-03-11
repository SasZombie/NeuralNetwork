#include <iostream>
#include <random>

float getRandom();

float cost(float w, float b);

float trainingData[][2] = {
    {0, 0},
    {1, 2},
    {2, 4},
    {4, 8}
};


const int train_count = sizeof(trainingData)/sizeof(trainingData[0]);

float getRandom()
{
    std::random_device rd;

    std::default_random_engine engine(rd());

    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    return distribution(engine);
}


float cost(float w, float b)
{
    float result = 0.0f;

    for (size_t i = 0; i < train_count; ++i)
    {
        float x = trainingData[i][0];
        float y = x * w + b;
        float d = y - trainingData[i][1];
        
        result = result + d*d;
    }

    result = result/train_count;

    return result;   
}

int main()
{

    float w = getRandom() * 10.f;
    float b = getRandom() * 10.f;
    float eps = 1e-3;
    float rate = 1e-3;


    for (size_t i = 0; i < 15000; ++i)
    {
        float dw = (cost(w + eps, b) - cost(w, b))/eps;
        float db = (cost(w, b + eps) - cost(w, b))/eps;
        w = w - rate * dw;
        b = b - rate * db;
    }

    std::cout << w;

}