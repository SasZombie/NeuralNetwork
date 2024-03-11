#include <iostream>
#include <random>
#include <memory>
#include <iomanip>
#include <algorithm>

float getRandom();
float sigmoid(float x);
float cost(float w1, float w2, float b);

using sample = float[3];


//Or gate
sample train_or[] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 1}
};

//And gate
sample train_and[] = {
    {0, 0, 0},
    {1, 0, 0},
    {0, 1, 0},
    {1, 1, 1}
};

//Nand gate
sample train_nand[] = {
    {0, 0, 1},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 0}
};

const int train_count = 4;

sample *train = train_or;



float getRandom()
{
    std::random_device rd;

    std::default_random_engine engine(rd());

    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    return distribution(engine);
}

float sigmoid(float x)
{
    return 1.f/(1.f + std::exp(-x));
}


float cost(float w1, float w2, float b)
{
    float result = 0.0f;

    for (size_t i = 0; i < train_count; ++i)
    {
        float x0 = train[i][0];
        float x1 = train[i][1];

        float y = sigmoid(x0 * w1 + x1 * w2 + b);
        float d = y - train[i][2];

        result = result + d*d;
    }

    result = result/train_count;

    return result;   
}

int main()
{


    float w1 = getRandom();
    float w2 = getRandom();
    float b = getRandom();
    float eps = 1e-1;
    float rate = 1e-1;


    for (size_t i = 0; i < 100 * 1000; ++i)
    {
        float dw1 = (cost(w1 + eps, w2, b) - cost(w1, w2, b))/eps;
        float dw2 = (cost(w1, w2 + eps, b) - cost(w1, w2, b))/eps;
        float dwb = (cost(w1, w2, b + eps) - cost(w1, w2, b))/eps;

        w1 = w1 - rate * dw1;
        w2 = w2 - rate * dw2;
        b = b - rate * dwb;
    }

    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t j = 0; j < 2; ++j)
        {
            std::cout << i << " | " << j << " => "<< std::fixed<< std::setprecision(4) << sigmoid(i * w1 + j * w2 + b) << '\n';
        }
        
    }
    
    
}