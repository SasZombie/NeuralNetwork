#include <iostream>
#include <random>
#include "NeuralNetwork.hpp"



int getRandomNumber(int min, int max)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> distribution(min, max);

    return distribution(mt);
}

enum class shapes {
    circle, 
    rectangle
};

int main()
{

    constexpr int WIDTH = 28, HEIGHT = 28;
    constexpr int SHAPES = 2, SHAPES_NUMBER = 100 * SHAPES;
    int x = getRandomNumber(0, WIDTH), y = getRandomNumber(0, HEIGHT), r = getRandomNumber(5, 50), w = getRandomNumber(0, 50), h = getRandomNumber(0, 50);

    constexpr int factor = 28 * 32;
    nn::Mat bigMat{SHAPES_NUMBER, 4 * factor + 2};

    for(size_t k = 0; k < SHAPES_NUMBER; ++k)
    {
        // do{
        //     r = getRandomNumber(5, 50);
        //     x = getRandomNumber(0, WIDTH);
        //     y = getRandomNumber(0, HEIGHT);
        // }while(x + r > WIDTH || x - r < 0 || y + r > HEIGHT || y - r < 0);

        // int distX = x - r, distY = y - r, pDistX = x + r, pDistY = y + r;
        int pDistX = 10, pDistY = 10, distX = 10, distY = 10;
        for(int i = 0; i < 28; ++i)
        {
            for(int j = 0; j < 32; ++j)
            {
                int index = (i * 32 + j) * 4;

                if(i > distX && i < pDistX && j > distY && j < pDistY)
                {
                    bigMat.setAt(k, index, 1);
                    bigMat.setAt(k, index + 1, 1);
                    bigMat.setAt(k, index + 2, 1);
                    bigMat.setAt(k, index + 3, 1);

                }
                else
                {
                    bigMat.setAt(k, index, 0);
                    bigMat.setAt(k, index + 1, 0);
                    bigMat.setAt(k, index + 2, 0);
                    bigMat.setAt(k, index + 3, 0);
                }
            }
        }

        bigMat.setAt(k, factor - 1, 1);
        bigMat.setAt(k, factor, 0);

        // do{
        //     w = getRandomNumber(10, 100);
        //     h = getRandomNumber(10, 100);
        //     x = getRandomNumber(0, WIDTH);
        //     y = getRandomNumber(0, HEIGHT);
        // }while(x + w > WIDTH || x - w < 0 || y + h > HEIGHT || y - h < 0);

        // distX = x - w, distY = y - h, pDistX = x + w, pDistY = y + h;


        // for(int i = 0; i < 28; ++i)
        // {
        //     for(int j = 0; j < 32; ++j)
        //     {
        //         int index = (i * 32 + j) * 4;
            
        //         bigMat.setAt(index, 0, static_cast<float>(j)/(WIDTH - 1));
        //         bigMat.setAt(index, 1, static_cast<float>(i)/(HEIGHT - 1));

        //         if(i > distX && i < pDistX && j > distY && j < pDistY)
        //         {
        //             bigMat.setAt(index, 2, 1);
        //             bigMat.setAt(index, 3, 1);
        //             bigMat.setAt(index, 4, 1);
        //             bigMat.setAt(index, 5, 1);
        //         }
        //         else
        //         {
        //             bigMat.setAt(index, 2, 0);
        //             bigMat.setAt(index, 3, 0);
        //             bigMat.setAt(index, 4, 0);
        //             bigMat.setAt(index, 5, 0);
        //         }

        //         bigMat.setAt(index, 6, static_cast<int>(shapes::rectangle));
        //     }
        // }
    }

    std::ofstream f{"shapesBig.mat"};
    bigMat.save(f);

    f.close();
    
}