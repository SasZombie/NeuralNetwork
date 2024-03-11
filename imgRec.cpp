#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <raylib.h>
#include "stb_image_write.h"
#include <limits>
#include "NeuralNetwork.hpp"

constexpr int WIDTH = 1280, HEIGHT = 720;

void render(nn::NN nn, int x, int y, int w, int h);
std::tuple<float, float> minMaxPlot(const std::vector<float>&plot);
void renderPlot(const std::vector<float>& cost, int x, int y, int w, int h);
bool endsWith(const std::string& fullString, const std::string& ending);
void verification();
int getRandomNumber(int min, int max);

enum class shapes {
    circle, 
    rectangle
};


int getRandomNumber(int min, int max)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> distribution(min, max);

    return distribution(mt);
}

void render(nn::NN nn, int x, int y, int w, int h)
{

    Color lowColor{0xFF, 0x00, 0xFF, 0xFF};
    Color highColor{0xFF, 0xFF, 0x00, 0xFF};

    size_t archCount = nn.getCount() + 1;

    float radios = h * 0.04f;
    int layerBorderVPad = 50;
    int layerBorderHPad = 50;
    int nnWidtdh = w - 2*layerBorderHPad;
    int nnHeight = h - 2*layerBorderVPad;
    int nnX = x + w/2 - nnWidtdh/2;
    int nnY = y + h/2 - nnHeight/2;
    int layerHPad = nnWidtdh / archCount;


    for(size_t l = 0; l < archCount; ++l)
    {    
        int layerVPad = nnHeight/nn.getAsCols(l);
        for(size_t i = 0; i < nn.getAsCols(l); ++i)
        {
            int cx = nnX + layerHPad * l + layerHPad/2;
            int cy = nnY + layerVPad * i + layerVPad/2;

            if(l+1 < archCount)
            { 
                int layerVPad2 = nnHeight/nn.getAsCols(l+1);

                for (size_t j = 0; j < nn.getAsCols(l+1); ++j)
                {
                    
                    int cx2 = nnX + (l+1) * layerHPad + layerHPad/2;
                    int cy2 = nnY + j*layerVPad2 + layerVPad2/2; 
                        // std::cout << nn.getAsCols(l) << ' ' << nn.getAsCols(l+1) << ' ' << nn.getAtAs(l, j, i)<< '\n';    
                    highColor.a = std::floor(255.f * nn::sig(nn.getAtWs(l, j, i)));
                    DrawLine(cx, cy, cx2, cy2, ColorAlphaBlend(lowColor, highColor, WHITE));

                }

            }
            if(l > 0)
            {
                highColor.a = std::floor(255.f * nn::sig(nn.getAtBs(l-1, 0, i)));
                DrawCircle(cx, cy, radios, ColorAlphaBlend(lowColor, highColor, WHITE));
            }
            else
            {
                DrawCircle(cx, cy, radios, GRAY);
            }
            
        }
    }
}

std::tuple<float, float> minMaxPlot(const std::vector<float>&plot)
{
    float max = std::numeric_limits<float>::min();
    float min = std::numeric_limits<float>::max();

    for(const float f : plot)
    {
        if(f > max)
            max = f;
        if(f < min)
            min = f;
    }

    return std::make_tuple(min, max);
}
constexpr int factor = 28 * 28;

void generate(nn::Mat &bigMat, int width, int height)
{
 int pDistX = 10, pDistY = 10, distX = 10, distY = 10;
        for(int i = 0; i < 28; ++i)
        {

 for(int j = 0; j < 32; ++j)
            {
                int index = (i * 32 + j) * 4;

                if(i > distX && i < pDistX && j > distY && j < pDistY)
                {
                    bigMat.setAt(0, index, 1);
                    bigMat.setAt(0, index + 1, 1);
                    bigMat.setAt(0, index + 2, 1);
                    bigMat.setAt(0, index + 3, 1);

                }
                else
                {
                    bigMat.setAt(0, index, 0);
                    bigMat.setAt(0, index + 1, 0);
                    bigMat.setAt(0, index + 2, 0);
                    bigMat.setAt(0, index + 3, 0);
                }
            }
        }

        bigMat.setAt(0, factor - 1, 1);
        bigMat.setAt(0, factor, 0);


    // t.clear();
    // int x = getRandomNumber(0, WIDTH), y = getRandomNumber(0, HEIGHT), r = getRandomNumber(5, 50), w = getRandomNumber(0, 50), h = getRandomNumber(0, 50);

    // do{
    //     w = getRandomNumber(10, 100);
    //     h = getRandomNumber(10, 100);
    //     x = getRandomNumber(0, WIDTH);
    //     y = getRandomNumber(0, HEIGHT);
    // }while(x + w > WIDTH || x - w < 0 || y + h > HEIGHT || y - h < 0);

    // int distX = x - w, distY = y - h, pDistX = x + w, pDistY = y + h;


    // for(int i = 0; i < 28; ++i)
    // {
    //     for(int j = 0; j < 32; ++j)
    //     {
    //         int index = (i * 32 + j) * 4;
        
    //         t.setAt(index, 0, static_cast<float>(j)/(WIDTH - 1));
    //         t.setAt(index, 1, static_cast<float>(i)/(HEIGHT - 1));

    //         if(i > distX && i < pDistX && j > distY && j < pDistY)
    //         {
    //             t.setAt(index, 2, 1);
    //             t.setAt(index, 3, 1);
    //             t.setAt(index, 4, 1);
    //             t.setAt(index, 5, 1);
    //         }
    //         else
    //         {
    //             t.setAt(index, 2, 0);
    //             t.setAt(index, 3, 0);
    //             t.setAt(index, 4, 0);
    //             t.setAt(index, 5, 0);
    //         }

    //         t.setAt(index, 6, static_cast<int>(shapes::rectangle));
    //     }
    // }
}
 


void generateCircle(nn::Mat &t, int width, int height) 
{
    t.clear();
    int x = getRandomNumber(0, WIDTH), y = getRandomNumber(0, HEIGHT), r = getRandomNumber(5, 50), w = getRandomNumber(0, 50), h = getRandomNumber(0, 50);
    do{
        r = getRandomNumber(5, 50);
        x = getRandomNumber(0, WIDTH);
        y = getRandomNumber(0, HEIGHT);
    }while(x + r > WIDTH || x - r < 0 || y + r > HEIGHT || y - r < 0);

    int distX = x - r, distY = y - r, pDistX = x + r, pDistY = y + r;
    for(int i = 0; i < 28; ++i)
    {
        for(int j = 0; j < 32; ++j)
        {
            int index = (i * 32 + j) * 4;
        
            t.setAt(index, 0, static_cast<float>(j)/(WIDTH - 1));
            t.setAt(index, 1, static_cast<float>(i)/(HEIGHT - 1));

            if(i > distX && i < pDistX && j > distY && j < pDistY)
            {
                t.setAt(index, 2, 1);
                t.setAt(index, 3, 1);
                t.setAt(index, 4, 1);
                t.setAt(index, 5, 1);
            }
            else
            {
                t.setAt(index, 2, 0);
                t.setAt(index, 3, 0);
                t.setAt(index, 4, 0);
                t.setAt(index, 5, 0);
            }
            t.setAt(index, 6, static_cast<int>(shapes::circle));
        }
    }
}


void generateCircle(std::vector<float> &t, int width, int height) 
{
    t.clear();
    int x = getRandomNumber(0, WIDTH), y = getRandomNumber(0, HEIGHT), r = getRandomNumber(5, 50), w = getRandomNumber(0, 50), h = getRandomNumber(0, 50);
    do{
        r = getRandomNumber(5, 50);
        x = getRandomNumber(0, WIDTH);
        y = getRandomNumber(0, HEIGHT);
    }while(x + r > WIDTH || x - r < 0 || y + r > HEIGHT || y - r < 0);

    int distX = x - r, distY = y - r, pDistX = x + r, pDistY = y + r;
    for(int i = 0; i < 28; ++i)
    {
        for(int j = 0; j < 32; ++j)
        {
            int index = (i * 32 + j) * 6;
        
            t[index] = static_cast<float>(j)/(WIDTH - 1);
            t[index + 1] = static_cast<float>(i)/(HEIGHT - 1);

            if(i > distX && i < pDistX && j > distY && j < pDistY)
            {
                t[index + 2] =  1;
                t[index + 3] =  1;
                t[index + 4] =  1;
                t[index + 5] =  1;
            }
            else
            {
                t[index + 2] =  0;
                t[index + 3] =  0;
                t[index + 4] =  0;
                t[index + 5] =  0;
            }
            t[index + 6] = static_cast<int>(shapes::circle);
        }
    }
}


void generate(std::vector<float> &t, int width, int height) 
{
    t.clear();
    // int x = getRandomNumber(0, WIDTH), y = getRandomNumber(0, HEIGHT), r = getRandomNumber(5, 50), w = getRandomNumber(0, 50), h = getRandomNumber(0, 50);

    // do{
    //     w = getRandomNumber(10, 100);
    //     h = getRandomNumber(10, 100);
    //     x = getRandomNumber(0, WIDTH);
    //     y = getRandomNumber(0, HEIGHT);
    // }while(x + w > WIDTH || x - w < 0 || y + h > HEIGHT || y - h < 0);

    // int distX = x - r, distY = y - r, pDistX = x + r, pDistY = y + r;

    int pDistX = 10, pDistY = 10, distX = 10, distY = 10;

    for(int i = 0; i < 28; ++i)
    {
        for(int j = 0; j < 32; ++j)
        {
            int index = (i * 32 + j) * 4;

            if(i > distX && i < pDistX && j > distY && j < pDistY)
            {
                t[index] =  1;
                t[index + 1] =  1;
                t[index + 2] =  1;
                t[index + 3] =  1;
            }
            else
            {
                t[index] =  0;
                t[index + 1] =  0;
                t[index + 2] =  0;
                t[index + 3] =  0;
            }
            t[index + 4] = static_cast<int>(shapes::circle);
        }
    }
}

void renderPlot(const std::vector<float>& cost, int x, int y, int w, int h)
{

    static constexpr int additionalXOffset = 7;
    static constexpr int additionalYOffset = 20;

    auto[min, max] = minMaxPlot(cost);

    if(min > 0)
        min = 0;

    size_t n = cost.size();
    if(n < 1000)
        n = 1000;

    for(size_t i = 0; i + 1 < cost.size(); ++i)
    {
        int rx1 = x + additionalXOffset + static_cast<float>(w)/n * i;
        int ry1 = y + (1 - (cost[i] - min)/(max - min))*h;
        int rx2 = x + additionalXOffset + static_cast<float>(w)/n * (i+1);
        int ry2 = y + (1 - (cost[i+1] - min)/(max - min))*h;

       DrawLine(rx1, ry1, rx2, ry2, RED);
    }


    DrawLine(x, y, x, h + y, RAYWHITE);

    DrawLine(x, h + y, w, h + y, RAYWHITE);
    DrawText("Time->", x + w - 50, h + y, 0.04f * h, RAYWHITE);
    DrawText("0", x, h + y + additionalYOffset, 0.04f * h, RAYWHITE);
    DrawText("Cost", x + additionalXOffset, y - additionalYOffset, 0.04f * h, RAYWHITE);

}

bool endsWith(const std::string& fullString, const std::string& ending) {
    return fullString.size() >= ending.size() &&
           fullString.compare(fullString.size() - ending.size(), ending.size(), ending) == 0;
}



int main(int argc, char **argv)
{
    if(argc < 3)
    {
        std::cerr << "Not enough arguments\nUsage: file1.arch, file2.mat";
        return 1;
    }

    std::fstream file(argv[1]);

    if(!file.is_open())
    {
        std::cerr << "Unable to open file:" << argv[1];
        return 1;
    }

    if(!endsWith(argv[1], ".arch"))
    {
        std::cerr << "File 1 is not a neural network architecture:\nUsage: file1.arch, file2.mat" << argv[1];
        return 1;
    }
    
    if(!endsWith(argv[2], ".mat"))
    {
        std::cerr << "File 2 is not a matrix file:\nUsage: file1.arch, file2.mat" << argv[2];
        return 1;
    }

    std::vector<size_t> arch;
    size_t x;
    while(file>>x)
    {
        arch.push_back(x);
    }

    file.close();
    
    nn::NN nn(arch.data(), arch.size()); 
    nn::NN grad(arch.data(), arch.size());

    nn.rand(-1, 1);
    
    
    std::ifstream inFile(argv[2]);

     if(!inFile.is_open())
    {
        std::cerr << "Unable to open file:" << argv[2];
        return 1;
    }

    nn::Mat bigMat;

    bigMat.load(inFile);
    inFile.close();

    bigMat.shuffle();

    float rate = 0.5f;

    size_t stride = bigMat.getStride();
    size_t rows = bigMat.getRows();
    size_t cols = bigMat.getCols();

    size_t count = nn.getCount();
    size_t inSz = arch[0];
    size_t outSz = arch[count];
    size_t batchSize = 28;
    size_t batchPerFrame = 200;
    size_t batchBegin = 0;
    size_t batchCount = (rows + batchSize - 1)/batchSize;

    if(nn.getCount() < 1 || bigMat.getCols() != inSz + outSz)
    {
        std::cerr << "Incorrect format files\n";
        return 1;
    }

    bool pause = true;

    int iterations = 0;
    constexpr int maxIterations = 100 * 1000;

    float cost = 0.f;
    std::vector<float> costFunction = {0.f};
    
    Color color{0x18, 0x18, 0x18, 0xFF};
        
    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(WIDTH, HEIGHT, "Neural Network Visualiser");
    SetTargetFPS(60);
    Image prevImage = GenImageColor(28, 28, BLACK);
    Texture2D prevTexture = LoadTextureFromImage(prevImage);

    nn::Mat smallMat{1, 4 * 28 * 32 + 2};
    std::vector<float> smallerMat;
    bool draw = true;
    

    generate(smallMat, 10, 10);

    while (!WindowShouldClose())
    {
        if(IsKeyPressed(KEY_SPACE))
        {
            pause = !pause;
        }
   
        if(IsKeyPressed(KEY_R))
        {
            iterations = 0;
            costFunction.clear();
            cost = 0.f;
            nn.clear();
            nn.rand(-1, 1);
        }

        if(IsKeyPressed(KEY_C))
        {
            generateCircle(smallerMat, 28, 28);
            draw = true;
        }

        if(IsKeyPressed(KEY_F))
        {
            // generate(smallMat, 28, 28);
            generate(smallerMat, 28, 28);
        }

        for(size_t i = 0; i < batchPerFrame && iterations < maxIterations && !pause; ++i)
        {   
            size_t size = batchSize;
            if(batchBegin + batchSize >= rows)
            {
                size = rows - batchBegin;
            }

            const float *data = bigMat.getData();

            nn::Mat inBatchMat{size, inSz, stride, data + (batchBegin * cols + 0)};
            nn::Mat outBatchMat{size, outSz, stride, data + (batchBegin * cols + inSz)};

            nn.backProp(grad, inBatchMat, outBatchMat);   
            nn.learn(grad, rate);

            cost = cost + nn.cost(inBatchMat, outBatchMat);
            batchBegin = batchBegin + batchSize;

            if(batchBegin >= rows)
            {
                ++iterations;
                costFunction.push_back(cost/batchCount);
                cost = 0.f;
                batchBegin = 0;
                bigMat.shuffle();   
            }
            
        }
        
        BeginDrawing();
        ClearBackground(color);


        int rw, rh, rx, ry;
        int sw = GetRenderWidth();
        int sh = GetRenderHeight();
        
        rw = sw/3;
        rh = sh*2/3;
        rx = 3;
        ry = sh/2 - rh/2;
        renderPlot(costFunction, rx, ry, rw, rh);
        rx = rx + rw;
        render(nn, rx, ry, rw, rh);
        rx = rx + rw;

        for (size_t i = 0; i < 28; ++i)
        {
            for(size_t j = 0; j < 32; ++j)
            {

                int index = (i * 32 + j) * 4;


                nn.setAtAs(0, 0, 1, smallMat.getAt(0, index));
                nn.setAtAs(0, 0, 2, smallMat.getAt(0, index+1));
                nn.setAtAs(0, 0, 3, smallMat.getAt(0, index+2));
                nn.setAtAs(0, 0, 4, smallMat.getAt(0, index+3));

                
                nn.forward();

                float check = nn.getOutput().getAt(0, 5);
                std::cout << check << '\n';
                if(draw)
                {
                    if(static_cast<int>(std::round(check)) == 1)
                    {
                        DrawText("Rectangle", rx, ry, 10, RAYWHITE);
                    }
                    else
                    {                    
                        DrawText("Circle", rx, ry, 10, RAYWHITE);
                    }

                }
            }
        }

        UpdateTexture(prevTexture, prevImage.data);
        DrawTextureEx(prevTexture, Vector2{static_cast<float>(rx), static_cast<float>(ry)}, 0, 15, RAYWHITE);

        const std::string text = "Number of Iterations: " + std::to_string(iterations) + "/" + std::to_string(maxIterations) + " Rate = " + std::to_string(rate) + " Cost = " + std::to_string(costFunction.back());

        DrawText(text.c_str(), 0, 0, sh * 0.04f, RAYWHITE);

        EndDrawing();
    }


    size_t outWidth = 512;
    size_t outHeight = 512;

    unsigned char *outPixels = new unsigned char[sizeof(*outPixels) * outWidth * outHeight * 4];

    if(outPixels == nullptr)
    {
        std::cerr << "Could not allocate memory for the pixels";
    }

    // for (size_t i = 0; i < outHeight; ++i)
    // {
    //     for(size_t j = 0; j < outWidth; ++j)
    //     {
    //         int index = (i * outWidth + j) * 4;
    //         nn.setAtAs(0, 0, 0, static_cast<float>(j)/(outWidth-1));
    //         nn.setAtAs(0, 0, 1, static_cast<float>(i)/(outHeight-1));
    //         nn.forward();
    //         unsigned char R = nn.getOutput().getAt(0, 0) * 255.f;
    //         unsigned char G = nn.getOutput().getAt(0, 1) * 255.f;
    //         unsigned char B = nn.getOutput().getAt(0, 2) * 255.f;
    //         unsigned char A = nn.getOutput().getAt(0, 3) * 255.f;
            
    //         outPixels[index] = R;
    //         outPixels[index + 1] = G;
    //         outPixels[index + 2] = B;
    //         outPixels[index + 3] = A;
    //     }
    // }

    // const std::string path = "img.png";
    
    // if(!stbi_write_png(path.c_str(), outWidth, outHeight, 4, outPixels, 4 * outWidth * sizeof(*outPixels)))
    // {
    //     std::cerr << "Cannot write image\n";
    //     return 1;
    // }
    // std::cout << "Generated Image Sucesfully!\n";
    // delete[] outPixels;
}

void verification()
{
    constexpr int WIDTH = 28, HEIGHT = 28;
    constexpr int SHAPES = 2, SHAPES_NUMBER = 10 * SHAPES;
    int x = getRandomNumber(0, WIDTH), y = getRandomNumber(0, HEIGHT), r = getRandomNumber(5, 50), w = getRandomNumber(0, 50), h = getRandomNumber(0, 50);

    nn::Mat bigMat{SHAPES_NUMBER * 28 * 32 * 4, 7};

    for(size_t k = 0; k < SHAPES_NUMBER; ++k)
    {
        do{
            r = getRandomNumber(5, 50);
            x = getRandomNumber(0, WIDTH);
            y = getRandomNumber(0, HEIGHT);
        }while(x + r > WIDTH || x - r < 0 || y + r > HEIGHT || y - r < 0);

        int distX = x - r, distY = y - r, pDistX = x + r, pDistY = y + r;
        for(int i = 0; i < 28; ++i)
        {
            for(int j = 0; j < 32; ++j)
            {
                int index = (i * 32 + j) * 4;
            
                bigMat.setAt(index, 0, static_cast<float>(j)/(WIDTH - 1));
                bigMat.setAt(index, 1, static_cast<float>(i)/(HEIGHT - 1));

                if(i > distX && i < pDistX && j > distY && j < pDistY)
                {
                    bigMat.setAt(index, 2, 1);
                    bigMat.setAt(index, 3, 1);
                    bigMat.setAt(index, 4, 1);
                    bigMat.setAt(index, 5, 1);
                }
                else
                {
                    bigMat.setAt(index, 2, 0);
                    bigMat.setAt(index, 3, 0);
                    bigMat.setAt(index, 4, 0);
                    bigMat.setAt(index, 5, 0);
                }
                bigMat.setAt(index, 6, static_cast<int>(shapes::circle));
            }
        }

        do{
            w = getRandomNumber(10, 100);
            h = getRandomNumber(10, 100);
            x = getRandomNumber(0, WIDTH);
            y = getRandomNumber(0, HEIGHT);
        }while(x + w > WIDTH || x - w < 0 || y + h > HEIGHT || y - h < 0);

        distX = x - w, distY = y - h, pDistX = x + w, pDistY = y + h;


        for(int i = 0; i < 28; ++i)
        {
            for(int j = 0; j < 32; ++j)
            {
                int index = (i * 32 + j) * 4;
            
                bigMat.setAt(index, 0, static_cast<float>(j)/(WIDTH - 1));
                bigMat.setAt(index, 1, static_cast<float>(i)/(HEIGHT - 1));

                if(i > distX && i < pDistX && j > distY && j < pDistY)
                {
                    bigMat.setAt(index, 2, 1);
                    bigMat.setAt(index, 3, 1);
                    bigMat.setAt(index, 4, 1);
                    bigMat.setAt(index, 5, 1);
                }
                else
                {
                    bigMat.setAt(index, 2, 0);
                    bigMat.setAt(index, 3, 0);
                    bigMat.setAt(index, 4, 0);
                    bigMat.setAt(index, 5, 0);
                }

                bigMat.setAt(index, 6, static_cast<int>(shapes::rectangle));
            }
        }
    }
   
}