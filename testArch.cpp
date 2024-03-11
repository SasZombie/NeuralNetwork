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

int getRandom()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distribution(0, 9);

    int randomNumber = distribution(gen);

    return randomNumber;

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

constexpr size_t test = 700, rows = 1;

float numberRet(const Image& img, nn::NN &nn)
{

    float *data = (float*)img.data;

    for (size_t i = 0; i < rows ; ++i)
    {
        for(size_t j = 0; j < test - 2; ++j)
        {
            // size_t index = i * 28 + j;
        
            nn.setAtAs(0, 0, j, j);
        }
    }
    nn.forward();
    
    for(size_t i = 0; i < 2; ++i)
    {
        float output = nn.getOutput().getAt(0, i);
        std::cout << "All outputs are: " << output << '\n';
        
    }


    return 0;
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

    std::cout << bigMat.getCols() * bigMat.getRows() << '\n';


    bigMat.shuffle();

    float rate = 0.5f;

    size_t stride = bigMat.getStride();
    size_t rows = bigMat.getRows();
    size_t count = nn.getCount();
    size_t cols = bigMat.getCols();
    size_t inSz = arch[0];
    size_t outSz = arch[count];
    std::cout << inSz << ' ' << outSz << '\n';
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


    Image zero = LoadImage("imgs/img_1.png");
    Image one = LoadImage("imgs/img_0.png");
    Image two = LoadImage("imgs/img_22.png");
    Image three = LoadImage("imgs/img_7.png");
    Image four = LoadImage("imgs/4.png");
    Image five = LoadImage("imgs/1073.png");
    Image six = LoadImage("imgs/img_21.png");
    Image seven = LoadImage("imgs/img_6.png");
    Image eight = LoadImage("imgs/img_20.png");
    Image nine = LoadImage("imgs/img_11.png");

    const Image images[10] = {
        zero, one, two, three, four, five, six, seven, eight, nine
    };
    
    bool isFor = false;
    bool over = false;
    Image prevImage = GenImageColor(28, 28, BLACK);
    Texture2D prevTexture = LoadTextureFromImage(prevImage);

    Texture2D t0 = LoadTextureFromImage(zero);
    Texture2D t1 = LoadTextureFromImage(one);
    Texture2D t2 = LoadTextureFromImage(two);
    Texture2D t3 = LoadTextureFromImage(three);
    Texture2D t4 = LoadTextureFromImage(four);
    Texture2D t5 = LoadTextureFromImage(five);
    Texture2D t6 = LoadTextureFromImage(six);
    Texture2D t7 = LoadTextureFromImage(seven);
    Texture2D t8 = LoadTextureFromImage(eight);
    Texture2D t9 = LoadTextureFromImage(nine);
    
    const Texture2D texes[] ={
        t0, t1, t2, t3, t4, t5, t6, t7, t8, t9
    };

    int number = 0;


    while (!WindowShouldClose())
    {

        BeginDrawing();
        ClearBackground(color);
        
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


        // if(IsKeyPressed(KEY_FIVE))
        // {
        //     float *data = (float*)five.data;
        //     for (size_t i = 0; i < 28; ++i)
        //     {
        //         for(size_t j = 0; j < 28; ++j)
        //         {
        //             size_t index = i * 28 + j;
        //             nn.setAtAs(0, 0, index, data[index]/255.f);


        //         }
        //     }

        //     nn.forward();

        //     float output = nn.getOutput().getAt(0, 0);
            
        //     if(std::round(output) == 0)
        //         isFor = false;
        //     else
        //         isFor = true;

        //     over = 1;

        // }


        // if(IsKeyPressed(KEY_FOUR))
        // {
        //     float *data = (float*)four.data;
        //     for (size_t i = 0; i < 28; ++i)
        //     {
        //         for(size_t j = 0; j < 28; ++j)
        //         {
        //             size_t index = i * 28 + j;
        //             nn.setAtAs(0, 0, index, data[index]/255.f);
        //         }
        //     }

        //     nn.forward();

        //     float output = nn.getOutput().getAt(0, 0);
            
        //     if(std::round(output) == 1)
        //         isFor = false;
        //     else
        //         isFor = true;

        //     over = 0;
        // }


        if(IsKeyPressed(KEY_C))
        {
            prevImage = GenImageColor(28, 28, WHITE);
        }
        
        // if(isFor)
        //     DrawText("4", 300, 300, 15, RAYWHITE);
        // else
        //     DrawText("5", 300, 300, 15, RAYWHITE);


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
        
       
        if(IsKeyPressed(KEY_F))
        {
            number = getRandom();
            std::cout << number <<'\n';
            
            float out = numberRet(images[number], nn);
            std::cout << out << '\n';
            int rounded = std::round(out);

            switch (rounded)
            {
            case 0:
                DrawText("0", 300, 300, 15, RAYWHITE);
                break;
            case 1:
                DrawText("1", 300, 300, 15, RAYWHITE);
                break;
            case 2:
                DrawText("2", 300, 300, 15, RAYWHITE);              
                break;
            case 3:
                DrawText("3", 300, 300, 15, RAYWHITE);              
                break;
            case 4:
                DrawText("4", 300, 300, 15, RAYWHITE);              
                break;
            case 5:
                DrawText("5", 300, 300, 15, RAYWHITE);              
                break;
            case 6:
                DrawText("6", 300, 300, 15, RAYWHITE);              
                break;
            case 7:
                DrawText("7", 300, 300, 15, RAYWHITE);              
                break;
            case 8:
                DrawText("8", 300, 300, 15, RAYWHITE);              
                break;
            case 9:
                DrawText("9", 300, 300, 15, RAYWHITE);              
                break;
            default:
                std::cout << rounded << '\n';
                break;
            }
            

        }
        
        DrawTextureEx(texes[number], Vector2{static_cast<float>(rx), static_cast<float>(ry)}, 0, 15, RAYWHITE);


      
        
        // for (size_t i = 0; i < 28; ++i)
        // {
        //     for(size_t j = 0; j < 28; ++j)
        //     {
        //         size_t index = i * 28 + j;


        //         nn.setAtAs(0, 0, index, static_cast<float>(j)/(27));

        //         uint8_t pixel = nn.getOutput().getAt(0, 0) * 255.f;

        //         ImageDrawPixel(&prevImage, j, i, Color{pixel, pixel, pixel, 255});   
        //     }
        // }


        // if(over == 0)
        // {
        //     DrawTextureEx(temp, Vector2{static_cast<float>(rx), static_cast<float>(ry)}, 0, 15, RAYWHITE);
        // }
        // else if(over == 1)
        // {
        //     DrawTextureEx(temp2, Vector2{static_cast<float>(rx), static_cast<float>(ry)}, 0, 15, RAYWHITE);
        // }

        const std::string text = "Number of Iterations: " + std::to_string(iterations) + "/" + std::to_string(maxIterations) + " Rate = " + std::to_string(rate) + " Cost = " + std::to_string(costFunction.back());

        DrawText(text.c_str(), 0, 0, sh * 0.04f, RAYWHITE);

        EndDrawing();
    }
}