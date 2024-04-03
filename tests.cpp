#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <raylib.h>
#include <chrono>
#include "stb_image_write.h"
#include <limits>
#include "NeuralNetwork.hpp"

constexpr int WIDTH = 1280, HEIGHT = 720;

void render(nn::NN& nn, int x, int y, int w, int h);
std::tuple<float, float> minMaxPlot(const std::vector<float>&plot);
void renderPlot(const std::vector<float>& cost, int x, int y, int w, int h);
bool endsWith(const std::string& fullString, const std::string& ending);
int iterations = 0;

size_t stride;
size_t rows;
size_t count;
size_t inSz;
size_t outSz;
size_t batchSize;
size_t batchPerFrame;
size_t batchBegin;
size_t batchCount;

std::vector<float> costFunction = {0.f};

float rate = 0.5f;

void render(nn::NN& nn, int x, int y, int w, int h)
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
    
    nn::NN nn(arch.data(), arch.size(), nn::Flags::shuffle); 
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


    stride = bigMat.getStride();
    rows = bigMat.getRows();
    count = nn.getCount();
    inSz = arch[0];
    outSz = arch[count];
    batchSize = 28;
    batchPerFrame = 200;
    batchBegin = 0;
    batchCount = (rows + batchSize - 1)/batchSize;

    if(nn.getCount() < 1 || bigMat.getCols() != inSz + outSz)
    {
        std::cerr << "Incorrect format files\n";
        return 1;
    }

    bool pause = true;

    constexpr int maxIterations = 100 * 1000;

    float cost = 0.f;
  
    
    Color color{0x18, 0x18, 0x18, 0xFF};
        
    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(WIDTH, HEIGHT, "Neural Network Visualiser");
    SetTargetFPS(60);
    Image prevImage = GenImageColor(28, 28, BLACK);
    Texture2D prevTexture = LoadTextureFromImage(prevImage);


    nn::Batch batch(arch, bigMat);

    auto start = std::chrono::high_resolution_clock::now();
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

        if(!pause)
            costFunction.push_back(nn.autoLearn(grad, bigMat, batch, rate)/batch.getBatchCount());

        if(batch.isFinished() && !pause)
        {
            ++iterations;
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
            for(size_t j = 0; j < 28; ++j)
            {
                nn.setAtAs(0, 0, 0, static_cast<float>(j)/(27));
                nn.setAtAs(0, 0, 1, static_cast<float>(i)/(27));
                nn.forward();
                uint8_t pixel = nn.getOutput().getAt(0, 0) * 255.f;

                ImageDrawPixel(&prevImage, j, i, Color{pixel, pixel, pixel, 255});   
            }
        }

        UpdateTexture(prevTexture, prevImage.data);
        DrawTextureEx(prevTexture, Vector2{static_cast<float>(rx), static_cast<float>(ry)}, 0, 15, RAYWHITE);

        const std::string text = "Number of Iterations: " + std::to_string(iterations) + "/" + std::to_string(maxIterations) + " Rate = " + std::to_string(rate) + " Cost = " + std::to_string(costFunction.back());

        DrawText(text.c_str(), 0, 0, sh * 0.04f, RAYWHITE);

        EndDrawing();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Execution time: " << duration.count() << " microseconds\n";

    size_t outWidth = 512;
    size_t outHeight = 512;

    uint8_t *outPixels = new uint8_t[sizeof(*outPixels) * outWidth * outHeight];

    if(outPixels == nullptr)
    {
        std::cerr << "Could not allocate memory for the pixels";
    }

    for (size_t i = 0; i < outHeight; ++i)
    {
        for(size_t j = 0; j < outWidth; ++j)
        {
            nn.setAtAs(0, 0, 0, static_cast<float>(j)/(outWidth-1));
            nn.setAtAs(0, 0, 1, static_cast<float>(i)/(outHeight-1));
            nn.forward();
            uint8_t pixel = nn.getOutput().getAt(0, 0) * 255.f;
            
            outPixels[i * outWidth + j] = pixel;
        }
    }

    const std::string path = "img.png";
    
    if(!stbi_write_png(path.c_str(), outWidth, outHeight, 1, outPixels, outWidth * sizeof(*outPixels)))
    {
        std::cerr << "Cannot write image\n";
        return 1;
    }
    std::cout << "Generated Image Sucesfully!\n";
    delete[] outPixels;
}