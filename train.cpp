#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <raylib.h>
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
    
    nn::NN nn(arch.data(), arch.size()); 
    nn::NN grad(arch.data(), arch.size());
    float rate = 1;

    nn.rand();
    
    file.close();
    
    std::ifstream inFile(argv[2]);

     if(!inFile.is_open())
    {
        std::cerr << "Unable to open file:" << argv[2];
        return 1;
    }

    nn::Mat bigMat;

    bigMat.load(inFile);
    inFile.close();


    size_t rows = bigMat.getRows();
    size_t count = nn.getCount();
    size_t inSz = arch[0];
    size_t outSz = arch[count];

    if(nn.getCount() < 1 || bigMat.getCols() != inSz + outSz)
    {
        std::cerr << "Incorrect format files\n";
        return 1;
    }
    

    const float *data = bigMat.getData();
    bigMat.print();

    nn::Mat inMat(rows, inSz, bigMat.getStride(), data);  
    nn::Mat outMat(rows, outSz, bigMat.getStride(),  data+2);


    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(WIDTH, HEIGHT, "Neural Network Visualiser");
    SetTargetFPS(60);
    
    bool pause = true;

    int iterations = 0;
    constexpr int maxIterations = 10000;


    float cost = 0.f;
    std::vector<float> costFunction;
    while (!WindowShouldClose())
    {  

        if(IsKeyPressed(KEY_R))
        {
            iterations = 0;
            costFunction.clear();
            cost = 0.f;
            nn.clear();
            nn.rand();
            pause = false;
        }
           
        Color color{0x18, 0x18, 0x18, 0xFF};

        BeginDrawing();

        if(!pause)
        {
            ClearBackground(color);

            for(size_t i = 0; i < 10 && iterations < maxIterations; ++i)
            {

                if(iterations < maxIterations)
                {
                    nn.backProp(grad, inMat, outMat);   
                    nn.learn(grad, rate);

                    cost = nn.cost(inMat, outMat);
                    costFunction.push_back(cost);

                    ++iterations;
                }
            }            
        }

        int rw, rh, rx, ry;
        int sw = GetRenderWidth();
        int sh = GetRenderHeight();
        
        rw = sw/2;
        rh = sh*2/3;
        rx = sw - rw;
        ry = sh/2 - rh/2;
        // render(nn, rx, ry, rw, rh);

        rw = sw/2;
        rh = sh*2/3;
        rx = 3;
        ry = sh/2 - rh/2;

        renderPlot(costFunction, rx, ry, rw, rh);


        const std::string text = "Number of Iterations: " + std::to_string(iterations) + "/" + std::to_string(maxIterations) + " Rate = " + std::to_string(rate) + " Current cost = " + std::to_string(cost);

        DrawText(text.c_str(), 0, 0, sh * 0.04f, RAYWHITE);


        if(IsKeyPressed(KEY_SPACE))
        {
            pause = !pause;
        }

        EndDrawing();
    }


    for (size_t i = 0; i < 2; ++i)
    {
        for(size_t j = 0; j < 2; ++j)
        {
            nn.setAtAs(0, 0, 0, i);
            nn.setAtAs(0, 0, 1, j);
            nn.forward();
            std::cout << i << " ^ " << j << " => " << nn.getOutput().getAt(0, 0) << '\n';
        }
    }

}