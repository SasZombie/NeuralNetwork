#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <raylib.h>
#include <filesystem>
#include "stb_image_write.h"
#include <limits>
#include "NeuralNetwork.hpp"
#include "stb_image.h"

constexpr int WIDTH = 1280, HEIGHT = 720;

int getRandom(size_t min, size_t max) noexcept;
std::tuple<float, float> minMaxPlot(const std::vector<float>&plot);
void renderPlot(const std::vector<float>& cost, int x, int y, int w, int h);
float numberRet(const std::string &name, nn::NN &nn) noexcept;
bool endsWith(const std::string& fullString, const std::string& ending);


int getRandom(size_t min = 0, size_t max = 1) noexcept
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<size_t> distribution(min, max);

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

std::filesystem::path getImgPath(size_t nth, const std::string& path)
{

    size_t index = 0;
    for (const auto& entry : std::filesystem::directory_iterator(path))
    {
        if(index == nth)
            return entry;

        ++index; 
    }
    
    return std::string("Index is too big");
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

float numberRet(const std::string &name, nn::NN &nn) noexcept
{

    int imgWidth, imgHeight, imgComp;
    
    uint8_t *data = reinterpret_cast<uint8_t*>(stbi_load(name.c_str(), &imgWidth, &imgHeight, &imgComp, 0));

    for (size_t i = 0; i < 28; ++i)
    {
        for(size_t j = 0; j < 28; ++j)
        {
            size_t index = i * 28 + j;

            nn.setAtAs(0, 0, index, data[index]);
        }
    }

    nn.forward();
    
    float max = 0, ret = 0;
    size_t outPuts = nn.getOutput().getCols();
    for(size_t i = 0; i < outPuts; ++i)
    {
        float output = nn.getOutput().getAt(0, i);
        std::cout << output << '\n';
        if(output > max)
        {
            ret = i;
            max = output;
        }
    }

    std::cout << "Output inside = " << max << ' ' << ret <<'\n';
    return ret;
}

bool endsWith(const std::string& fullString, const std::string& ending) {
    return fullString.size() >= ending.size() &&
           fullString.compare(fullString.size() - ending.size(), ending.size(), ending) == 0;
}

void vectorPaths(std::vector<std::string> &paths)
{
    for (const auto& entry : std::filesystem::directory_iterator("MinstConvert/MinstMixed/"))
    {
        paths.push_back(entry.path()); 
    }
    
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
    size_t count = nn.getCount();
    size_t cols = bigMat.getCols();
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
    constexpr int maxIterations = 1000 * 1000;

    float cost = 0.f;
    std::vector<float> costFunction = {0.f};
    std::vector<std::string> paths;

    vectorPaths(paths);
    
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

    const std::string imgsNames[] = {
        "imgs/img_1.png", "imgs/img_0.png", "imgs/img_22.png", "imgs/img_7.png", "imgs/4.png", "imgs/1073.png", "imgs/img_21.png", "imgs/img_6.png",
         "imgs/img_20.png", "imgs/img_11.png"
    };

    const std::string imgsTestNames[] = {
        "0/img_9532.png", "1/img_41754.png", "imgs/img_22.png", "imgs/img_7.png", "imgs/4.png", "imgs/1073.png", "imgs/img_21.png", "imgs/img_6.png",
         "imgs/img_20.png", "imgs/img_11.png"
    };
    
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

    const size_t pathsSize = paths.size();

    int number = 0, rounded = 0;
    bool amVerifiying = false;
    Texture veryTex;

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
            grad.clear();
        }

        if(CheckCollisionPointCircle(GetMousePosition(), {200, 200}, 10) && IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
        {
            rate = rate - 1;
        }

        if(CheckCollisionPointCircle(GetMousePosition(), {200, 300}, 10) && IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
        {
            rate = rate + 1;
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

        int rw, rh, rx, ry;
        int sw = GetRenderWidth();
        int sh = GetRenderHeight();
        
        rw = sw/3;
        rh = sh*2/3;
        rx = 3;
        ry = sh/2 - rh/2;
        renderPlot(costFunction, rx, ry, rw, rh);
        rx = rx + rw;
        rx = rx + rw;
        
        DrawCircle(200, 200, 10, RED);
        DrawCircle(200, 300, 10, GREEN);

        if(IsKeyPressed(KEY_F))
        {
            number = getRandom(0, pathsSize);
            std::cout << number << '\n';
            std::string path = paths.at(number);
            float out = numberRet(path , nn);
            rounded = std::round(out);

            amVerifiying = true;
            veryTex = LoadTexture(path.c_str());

        }

        DrawText(std::to_string(rounded).c_str(), 300, 300, 15, RAYWHITE);
        
        if(amVerifiying)
            DrawTextureEx(veryTex, Vector2{static_cast<float>(rx), static_cast<float>(ry)}, 0, 15, RAYWHITE);
        else
            DrawTextureEx(texes[number], Vector2{static_cast<float>(rx), static_cast<float>(ry)}, 0, 15, RAYWHITE);

        const std::string text = "Number of Iterations: " + std::to_string(iterations) + "/" + std::to_string(maxIterations) + " Rate = " + std::to_string(rate) + " Cost = " + std::to_string(costFunction.back());

        DrawText(text.c_str(), 0, 0, sh * 0.04f, RAYWHITE);

        EndDrawing();
    }
}