#include <iostream>
#include <raylib.h>
#include "NeuralNetwork.hpp"

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

int getRandomNumber(int min, int max)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> distribution(min, max);

    return distribution(mt);
}

enum class shapes {
    circle,
    rectangle,
    shapes
};

constexpr int trainingPerSample = 100, verificationSamplesPerShape = trainingPerSample/2;
constexpr size_t WIDTH = 28, HEIGHT = 28;

size_t arch[] = {WIDTH * HEIGHT, 14, 7, 5, static_cast<size_t>(shapes::shapes)};

size_t batch_size = 20;
size_t batches_per_frame = 20;
float rate = 0.1f;
bool paused = true;


void generateRect(std::vector<float> &t, int width, int height) 
{
    t.clear();
    int x = getRandomNumber(0, WIDTH), y = getRandomNumber(0, HEIGHT), r = getRandomNumber(5, 50), w = getRandomNumber(0, 50), h = getRandomNumber(0, 50);

    do{
        w = getRandomNumber(10, 24);
        h = getRandomNumber(10, 24);
        x = getRandomNumber(0, WIDTH);
        y = getRandomNumber(0, HEIGHT);
    }while(x + w > WIDTH || x - w < 0 || y + h > HEIGHT || y - h < 0);

    int distX = x - r, distY = y - r, pDistX = x + r, pDistY = y + r;

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
            t[index + 4] = static_cast<int>(shapes::rectangle);
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
            int index = (i * 32 + j) * 4;
        
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
            t[index + 4] = static_cast<int>(shapes::circle);
        }
    }

}

void vecToMat(nn::Mat row, std::vector<float>&t)
{
    for (size_t i = 0; i < HEIGHT; ++i)
    {
        for (size_t j = 0; j < WIDTH; ++j)
        {
            row.setAt(i * WIDTH + j, t[i * WIDTH + j]/255.f);
        }
    }   


}
int ver = 0;
nn::Mat generateSample(size_t samples)
{
    size_t input_size = WIDTH*HEIGHT;
    size_t output_size = static_cast<int>(shapes::shapes);

    nn::Mat t{samples*static_cast<int>(shapes::shapes), input_size + output_size};
    // size_t s = region_save(r);
    //     Olivec_Canvas oc = {0};
    //     oc.pixels = region_alloc(r, WIDTH*HEIGHT*sizeof(*oc.pixels));
    //     oc.width = WIDTH;
    //     oc.height = HEIGHT;
    //     oc.stride = WIDTH;
    std::vector<float> pixels;
    pixels.resize(WIDTH * HEIGHT * 6);
        for (size_t i = 0; i < samples; ++i) {

            std::cout << ver << '\n';

            ver++;
            // int x, y, w, h;
            // random_boundary(oc.width, oc.height, &x, &y, &w, &h);
            // int r = (w < h ? w : h)/2;
            for (size_t j = 0; j < static_cast<int>(shapes::shapes); ++j)
            {   
                std::cout << j << '\n';
                nn::Mat row = t.matRow(i*2 + j);

                
               nn::Mat in = row.slice(0, 0, input_size);
               nn::Mat out = row.slice(0, input_size, output_size);


                switch (static_cast<shapes>(j)) {
                case shapes::circle: 
                    generateCircle(pixels, 10, 10);
                    break;
                case shapes::rectangle:   
                    generateRect(pixels, 10, 10);
                    break;
                default: assert(0 && "unreachable");
                }

                vecToMat(in, pixels);
                out.fill(0);
                out.setAt(j, 1.0f);
            }
        }

    return t;
}

void display_training_data(nn::Mat &t)
{
    for (size_t i = 0; i < t.getRows(); ++i) {
        nn::Mat row = t.matRow(i);
        nn::Mat in = row.slice(0, 0, WIDTH*HEIGHT);
        nn::Mat out = row.slice(0, WIDTH*HEIGHT, static_cast<int>(shapes::shapes));

        for (size_t y = 0; y < HEIGHT; ++y) {
            for (size_t x = 0; x < WIDTH; ++x) {
                if (in.getAt(y*WIDTH + x) > 1e-6f) {
                    printf("##");
                } else {
                    printf("  ");
                }
            }
            printf("\n");
        }
        out.print();
    }
}

int main()
{
    nn::NN nn{arch, 5};
    nn::NN grad{arch, 5};


    nn.rand(-1, 1);

    nn::Mat t; // =generateSample(trainingPerSample);
    nn::Mat v; // = generateSample(verificationSamplesPerShape);
    
    int factor = 80;
    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(16*factor, 9*factor, "Shape");
    SetTargetFPS(60);

    std::vector<float> pixles;
    pixles.resize(32 * 32);

    

    size_t count = nn.getCount();
    size_t rows = arch[0];
    size_t cols = arch[0] + arch[count];
    size_t inSz = arch[0];
    size_t outSz = arch[count];
    size_t batchSize = 28;
    size_t batchPerFrame = 200;
    size_t batchBegin = 0;
    size_t batchCount = (rows + batchSize - 1)/batchSize;
    bool pause = true;

    int iterations = 0;
    constexpr int maxIterations = 100 * 1000;

    float cost = 0.f;
    std::vector<float> costFunction = {0.f};
    
    Color color{0x18, 0x18, 0x18, 0xFF};
        
    Image prevImage = GenImageColor(28, 28, BLACK);
    Texture2D prevTexture = LoadTextureFromImage(prevImage);
    bool shape = false;
    
    std::vector<float> newImg;

    newImg.resize(100 * 100);

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
            generateCircle(newImg, 10, 10);
        }

        if(IsKeyPressed(KEY_F))
        {
            generateRect(newImg, 10, 10);
        }

        // for(size_t i = 0; i < batchPerFrame && iterations < maxIterations && !pause; ++i)
        // {   
        //     size_t size = batchSize;
        //     if(batchBegin + batchSize >= rows)
        //     {
        //         size = rows - batchBegin;
        //     }

        //     const float *data = t.getData();

        //     nn::Mat inBatchMat{size, inSz, cols, data + (batchBegin * cols + 0)};
        //     nn::Mat outBatchMat{size, outSz, cols, data + (batchBegin * cols + inSz)};

        //     nn.backProp(grad, inBatchMat, outBatchMat);   
        //     nn.learn(grad, rate);

        //     cost = cost + nn.cost(inBatchMat, outBatchMat);
        //     batchBegin = batchBegin + batchSize;

        //     if(batchBegin >= rows)
        //     {
        //         ++iterations;
        //         costFunction.push_back(cost/batchCount);
        //         cost = 0.f;
        //         batchBegin = 0;
        //         t.shuffle();   
        //     }
            
        // }
        
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
        //render(nn, rx, ry, rw, rh);
        rx = rx + rw;

        // vecToMat(nn.getInput() , newImg);
        // nn.forward();

        // if(nn.getOutput().getAt(0) > nn.getOutput().getAt(1))
        // {
        //     DrawText("Circle", rx, ry, 10, RED);
        // }else if(nn.getOutput().getAt(0) < nn.getOutput().getAt(1))
        // {
        //     DrawText("RECT", rx, ry, 10, RED);
        // }

        // for (size_t i = 0; i < 28; ++i)
        // {
        //     for(size_t j = 0; j < 28; ++j)
        //     {
        //         nn.setAtAs(0, 0, 0, static_cast<float>(j)/(27));
        //         nn.setAtAs(0, 0, 1, static_cast<float>(i)/(27));
        //         nn.setAtAs(0, 0, 2, shape);
        //         nn.forward();
        //         unsigned char R = nn.getOutput().getAt(0, 0) * 255.f;
        //         unsigned char G = nn.getOutput().getAt(0, 1) * 255.f;
        //         unsigned char B = nn.getOutput().getAt(0, 2) * 255.f;
        //         unsigned char A = nn.getOutput().getAt(0, 3) * 255.f;
        for (size_t i = 0; i < 28; ++i)
        {
            for(size_t j = 0; j < 28; ++j)
            {
                size_t index = (i * 32 + j) * 4;
                if(newImg[index] == 0)
                    ImageDrawPixel(&prevImage, j, i, Color{20, 20 , 20, 255});   
                else
                    ImageDrawPixel(&prevImage, j, i, Color{255, 255 , 255, 255});   

            }
        }


        //         float check = nn.getOutput().getAt(0, 4);
        //         if(static_cast<int>(std::round(check)) == 1)
        //         {
        //             DrawText("Rectangle", rx, ry, 10, RAYWHITE);
        //         }
        //         else
        //         {                    
        //             DrawText("Circle", rx, ry, 10, RAYWHITE);
        //         }
        //     }
        // }

        UpdateTexture(prevTexture, prevImage.data);
        DrawTextureEx(prevTexture, Vector2{static_cast<float>(rx), static_cast<float>(ry)}, 0, 15, RAYWHITE);

        const std::string text = "Number of Iterations: " + std::to_string(iterations) + "/" + std::to_string(maxIterations) + " Rate = " + std::to_string(rate) + " Cost = " + std::to_string(costFunction.back());

        DrawText(text.c_str(), 0, 0, sh * 0.04f, RAYWHITE);

        EndDrawing();
    }
    
    
}