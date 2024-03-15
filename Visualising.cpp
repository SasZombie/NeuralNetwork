#include "NeuralNetwork.hpp"
#include "raylib.h"
#include <string>

constexpr int WIDTH = 1280, HEIGHT = 720;

void render(nn::NN nn)
{

    Color color{0x18, 0x18, 0x18, 0xFF};
    Color lowColor{0xFF, 0x00, 0xFF, 0xFF};
    Color highColor{0xFF, 0xFF, 0x00, 0xFF};

    ClearBackground(color);

    size_t archCount = nn.getCount() + 1;

    int radios = 25;
    int layerBorderVPad = 50;
    int layerBorderHPad = 50;
    int nnWidtdh = WIDTH - 2*layerBorderHPad;
    int nnHeight = HEIGHT - 2*layerBorderVPad;
    int nnX = WIDTH/2 - nnWidtdh/2;
    int nnY = HEIGHT/2 - nnHeight/2;
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



float td[] = {
    0, 0, 0,
    1, 0, 0,
    2, 0, 0,
    3, 0, 1,
    0, 1, 0,
    0, 2, 1
};


int main()
{

    size_t arch[] = {2, 2, 1};

    size_t stride = sizeof(arch)/sizeof(arch[0]);

    size_t n = sizeof(td)/sizeof(td[0])/stride;
    
    nn::NN nn(arch, stride);
    nn::NN grad(arch, stride);

    nn::Mat ti(6, 2, stride, td);

    nn::Mat to(6, 1, stride, td+2);

    nn.rand();

    float rate = 1;

    InitWindow(WIDTH, HEIGHT, "Neural Network Visualiser");
    SetTargetFPS(60);
    
    int i = 0;
    while (!WindowShouldClose())
    {  
        if(i < 5000)
        {
            nn.backProp(grad, ti, to);   
            nn.learn(grad, rate);
            ++i;
        }

        BeginDrawing();
        render(nn);
        std::cout << i << ' ' << nn.cost(ti, to) << '\n';
        EndDrawing();
    }
    
    for (size_t i = 0; i <= 3; ++i)
    {
        for(size_t j = 0; j <= 3; ++j)
        {
            nn.setAtAs(0, 0, 0, i);
            nn.setAtAs(0, 0, 1, j);
            nn.forward();
            std::cout << i << " ^ " << j << " => " << nn.getOutput().getAt(0, 0) << '\n';
        }
    }
}