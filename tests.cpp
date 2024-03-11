#include <iostream>
#include <random>
#include <raylib.h>

float getRandom1() noexcept
{
    std::random_device rd;

    std::default_random_engine engine(rd());

    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    return distribution(engine);
}

int main()
{

    InitWindow(600, 800, "SaS");
    SetTargetFPS(60);

    Image img = LoadImage("imgs/1073.jpg");

    
    Texture2D tex = LoadTextureFromImage(img);

    while (!WindowShouldClose())
    {
        BeginDrawing();
        ClearBackground(BLACK);
        DrawTexture(tex, 40, 40, WHITE);
        EndDrawing();
    }
    
}