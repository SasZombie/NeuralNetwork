#pragma once
#include <iostream>
#include <assert.h>
#include <random>

struct Mat
{
    size_t rows;
    size_t cols;
    float *es;
};


#define mat_at(m, i, j)  (m).es[(i) * (m).cols + (j)]


Mat mat_alloc(size_t rows, size_t cols);
void mat_rand(Mat m, float low = 0, float max = 1);
void mat_fill(Mat m, float x);
void mat_dot(Mat dest, Mat a, Mat b);
void mat_sum(Mat dest, Mat a);
void mat_print(Mat m);
float getRandom();