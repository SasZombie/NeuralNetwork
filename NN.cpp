
#include "NN.hpp"

Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;

    m.es = static_cast<float*>(malloc(sizeof(*m.es) * rows * cols));

    assert(m.es != nullptr);
    
    return m;    
}

float getRandom()
{
    std::random_device rd;

    std::default_random_engine engine(rd());

    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    return distribution(engine);
}



void mat_rand(Mat m, float low, float max)
{
    for (size_t i = 0; i < m.rows; ++i)
    {
        for (size_t j = 0; j < m.cols; ++j)
        {
          mat_at(m, i, j) = getRandom() * (max - low) + low ;
        }
        
    }
}

void mat_fill(Mat m, float x)
{
    for (size_t i = 0; i < m.rows; ++i)
    {
        for (size_t j = 0; j < m.cols; ++j)
        {
            mat_at(m, i, j) = x;
        }        
    }

    
}

void mat_dot(Mat dest, Mat a, Mat b)
{
    assert(a.cols == b.cols);
    size_t n = a.cols;
    assert(dest.cols == b.cols);
    assert(dest.rows == a.rows);
   
    for (size_t i = 0; i < dest.rows; ++i)
    {
        for (size_t j = 0; j < dest.cols; ++j)
        {
            for (size_t k = 0; k < n; ++k)
            {
                mat_at(dest, i, j ) += mat_at(a, i, k) * mat_at(b, k, j);
            }
        }        
    }    
}

void mat_sum(Mat dest, Mat a)
{
    assert(dest.rows == a.rows);
    assert(dest.cols == dest.cols);
    
    for (size_t i = 0; i < a.rows; ++i)
    {
        for (size_t j = 0; j < a.cols; ++j)
        {
            mat_at(dest, i, j) = mat_at(dest, i, j ) + mat_at(a, i, j);
        }        
    }

}

void mat_print(Mat m)
{
    for (size_t i = 0; i < m.rows; ++i)
    {
        for (size_t j = 0; j < m.cols; ++j)
        {
            std::cout << mat_at(m, i, j) << " ";
        }

        std::cout << '\n';
        
    }
    
}
