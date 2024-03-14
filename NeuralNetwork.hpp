#pragma once

#include <iostream>
#include <algorithm>
#include <assert.h>
#include <random>
#include <memory>
#include <fstream>

#define mat_at(m, i, j)  (m)->es[(i) * (m)->stride + (j)]
#define mat_at_non(m, i, j)  (m).es[(i) * (m).stride + (j)]

namespace nn
{
    float sig(const float x) noexcept;
    
    class Mat
    {
    private:
        size_t rows, cols, stride;
        std::vector<float> es;
        float getRandom() const noexcept;

    public:
        bool empty;
        Mat() = default;
        Mat(size_t n_rows, size_t n_cols)
            : rows(n_rows), cols(n_cols), stride(n_cols)
        {
            empty = true;
            es.resize(rows * stride);
        }
        Mat(size_t n_rows, size_t n_cols, size_t n_stride)
            : rows(n_rows), cols(n_cols), stride(n_stride)
        {
            empty = true;
            es.resize(rows * stride);
        }

        Mat(size_t n_rows, size_t n_cols, const float *n_es)
            : rows(n_rows), cols(n_cols), stride(n_cols)
        {
            empty = false;
            es.resize(rows * stride);
            for(size_t i = 0; i < rows; ++i)
            {
                for(size_t j = 0; j < cols; ++j)
                {
                    mat_at(this, i, j) = n_es[i * stride + j];
                }
            };
        }

        Mat(size_t n_rows, size_t n_cols, size_t n_stride, const float *n_es)
            : rows(n_rows), cols(n_cols), stride(n_stride)
        {
            empty = false;
            es.resize(rows * stride);
            for(size_t i = 0; i < rows; ++i)
            {
                for(size_t j = 0; j < cols; ++j)
                {
                    mat_at(this, i, j) = n_es[i * stride + j];
                }
            }
        }


        ~Mat() = default;
        
        Mat matRow(size_t row) const;
        size_t getRows() const noexcept;
        size_t getCols() const noexcept;
        size_t getStride() const noexcept;
        const float* getData() const noexcept;
        float getAt(const size_t i, const size_t j) const noexcept;
        float getAt(const size_t i) noexcept;

        void clear() noexcept; 
        void sum(const Mat a);
        void shuffle() noexcept;
        void setEs(const float *n_es);
        void apply_sigmoid() noexcept;
        void fill(const float x) noexcept;
        void append(const Mat& m) noexcept;
        void dot(const Mat a, const Mat b);
        void load(std::ifstream &file) noexcept;
        void save(std::ofstream &file) const noexcept;
        void alloc(const size_t n_rows, const size_t n_cols);
        void print(const std::string& name = "") const noexcept; 
        void rand(const float low = 0, const float max = 1) noexcept;
        void setAt(const size_t i, const size_t j, const float number) noexcept;
        void setAt(const size_t i, const float number) noexcept;

        nn::Mat slice(size_t row, size_t i, size_t cols) noexcept;


        void alloc(const size_t n_rows, const size_t n_cols, const size_t n_stride);

        Mat operator+(const Mat& other) const noexcept;
    
    };    


    class NN
    {
    private:

        size_t count;
        std::vector<nn::Mat> ws;
        std::vector<nn::Mat> bs;
        std::vector<nn::Mat> as;//Count + 1 

    public:

        NN() = default;
        NN(size_t *arch, size_t arch_count);
        void clear() noexcept;
        void print() const noexcept;
        void alloc(size_t *arch, size_t arch_count);
        void rand(const float low = 0, const float max = 1);

        void forward() noexcept;
        void learn(const NN &grad, float rate);
        void backProp(NN &grad, const Mat &ti, const Mat &to);
        void fineDiff(NN &grad, const float eps, const Mat& ti, const Mat& to);
        float cost(const Mat& ti, const Mat& to);

        void save(std::ofstream& path) const noexcept;
        void load(std::ifstream& path) noexcept;

     
        //Getters And Setters >_<

        void setInput(const nn::Mat& m) noexcept;
        void setOutput(const nn::Mat& m) noexcept;

        void setAtWs(const size_t i, const size_t j, const size_t k, const float number) noexcept;
        void setAtBs(const size_t i, const size_t j, const size_t k, const float number) noexcept;
        void setAtAs(const size_t i, const size_t j, const size_t k, const float number) noexcept;
        
        float getBs(size_t i) const noexcept;
        float getWsCols(const size_t i) const noexcept;
        float getAsCols(const size_t i) const noexcept;
        float getAtBs(const size_t i, const size_t j, const size_t k) const noexcept;
        float getAtWs(const size_t i, const size_t j, const size_t k) const noexcept;
        float getAtAs(const size_t i, const size_t j, const size_t k) const noexcept;

        size_t getCount() const noexcept;
        
        nn::Mat& getInput() noexcept;
        nn::Mat& getOutput() noexcept;
        

        ~NN();
    };    
 
    
} // namespace nn

