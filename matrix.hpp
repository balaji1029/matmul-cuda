#pragma once

#include <cuda_runtime.h>

#include <vector>
#include <random>
#include <iostream>

class Matrix {
    int rows_;
    int cols_;
    float* data_;
    void fill_random();
public:
    Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
        cudaMalloc(&data_, rows_ * cols_ * sizeof(float));
        std::cout << "Filling matrix with random values..." << std::endl;
        fill_random();
        std::cout << "Created matrix of size " << rows_ << "x" << cols_ << std::endl;
    }
    ~Matrix() {
        cudaFree(data_);
    }
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    const float& operator[](size_t index) const { return data_[index]; }
    float& operator[](size_t index) { return data_[index]; }
    Matrix naive_matmul(const Matrix& other);
    Matrix cuda_matmul(const Matrix& other);
};
