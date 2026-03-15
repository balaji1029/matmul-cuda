#pragma once

#include <cuda_runtime.h>

#include <vector>
#include <random>
#include <iostream>

class Matrix {
    int rows_;
    int cols_;
    std::vector<float> data_;
    float* device_data_;
    void fill_random();
public:
    Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols), data_(rows * cols) {
        // data_ = new float[rows_ * cols_];
        cudaMalloc(&device_data_, rows_ * cols_ * sizeof(float));
        fill_random();
    }
    ~Matrix() {
        cudaFree(device_data_);
    }
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    const float& operator[](size_t index) const { return data_[index]; }
    float& operator[](size_t index) { return data_[index]; }
    Matrix naive_matmul(const Matrix& other);
    Matrix uncoalesced_cuda_matmul(const Matrix& other);
    Matrix another_matmul(const Matrix& other);
    Matrix cuda_matmul(const Matrix& other);
    void copy_to_device() {
        cudaMemcpy(device_data_, data_.data(), rows_ * cols_ * sizeof(float), cudaMemcpyHostToDevice);
    }
    void copy_to_host() {
        cudaMemcpy(data_.data(), device_data_, rows_ * cols_ * sizeof(float), cudaMemcpyDeviceToHost);
    }
};
