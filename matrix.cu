#include "matrix.hpp"

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>


void Matrix::fill_random() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            data_[i * cols_ + j] = dis(gen);
        }
    }
}

Matrix Matrix::naive_matmul(const Matrix& other) {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
    }
    Matrix result(rows_, other.cols_);
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < other.cols_; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < cols_; ++k) {
                sum += data_[i * cols_ + k] * other[k * other.cols_ + j];
            }
            result[i * result.cols_ + j] = sum;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    // Log the time taken for the multiplication in nanoseconds
    std::cout << "Naive matrix multiplication took " << elapsed.count() * 1e9 << " nanoseconds" << std::endl;
    return result;
}

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t M, size_t N, size_t K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (size_t k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

Matrix Matrix::cuda_matmul(const Matrix& other) {
    Matrix result(rows_, other.cols_);
    dim3 blockSize(16, 16);
    dim3 gridSize((other.cols_ + blockSize.x - 1) / blockSize.x, (rows_ + blockSize.y - 1) / blockSize.y);
    std::cout << "Launching CUDA kernel with grid size (" << gridSize.x << ", " << gridSize.y << ") and block size (" << blockSize.x << ", " << blockSize.y << ")" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    matmul_kernel<<<gridSize, blockSize>>>(data_, other.data_, result.data_, rows_, other.cols_, cols_);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    // Log the time taken for the multiplication in nanoseconds
    std::cout << "CUDA matrix multiplication took " << elapsed.count() * 1e9 << " nanoseconds" << std::endl;
    return result;
}