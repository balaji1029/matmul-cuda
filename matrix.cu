#include "matrix.hpp"

#include <iostream>
#include <chrono>


void Matrix::fill_random() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            data_[i][j] = dis(gen);
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
                sum += data_[i][k] * other[k][j];
            }
            result[i][j] = sum;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    // Log the time taken for the multiplication
    std::cout << "Time taken for naive matrix multiplication: " << elapsed.count() << " seconds" << std::endl;
    return result;
}