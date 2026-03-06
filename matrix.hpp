#pragma once

#include <vector>
#include <random>

class Matrix {
    int rows_;
    int cols_;
    std::vector<std::vector<float>> data_;
    void fill_random();
public:
    Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols), data_(rows, std::vector<float>(cols)) {
        fill_random();
    }
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    const std::vector<float>& operator[](size_t index) const { return data_[index]; }
    std::vector<float>& operator[](size_t index) { return data_[index]; }
    Matrix naive_matmul(const Matrix& other);
    Matrix cuda_matmul(const Matrix& other);
};
