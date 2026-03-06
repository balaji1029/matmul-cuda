#include <iostream>

#include "matrix.hpp"

int main() {
    // std::cout << "Creating matrices A and B..." << std::endl;
    Matrix A(200, 200);
    Matrix B(200, 200);
    // std::cout << "Matrix A: " << A.rows() << "x" << A.cols() << std::endl;
    Matrix C = A.naive_matmul(B);
    Matrix D = A.cuda_matmul(B);
    return 0;
}