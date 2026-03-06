#include <iostream>

#include "matrix.hpp"

int main() {
    // std::cout << "Creating matrices A and B..." << std::endl;
    Matrix A(1000, 1000);
    Matrix B(1000, 1000);
    // std::cout << "Matrix A: " << A.rows() << "x" << A.cols() << std::endl;
    Matrix C = A.naive_matmul(B);
    Matrix D = A.cuda_matmul(B);
    Matrix E = A.uncoalesced_cuda_matmul(B);
    return 0;
}