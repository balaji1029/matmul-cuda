#include <iostream>

#include "matrix.hpp"

int main() {
    // std::cout << "Creating matrices A and B..." << std::endl;
    Matrix A(1000, 1000);
    Matrix B(1000, 1000);
    // std::cout << "Matrix A: " << A.rows() << "x" << A.cols() << std::endl;
    Matrix C = A.naive_matmul(B);
    Matrix D = A.another_matmul(B);
    if (D != C) {
        std::cout << "D not equal" << std::endl;
    }
    Matrix E = A.cuda_matmul(B);
    if (D != C) {
        std::cout << "E not equal" << std::endl;
    }
    Matrix F = A.uncoalesced_cuda_matmul(B);
    if (F != C) {
        std::cout << "F not equal" << std::endl;
    }
    return 0;
}