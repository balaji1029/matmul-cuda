#include <iostream>

#include "matrix.hpp"

int main() {
    Matrix A(200, 200);
    Matrix B(200, 200);
    Matrix C = A.naive_matmul(B);
    Matrix D = A.cuda_matmul(B);
    return 0;
}