#include <iostream>

#include "matrix.hpp"

int main() {
    Matrix A(1000, 1000);
    Matrix B(1000, 1000);
    Matrix C = A.naive_matmul(B);
    return 0;
}