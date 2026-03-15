#include <iostream>

#include "matrix.hpp"

int main(int argc, char** argv) {
    // std::cout << "Creating matrices A and B..." << std::endl;
    int M, N, K;
    if (argc == 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    } else {
        M = N = K = 1000;
    }
    Matrix A(M, K);
    Matrix B(K, N);
    // std::cout << "Matrix A: " << A.rows() << "x" << A.cols() << std::endl;
    // Matrix C = A.naive_matmul(B);
    Matrix D = A.another_matmul(B);
    // if (D != C) {
    //     std::cout << "D not equal" << std::endl;
    // }
    Matrix E = A.cuda_matmul(B);
    if (E != D) {
        std::cout << "E not equal" << std::endl;
    }
    Matrix F = A.uncoalesced_cuda_matmul(B);
    if (F != D) {
        std::cout << "F not equal" << std::endl;
    }
    Matrix G = A.cuBLAS(B);
    if (G != D) {
        std::cout << "G not equal" << std::endl;
    }
    Matrix H = A.tiling_matmul(B);
    if (H != D) {
        std::cout << "H not equal" << std::endl;
    }
    Matrix I = A.tiling_matmul_row_based(B);
    if (I != D) {
        std::cout << "I not equal" << std::endl;
    }
    return 0;
}