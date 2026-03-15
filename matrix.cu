#include "matrix.hpp"

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 32
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define ROW 512

void Matrix::fill_random() {
    // std::random_device rd;
    // std::mt19937 gen(42);
    // std::uniform_real_distribution<> dis(0.0, 1.0);
    // std::cout << "Filling matrix with random values..." << std::endl;
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            data_.at(i * cols_ + j) = static_cast<float>(rand()) / RAND_MAX; // dis(gen);
        }
    }
    // std::cout << "Copying matrix data to GPU..." << std::endl;
    // cudaMemcpy(device_data_, data_.data(), rows_ * cols_ * sizeof(float), cudaMemcpyHostToDevice);
    // std::cout << "Matrix filled with random values." << std::endl;
    copy_to_device();
}

Matrix Matrix::naive_matmul(const Matrix& other) {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
    }
    std::cout << "Performing naive matrix multiplication..." << std::endl;
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

// __global__void tiling_matmul_kernel(const float* A, const float* B, float* C, size_t M, size_t N, size_t K) {
//     __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
//     __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

//     int globalStartRow = blockIdx.y * BLOCK_SIZE;
//     int globalStartCol = blockIdx.x * BLOCK_SIZE;

//     int localRow = threadIdx.y;
//     int localCol = threadIdx.x;


// }

// Matrix Matrix::cuBLAS(const Matrix& other) {
//     if (cols_ != other.rows_) {
//         throw std::invalid_argument("Incompatible matrix dimensions");
//     }

//     Matrix result(rows_, other.cols_);

//     cublasHandle_t handle;
//     cublasCreate(&handle);

//     float alpha = 1.0f;
//     float beta = 0.0f;

//     // C = A * B
//     // cuBLAS expects column-major, so we compute:
//     // C^T = B^T * A^T
//     auto start = std::chrono::high_resolution_clock::now();
//     cublasSgemm(
//         handle,
//         CUBLAS_OP_N, CUBLAS_OP_N,
//         rows_,                 // m
//         other.cols_,           // n
//         cols_,                 // k
//         &alpha,
//         device_data_,          // A
//         rows_,
//         other.device_data_,    // B
//         cols_,
//         &beta,
//         result.device_data_,   // C
//         rows_
//     );
//     cudaDeviceSynchronize();
//     auto end = std::chrono::high_resolution_clock::now();
//     result.copy_to_host();
//     cublasDestroy(handle);
//     std::chrono::duration<double> elapsed = end - start;

//     std::cout << "cuBLAS matrix multiplication took " << elapsed.count() * 1e9 << " nanoseconds" << std::endl;
//     return result;
// }

Matrix Matrix::cuBLAS(const Matrix& other) {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Incompatible matrix dimensions");
    }

    Matrix result(rows_, other.cols_);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    auto start = std::chrono::high_resolution_clock::now();

    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        other.cols_,
        rows_,
        cols_,
        &alpha,
        other.device_data_, other.cols_,
        device_data_, cols_,
        &beta,
        result.device_data_, other.cols_
    );

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();

    result.copy_to_host();

    cublasDestroy(handle);

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "cuBLAS matrix multiplication took "
        << elapsed.count() * 1e9 << " nanoseconds" << std::endl;

    return result;
}

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t M, size_t N, size_t K) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x < M && y < N) {
        float sum = 0.0f;
        for (size_t k = 0; k < K; ++k) {
            sum += A[x * K + k] * B[k * N + y];
        }
        C[x * N + y] = sum;
    }
}

__global__ void another_matmul_kernel(const float* A, const float* B, float* C, size_t M, size_t N, size_t K) {
    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    int x = (blockIdx.x * BLOCK_SIZE) + (tid % BLOCK_SIZE);
    int y = (blockIdx.y * BLOCK_SIZE) + (tid / BLOCK_SIZE);
    if (x < M && y < N) {
        float sum = 0.0f;
        for (size_t k = 0; k < K; ++k) {
            sum += A[x * K + k] * B[k * N + y];
        }
        C[x * N + y] = sum;
    }
}

__global__ void uncoalesced_matmul_kernel(const float* A, const float* B, float* C, size_t M, size_t N, size_t K) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < M && y < N) {
        float sum = 0.0f;
        for (size_t k = 0; k < K; ++k) {
            sum += A[x * K + k] * B[k * N + y];
        }
        C[x * N + y] = sum;
    }
}

__global__ void tiling_matmul_row_based_kernel(const float* A, const float* B, float* C, size_t M, size_t N, size_t K) {
    __shared__ float tileA[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE * CEIL_DIV(ROW, BLOCK_SIZE) * BLOCK_SIZE];
    int localX = threadIdx.x;
    int localY = threadIdx.y;

    int globalX = blockDim.x * blockIdx.x + localX;
    int globalY = blockDim.y * blockIdx.y + localY;

    int numTiles = CEIL_DIV(K, BLOCK_SIZE);

    float ans[CEIL_DIV(ROW, BLOCK_SIZE)] = {0.0f};

    for (int i = 0; i < numTiles; i++) {
        int tileAx = i * BLOCK_SIZE + localX;
        int tileAy = globalY;

        tileA[localY * BLOCK_SIZE + localX] = (tileAx < M && tileAy < K) ? A[tileAy * K + tileAx] : 0.0f;

        // int tileBx = globalX;
        // int tileBy = i * BLOCK_SIZE + localY;
        int tileByt = i * BLOCK_SIZE + localY;

        for (int t = 0; t < numTiles - 1; t++) {
            int tileBxt = t * BLOCK_SIZE + localX;
            tileB[localY * numTiles * BLOCK_SIZE + t * BLOCK_SIZE + localX] = B[tileByt * N + tileBxt];
        }

        int tileBxt = (numTiles - 1) * BLOCK_SIZE + localX;
        tileB[localY * numTiles * BLOCK_SIZE + (numTiles - 1) * BLOCK_SIZE + localX] = (tileBxt < K && tileByt < N) ? B[tileByt * N + tileBxt] : 0.0f;

        __syncthreads();

        for (int j = 0; j < BLOCK_SIZE; j++) {
            for (int t = 0; t < numTiles; t++) {
                ans[t] += tileA[localY * BLOCK_SIZE + j] * tileB[j * BLOCK_SIZE + (localX + t * BLOCK_SIZE)];
            }
        }

        __syncthreads();
    }

    if (globalX < M && globalY < N)
        for (int t = 0; t < numTiles; t++)
            C[globalY * N + globalX] += ans[t];
}

__global__ void tiling_matmul_kernel(const float* A, const float* B, float* C, size_t M, size_t N, size_t K) {
    __shared__ float tileA[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE * BLOCK_SIZE];
    int localX = threadIdx.x;
    int localY = threadIdx.y;

    int globalX = blockDim.x * blockIdx.x + localX;
    int globalY = blockDim.y * blockIdx.y + localY;

    int numTiles = CEIL_DIV(K, BLOCK_SIZE);

    float ans = 0.0f;

    for (int i = 0; i < numTiles; i++) {
        int tileAx = i * BLOCK_SIZE + localX;
        int tileAy = globalY;

        tileA[localY * BLOCK_SIZE + localX] = (tileAx < M && tileAy < K) ? A[tileAy * K + tileAx] : 0.0f;

        int tileBx = globalX;
        int tileBy = i * BLOCK_SIZE + localY;

        tileB[localY * BLOCK_SIZE + localX] = (tileBx < K && tileBy < N) ? B[tileBy * N + tileBx] : 0.0f;

        __syncthreads();

        for (int j = 0; j < BLOCK_SIZE; j++) {
            ans += tileA[localY * BLOCK_SIZE + j] * tileB[j * BLOCK_SIZE + localX];
        }

        __syncthreads();
    }

    if (globalX < M && globalY < N)
        C[globalY * N + globalX] = ans;
}

Matrix Matrix::uncoalesced_cuda_matmul(const Matrix& other) {
    Matrix result(rows_, other.cols_);
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(CEIL_DIV(rows_, BLOCK_SIZE), CEIL_DIV(other.cols_, BLOCK_SIZE));
    std::cout << "Launching uncoalesced CUDA kernel with grid size (" << gridSize.x << ", " << gridSize.y << ") and block size (" << blockSize.x << ", " << blockSize.y << ")" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    uncoalesced_matmul_kernel << <gridSize, blockSize >> > (device_data_, other.device_data_, result.device_data_, rows_, other.cols_, cols_);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    if (cudaGetLastError() != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
    }
    std::chrono::duration<double> elapsed = end - start;
    result.copy_to_host();
    // Log the time taken for the multiplication in nanoseconds
    std::cout << "Uncoalesced CUDA matrix multiplication took " << elapsed.count() * 1e9 << " nanoseconds" << std::endl;
    return result;
}

Matrix Matrix::cuda_matmul(const Matrix& other) {
    Matrix result(rows_, other.cols_);
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(CEIL_DIV(rows_, BLOCK_SIZE), CEIL_DIV(other.cols_, BLOCK_SIZE));
    std::cout << "Launching CUDA kernel with grid size (" << gridSize.x << ", " << gridSize.y << ") and block size (" << blockSize.x << ", " << blockSize.y << ")" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    matmul_kernel << <gridSize, blockSize >> > (device_data_, other.device_data_, result.device_data_, rows_, other.cols_, cols_);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    std::chrono::duration<double> elapsed = end - start;
    result.copy_to_host();
    // Log the time taken for the multiplication in nanoseconds
    std::cout << "CUDA matrix multiplication took " << elapsed.count() * 1e9 << " nanoseconds" << std::endl;
    return result;
}

Matrix Matrix::another_matmul(const Matrix& other) {
    Matrix result(rows_, other.cols_);
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(CEIL_DIV(rows_, BLOCK_SIZE), CEIL_DIV(other.cols_, BLOCK_SIZE));
    std::cout << "Launching another CUDA kernel with grid size (" << gridSize.x << ", " << gridSize.y << ") and block size (" << blockSize.x << ", " << blockSize.y << ")" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    another_matmul_kernel << <gridSize, blockSize >> > (device_data_, other.device_data_, result.device_data_, rows_, other.cols_, cols_);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    std::chrono::duration<double> elapsed = end - start;
    result.copy_to_host();
    // Log the time taken for the multiplication in nanoseconds
    std::cout << "Another CUDA matrix multiplication took " << elapsed.count() * 1e9 << " nanoseconds" << std::endl;
    return result;
}

Matrix Matrix::tiling_matmul(const Matrix& other) {
    Matrix result(rows_, other.cols_);
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(CEIL_DIV(rows_, BLOCK_SIZE), CEIL_DIV(other.cols_, BLOCK_SIZE));
    std::cout << "Launching tiling CUDA kernel with grid size (" << gridSize.x << ", " << gridSize.y << ") and block size (" << blockSize.x << ", " << blockSize.y << ")" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    tiling_matmul_kernel << < gridSize, blockSize >> > (device_data_, other.device_data_, result.device_data_, rows_, other.cols(), cols_);

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    std::chrono::duration<double> elapsed = end - start;
    result.copy_to_host();
    // Log the time taken for the multiplication in nanoseconds
    std::cout << "Tiling CUDA matrix multiplication took " << elapsed.count() * 1e9 << " nanoseconds" << std::endl;
    result.copy_to_host();
    return result;
}

Matrix Matrix::tiling_matmul_row_based(const Matrix& other) {
    Matrix result(rows_, other.cols_);
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(CEIL_DIV(rows_, BLOCK_SIZE), CEIL_DIV(other.cols_, BLOCK_SIZE));
    std::cout << "Launching tiling CUDA kernel with grid size (" << gridSize.x << ", " << gridSize.y << ") and block size (" << blockSize.x << ", " << blockSize.y << ")" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    tiling_matmul_row_based_kernel << < gridSize, blockSize >> > (device_data_, other.device_data_, result.device_data_, rows_, other.cols(), cols_);

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    std::chrono::duration<double> elapsed = end - start;
    result.copy_to_host();
    // Log the time taken for the multiplication in nanoseconds
    std::cout << "Tiling CUDA matrix multiplication took " << elapsed.count() * 1e9 << " nanoseconds" << std::endl;
    result.copy_to_host();
    return result;
}

Matrix Matrix::transpose(const Matrix& other) {
    Matrix result(other.cols(), other.rows());
    for (size_t i = 0; i < other.rows(); ++i) {
        for (size_t j = 0; j < other.cols(); ++j) {
            result[j * result.cols() + i] = other[i * other.cols() + j];
        }
    }
    return result;
}

bool operator==(const Matrix& X, const Matrix& Y) {
    if (X.rows() != Y.rows() || X.cols() != Y.cols()) {
        return false;
    }
    float max_diff = 0.0f;
    for (size_t i = 0; i < X.rows() * X.cols(); ++i) {
        float diff = std::abs(X[i] - Y[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    std::cout << "Maximum difference between matrices: " << max_diff << std::endl;
    return max_diff < 1e-3f; // Allow for a small numerical tolerance
}

bool operator!=(const Matrix& X, const Matrix& Y) {
    return !(X == Y);
}