#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h> // Header for cuBLAS

// Macro for robust CUDA error checking
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " in file " << __FILE__ \
                  << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

// Macro for robust cuBLAS error checking
#define CHECK_CUBLAS_ERROR(status) \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error in file " << __FILE__ \
                  << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

const double PI = acos(-1.0);

/**
 * @brief Kernel to generate the N x N DFT matrix directly on the GPU.
 * Each thread computes one element of the matrix.
 * @param dft_matrix The output matrix W.
 * @param n The dimension of the matrix (N).
 */
__global__ void generate_dft_matrix_kernel(cuDoubleComplex* dft_matrix, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // k
    int col = blockIdx.x * blockDim.x + threadIdx.x; // t

    if (row < n && col < n) {
        double angle = -2.0 * PI * row * col / n;
        dft_matrix[row * n + col] = make_cuDoubleComplex(cos(angle), sin(angle));
    }
}

/**
 * @brief Host function to execute the DFT on the GPU using cuBLAS.
 * This version generates the DFT matrix on the GPU and then uses
 * cuBLAS for a highly optimized matrix-vector multiplication.
 * @param input The input signal vector.
 * @return A vector containing the DFT result.
 */
std::vector<std::complex<double>> dft_gpu_cublas(const std::vector<std::complex<double>>& input) {
    cudaEvent_t start_total, stop_total, start_compute, stop_compute;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_total));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_total));
    CHECK_CUDA_ERROR(cudaEventCreate(&start_compute));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_compute));

    CHECK_CUDA_ERROR(cudaEventRecord(start_total));

    int n = input.size();
    size_t vector_size = n * sizeof(cuDoubleComplex);
    size_t matrix_size = n * n * sizeof(cuDoubleComplex);

    // 1. Initialize cuBLAS
    cublasHandle_t handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&handle));

    // 2. Allocate GPU memory
    cuDoubleComplex *d_input, *d_output, *d_dft_matrix;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, vector_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, vector_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_dft_matrix, matrix_size));

    // Convert input to cuDoubleComplex format
    std::vector<cuDoubleComplex> h_input(n);
    for (int i = 0; i < n; i++) {
        h_input[i] = make_cuDoubleComplex(input[i].real(), input[i].imag());
    }

    // 3. Copy input vector from Host to Device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), vector_size, cudaMemcpyHostToDevice));

    CHECK_CUDA_ERROR(cudaEventRecord(start_compute));

    // 4. Generate DFT Matrix on the GPU
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (n + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (n + threadsPerBlock.y - 1) / threadsPerBlock.y
    );
    generate_dft_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_dft_matrix, n);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 5. Perform Matrix-Vector Multiplication using cuBLAS: d_output = d_dft_matrix * d_input
    const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0); // alpha = 1
    const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);  // beta = 0
    
    // cublasZgemv performs Y = alpha*A*x + beta*Y
    CHECK_CUBLAS_ERROR(cublasZgemv(handle, CUBLAS_OP_N, n, n, &alpha, d_dft_matrix, n, d_input, 1, &beta, d_output, 1));

    CHECK_CUDA_ERROR(cudaEventRecord(stop_compute));

    // 6. Copy result from Device to Host
    std::vector<cuDoubleComplex> h_output(n);
    CHECK_CUDA_ERROR(cudaMemcpy(h_output.data(), d_output, vector_size, cudaMemcpyDeviceToHost));

    // Convert result back to std::complex
    std::vector<std::complex<double>> output(n);
    for (int i = 0; i < n; i++) {
        output[i] = { cuCreal(h_output[i]), cuCimag(h_output[i]) };
    }

    CHECK_CUDA_ERROR(cudaEventRecord(stop_total));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_total));

    float ms_compute = 0, ms_total = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms_compute, start_compute, stop_compute));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms_total, start_total, stop_total));

    std::cout << "--- cuBLAS Version ---" << std::endl;
    std::cout << "Input size N: " << n << std::endl;
    std::cout << "GPU computation time (Matrix Gen + GEMV) (ms): " << ms_compute << std::endl;
    std::cout << "Total time (Transfers + Computation) (ms): " << ms_total << std::endl;

    // 7. Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_dft_matrix));
    CHECK_CUBLAS_ERROR(cublasDestroy(handle));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_total));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_total));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_compute));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_compute));

    return output;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>\n";
        return 1;
    }

    const char* filename = argv[1];
    std::ifstream ifs(filename);
    if (!ifs) {
        std::cerr << "Error: could not open file " << filename << "\n";
        return 1;
    }

    std::vector<std::complex<double>> data;
    double real, imag;
    while (ifs >> real >> imag) {
        data.emplace_back(real, imag);
    }
    ifs.close();

    if (data.empty()) {
        std::cerr << "Error: no data read from file " << filename << "\n";
        return 1;
    }

    // Execute the DFT on the GPU using cuBLAS
    auto result = dft_gpu_cublas(data);

    return 0;
}
