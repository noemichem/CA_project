#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>      
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <chrono>


// Macro for robust CUDA error checking
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " in file " << __FILE__ \
                  << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

// Declare PI in GPU constant memory
__constant__ double d_PI;

/*=== Kernel: Bit-reversal ===*/
__global__ void bit_reverse_kernel(cuDoubleComplex* data, size_t n, int logn) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    size_t rev = 0, x = i;
    for (int b = 0; b < logn; ++b) {
        rev = (rev << 1) | (x & 1);
        x >>= 1;
    }

    if (i < rev) {
        cuDoubleComplex tmp = data[i];
        data[i] = data[rev];
        data[rev] = tmp;
    }
}

/*=== Kernel: Butterfly per livello ===*/
__global__ void butterfly_kernel(cuDoubleComplex* data, size_t n, size_t len, double ang) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = tid % (len/2);
    size_t block = tid / (len/2);
    size_t i = block * len;
    if (i + j + len/2 >= n) return;

    cuDoubleComplex w = make_cuDoubleComplex(cos(ang*j), sin(ang*j));
    cuDoubleComplex u = data[i + j];
    cuDoubleComplex v = cuCmul(data[i + j + len/2], w);

    data[i + j] = cuCadd(u, v);
    data[i + j + len/2] = cuCsub(u, v);
}

/*=== Host function: FFT GPU ===*/
std::vector<std::complex<double>> fft_gpu(const std::vector<std::complex<double>>& input) {
    cudaEvent_t start_total, stop_total, start_kernel, stop_kernel;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_total));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_total));
    CHECK_CUDA_ERROR(cudaEventCreate(&start_kernel));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_kernel));

    CHECK_CUDA_ERROR(cudaEventRecord(start_total));

    const double h_PI = acos(-1.0);
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_PI, &h_PI, sizeof(double)));

    size_t n = input.size();
    size_t buffer_size = n * sizeof(cuDoubleComplex);

    cuDoubleComplex* d_data;
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, buffer_size));

    std::vector<cuDoubleComplex> h_input(n);
    for (size_t i = 0; i < n; ++i) {
        h_input[i] = make_cuDoubleComplex(input[i].real(), input[i].imag());
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_input.data(), buffer_size, cudaMemcpyHostToDevice));

    CHECK_CUDA_ERROR(cudaEventRecord(start_kernel));

    // Bit-reversal
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    int logn = 0; while ((1u << logn) < n) ++logn;
    bit_reverse_kernel<<<blocks, threads>>>(d_data, n, logn);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Iterative FFT stages
    for (size_t len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * h_PI / len;
        size_t total_pairs = (n / len) * (len/2);
        int blocks_b = (total_pairs + threads - 1) / threads;
        butterfly_kernel<<<blocks_b, threads>>>(d_data, n, len, ang);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    // Copy back results
    std::vector<cuDoubleComplex> h_output(n);
    CHECK_CUDA_ERROR(cudaMemcpy(h_output.data(), d_data, buffer_size, cudaMemcpyDeviceToHost));

    std::vector<std::complex<double>> output(n);
    for (size_t i = 0; i < n; ++i) {
        output[i] = { cuCreal(h_output[i]), cuCimag(h_output[i]) };
    }

    CHECK_CUDA_ERROR(cudaEventRecord(stop_kernel));
    CHECK_CUDA_ERROR(cudaEventRecord(stop_total));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_total));

    float ms_kernel = 0, ms_total = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms_kernel, start_kernel, stop_kernel));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms_total, start_total, stop_total));

    std::cout << "--- Custom FFT GPU ---\n";
    std::cout << "Input size N: " << n << "\n";
    std::cout << "GPU kernel time (ms): " << ms_kernel << "\n";
    std::cout << "Total time (ms, including memory transfers): " << ms_total << "\n";

    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_total));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_total));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_kernel));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_kernel));

    return output;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>\n";
        return 1;
    }

    const char* filename = argv[1];

    // Measure file reading time
    auto t_start_read = std::chrono::high_resolution_clock::now();
    std::ifstream ifs(filename);
    if (!ifs) { std::cerr << "Error opening file " << filename << "\n"; return 1; }

    std::vector<std::complex<double>> data;
    double re, im;
    while (ifs >> re >> im) data.emplace_back(re, im);
    ifs.close();
    auto t_end_read = std::chrono::high_resolution_clock::now();
    auto duration_read = std::chrono::duration_cast<std::chrono::milliseconds>(t_end_read - t_start_read);
    std::cout << "File reading time: " << duration_read.count() << " ms\n";

    if (data.empty()) { std::cerr << "No data read\n"; return 1; }
    if ((data.size() & (data.size()-1)) != 0) {
        std::cerr << "Input size must be power of 2\n";
        return 1;
    }

    auto result = fft_gpu(data);

    // Print first few results
    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < std::min<size_t>(4, result.size()); ++i) {
        std::cout << "Out[" << i << "] = (" << result[i].real() << ", " << result[i].imag() << ")\n";
    }

    return 0;
}