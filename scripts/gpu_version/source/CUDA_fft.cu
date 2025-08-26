#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <chrono>

// --- CUDA error checking ---
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " in file " << __FILE__ << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

// --- Constant PI in GPU memory ---
__constant__ double d_PI;

// --- Bit-reversal kernel ---
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

// --- FFT butterfly kernel per stage ---
__global__ void butterfly_kernel(cuDoubleComplex* data, size_t n, size_t len, double ang) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = tid % (len / 2);
    size_t block = tid / (len / 2);
    size_t i = block * len;
    if (i + j + len / 2 >= n) return;

    cuDoubleComplex w = make_cuDoubleComplex(cos(ang * j), sin(ang * j));
    cuDoubleComplex u = data[i + j];
    cuDoubleComplex v = cuCmul(data[i + j + len / 2], w);

    data[i + j] = cuCadd(u, v);
    data[i + j + len / 2] = cuCsub(u, v);
}

// --- Host GPU FFT wrapper with timing ---
std::vector<std::complex<double>> fft_gpu(
    const std::vector<std::complex<double>>& input,
    float& totalExecTime,
    float& kernelTime,
    float& h2dTime,
    float& d2hTime
) {
    cudaEvent_t start_total, stop_total, start_kernel, stop_kernel, start_h2d, stop_h2d, start_d2h, stop_d2h;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_total));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_total));
    CHECK_CUDA_ERROR(cudaEventCreate(&start_kernel));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_kernel));
    CHECK_CUDA_ERROR(cudaEventCreate(&start_h2d));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_h2d));
    CHECK_CUDA_ERROR(cudaEventCreate(&start_d2h));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_d2h));

    CHECK_CUDA_ERROR(cudaEventRecord(start_total));

    const double h_PI = acos(-1.0);
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_PI, &h_PI, sizeof(double)));

    size_t n = input.size();
    size_t buffer_size = n * sizeof(cuDoubleComplex);

    cuDoubleComplex* d_data;
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, buffer_size));

    std::vector<cuDoubleComplex> h_input(n);
    for (size_t i = 0; i < n; ++i)
        h_input[i] = make_cuDoubleComplex(input[i].real(), input[i].imag());

    // --- Host -> Device transfer ---
    CHECK_CUDA_ERROR(cudaEventRecord(start_h2d));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_input.data(), buffer_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaEventRecord(stop_h2d));

    // --- Kernel execution ---
    CHECK_CUDA_ERROR(cudaEventRecord(start_kernel));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    int logn = 0; while ((1u << logn) < n) ++logn;

    bit_reverse_kernel<<<blocks, threads>>>(d_data, n, logn);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    for (size_t len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * h_PI / len;
        size_t total_pairs = (n / len) * (len / 2);
        int blocks_b = (total_pairs + threads - 1) / threads;
        butterfly_kernel<<<blocks_b, threads>>>(d_data, n, len, ang);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    // --- Device -> Host transfer ---
    std::vector<cuDoubleComplex> h_output(n);
    CHECK_CUDA_ERROR(cudaEventRecord(start_d2h));
    CHECK_CUDA_ERROR(cudaMemcpy(h_output.data(), d_data, buffer_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaEventRecord(stop_d2h));

    CHECK_CUDA_ERROR(cudaEventRecord(stop_kernel));
    CHECK_CUDA_ERROR(cudaEventRecord(stop_total));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_total));

    // --- Compute timings ---
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&h2dTime, start_h2d, stop_h2d));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime, start_kernel, stop_kernel));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&d2hTime, start_d2h, stop_d2h));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&totalExecTime, start_total, stop_total));

    // --- Convert results ---
    std::vector<std::complex<double>> output(n);
    for (size_t i = 0; i < n; ++i)
        output[i] = { cuCreal(h_output[i]), cuCimag(h_output[i]) };

    CHECK_CUDA_ERROR(cudaFree(d_data));
    cudaEventDestroy(start_total); cudaEventDestroy(stop_total);
    cudaEventDestroy(start_kernel); cudaEventDestroy(stop_kernel);
    cudaEventDestroy(start_h2d); cudaEventDestroy(stop_h2d);
    cudaEventDestroy(start_d2h); cudaEventDestroy(stop_d2h);

    return output;
}

// --- MAIN ---
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <threads_per_block> <input_file> [num_runs]\n";
        return 1;
    }

    int threadsPerBlock = std::max(1, std::stoi(argv[1]));
    const char* filename = argv[2];
    int num_runs = (argc >= 4) ? std::max(1, std::stoi(argv[3])) : 1;

    // --- Read input file ---
    auto t_start_read = std::chrono::high_resolution_clock::now();
    std::ifstream ifs(filename);
    if (!ifs) { std::cerr << "Error opening file " << filename << "\n"; return 1; }

    std::vector<std::complex<double>> data;
    double re, im;
    while (ifs >> re >> im) data.emplace_back(re, im);
    ifs.close();
    auto t_end_read = std::chrono::high_resolution_clock::now();
    float read_ms = std::chrono::duration<float, std::milli>(t_end_read - t_start_read).count();
    std::cout << "[RESULTS] ReadingTime: " << read_ms << "ms\n";

    if (data.empty()) { std::cerr << "No data read\n"; return 1; }
    if ((data.size() & (data.size()-1)) != 0) { std::cerr << "Input size must be power of 2\n"; return 1; }

    if (data.size() > 65536) {
        std::cerr << "[WARNING] Input size > 65536, naive GPU FFT may be very slow!\n";
    }

    // --- Multiple runs ---
    float total_exec_ms = 0.0f;
    for (int run = 1; run <= num_runs; ++run) {
        float execTime=0, kernelTime=0, h2dTime=0, d2hTime=0;
        auto result = fft_gpu(data, execTime, kernelTime, h2dTime, d2hTime);

        std::cout << "[RESULTS] ExecutionTime(run=" << run << "): " << execTime << "ms\n";
        std::cout << "  (Details) Host->Device: " << h2dTime << "ms\n";
        std::cout << "  (Details) Kernel: " << kernelTime << "ms\n";
        std::cout << "  (Details) Device->Host: " << d2hTime << "ms\n";

        total_exec_ms += execTime;
    }

    std::cout << "[RESULTS] TotalTime: " << read_ms + total_exec_ms << "ms\n";

    return 0;
}