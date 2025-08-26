#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
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

// Constant PI in GPU memory
__constant__ double d_PI;

// ==================== CUDA DFT Kernel ====================
__global__ void dft_kernel(const cuDoubleComplex* input, cuDoubleComplex* output, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) {
        cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
        for (int t = 0; t < n; ++t) {
            double angle = -2.0 * d_PI * k * t / n;
            cuDoubleComplex W = make_cuDoubleComplex(cos(angle), sin(angle));
            sum = cuCadd(sum, cuCmul(input[t], W));
        }
        output[k] = sum;
    }
}

// ==================== Host GPU DFT Wrapper ====================
std::vector<std::complex<double>> dft_gpu(
    const std::vector<std::complex<double>>& input,
    int threadsPerBlock,      // numero di thread per block
    float& totalExecTime,     // totale (alloc+transfers+kernel+dealloc)
    float& kernelTime,        // solo kernel
    float& h2dTime,           // Host->Device
    float& d2hTime            // Device->Host
) {
    int n = input.size();
    size_t buffer_size = n * sizeof(cuDoubleComplex);

    cudaEvent_t start_total, stop_total, start_kernel, stop_kernel;
    cudaEvent_t start_h2d, stop_h2d, start_d2h, stop_d2h;

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

    cuDoubleComplex* d_input;
    cuDoubleComplex* d_output;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, buffer_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, buffer_size));

    std::vector<cuDoubleComplex> h_input(n);
    for (int i = 0; i < n; ++i)
        h_input[i] = make_cuDoubleComplex(input[i].real(), input[i].imag());

    CHECK_CUDA_ERROR(cudaEventRecord(start_h2d));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), buffer_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaEventRecord(stop_h2d));

    CHECK_CUDA_ERROR(cudaEventRecord(start_kernel));
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    dft_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaEventRecord(stop_kernel));

    std::vector<cuDoubleComplex> h_output(n);
    CHECK_CUDA_ERROR(cudaEventRecord(start_d2h));
    CHECK_CUDA_ERROR(cudaMemcpy(h_output.data(), d_output, buffer_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaEventRecord(stop_d2h));

    CHECK_CUDA_ERROR(cudaEventRecord(stop_total));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_total));

    CHECK_CUDA_ERROR(cudaEventElapsedTime(&h2dTime, start_h2d, stop_h2d));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime, start_kernel, stop_kernel));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&d2hTime, start_d2h, stop_d2h));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&totalExecTime, start_total, stop_total));

    std::vector<std::complex<double>> output(n);
    for (int i = 0; i < n; ++i)
        output[i] = { cuCreal(h_output[i]), cuCimag(h_output[i]) };

    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    cudaEventDestroy(start_total); cudaEventDestroy(stop_total);
    cudaEventDestroy(start_kernel); cudaEventDestroy(stop_kernel);
    cudaEventDestroy(start_h2d); cudaEventDestroy(stop_h2d);
    cudaEventDestroy(start_d2h); cudaEventDestroy(stop_d2h);

    return output;
}

// ==================== MAIN ====================
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <threads_per_block> <input_file> [num_runs]\n";
        return 1;
    }

    int threadsPerBlock = std::max(1, std::stoi(argv[1]));
    const char* filename = argv[2];
    int num_runs = 1;
    if (argc >= 4) num_runs = std::max(1, std::stoi(argv[3]));

    // Read input file
    auto start_read = std::chrono::high_resolution_clock::now();
    std::ifstream ifs(filename);
    if (!ifs) { std::cerr << "Error opening file " << filename << "\n"; return 1; }

    std::vector<std::complex<double>> data;
    double real, imag;
    while (ifs >> real >> imag) data.emplace_back(real, imag);
    ifs.close();

    if (data.empty()) { std::cerr << "No data read from file\n"; return 1; }
    auto end_read = std::chrono::high_resolution_clock::now();
    float read_ms = std::chrono::duration<float, std::milli>(end_read - start_read).count();
    std::cout << "[RESULTS] ReadingTime: " << read_ms << "ms\n";

    float total_exec_ms = 0.0f;
    for (int run = 1; run <= num_runs; ++run) {
        float execTime = 0, kernelTime = 0, h2dTime = 0, d2hTime = 0;
        auto result = dft_gpu(data, threadsPerBlock, execTime, kernelTime, h2dTime, d2hTime);

        std::cout << "[RESULTS] ExecutionTime(run=" << run << "): " << execTime << "ms\n";
        std::cout << "  (Details) Host -> Device: " << h2dTime << "ms\n";
        std::cout << "  (Details) Kernel: " << kernelTime << "ms\n";
        std::cout << "  (Details) Device -> Host: " << d2hTime << "ms\n";
        total_exec_ms += execTime;
    }

    float totalTime = read_ms + total_exec_ms;
    std::cout << "[RESULTS] TotalTime: " << totalTime << "ms\n";

    return 0;
}