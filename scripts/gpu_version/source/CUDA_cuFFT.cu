#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>
#include <chrono>

#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " in file " << __FILE__ << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

#define CHECK_CUFFT_ERROR(status) \
    if (status != CUFFT_SUCCESS) { \
        std::cerr << "cuFFT Error in file " << __FILE__ \
                  << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

// --- Host function: FFT with cuFFT and detailed timing ---
std::vector<std::complex<double>> fft_gpu_cufft(const std::vector<std::complex<double>>& input,
                                               float& totalExecTime, float& kernelTime,
                                               float& h2dTime, float& d2hTime) {

    int n = input.size();
    size_t vector_size = n * sizeof(cuDoubleComplex);

    // Allocate device memory
    cuDoubleComplex* d_data;
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, vector_size));

    // Convert input to cuDoubleComplex
    std::vector<cuDoubleComplex> h_input(n);
    for (int i = 0; i < n; i++)
        h_input[i] = make_cuDoubleComplex(input[i].real(), input[i].imag());

    // Host->Device
    cudaEvent_t startH2D, stopH2D;
    CHECK_CUDA_ERROR(cudaEventCreate(&startH2D));
    CHECK_CUDA_ERROR(cudaEventCreate(&stopH2D));
    CHECK_CUDA_ERROR(cudaEventRecord(startH2D));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_input.data(), vector_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaEventRecord(stopH2D));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stopH2D));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&h2dTime, startH2D, stopH2D));

    // Kernel (FFT) execution
    cudaEvent_t startKernel, stopKernel;
    CHECK_CUDA_ERROR(cudaEventCreate(&startKernel));
    CHECK_CUDA_ERROR(cudaEventCreate(&stopKernel));
    CHECK_CUDA_ERROR(cudaEventRecord(startKernel));

    cufftHandle plan;
    CHECK_CUFFT_ERROR(cufftPlan1d(&plan, n, CUFFT_Z2Z, 1));
    CHECK_CUFFT_ERROR(cufftExecZ2Z(plan, d_data, d_data, CUFFT_FORWARD));

    CHECK_CUDA_ERROR(cudaEventRecord(stopKernel));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stopKernel));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime, startKernel, stopKernel));

    // Device->Host
    std::vector<cuDoubleComplex> h_output(n);
    cudaEvent_t startD2H, stopD2H;
    CHECK_CUDA_ERROR(cudaEventCreate(&startD2H));
    CHECK_CUDA_ERROR(cudaEventCreate(&stopD2H));
    CHECK_CUDA_ERROR(cudaEventRecord(startD2H));
    CHECK_CUDA_ERROR(cudaMemcpy(h_output.data(), d_data, vector_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaEventRecord(stopD2H));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stopD2H));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&d2hTime, startD2H, stopD2H));

    // Convert to std::complex
    std::vector<std::complex<double>> output(n);
    for (int i = 0; i < n; i++)
        output[i] = { cuCreal(h_output[i]), cuCimag(h_output[i]) };

    // Total execution time
    totalExecTime = h2dTime + kernelTime + d2hTime;

    // Cleanup
    CHECK_CUFFT_ERROR(cufftDestroy(plan));
    CHECK_CUDA_ERROR(cudaFree(d_data));
    cudaEventDestroy(startH2D); cudaEventDestroy(stopH2D);
    cudaEventDestroy(startKernel); cudaEventDestroy(stopKernel);
    cudaEventDestroy(startD2H); cudaEventDestroy(stopD2H);

    return output;
}

// --- MAIN ---
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <threads_per_block> <input_file> [num_runs]\n";
        return 1;
    }

    int threadsPerBlock = std::max(1, std::stoi(argv[1])); // Not used in this algorithm but kept for compatibility with other implementations
    const char* filename = argv[2];
    int num_runs = (argc >= 4) ? std::max(1, std::stoi(argv[3])) : 1;

    // File reading
    auto t_start = std::chrono::high_resolution_clock::now();
    std::ifstream ifs(filename);
    if (!ifs) { std::cerr << "Error opening file.\n"; return 1; }

    std::vector<std::complex<double>> data;
    double re, im;
    while (ifs >> re >> im)
        data.emplace_back(re, im);
    ifs.close();
    auto t_end = std::chrono::high_resolution_clock::now();
    float readTime = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    std::cout << "[RESULTS] ReadingTime: " << readTime << "ms\n";

    // Run multiple times
    float totalTimeSum = 0;
    for (int run = 1; run <= num_runs; ++run) {
        float totalExec = 0, kernelTime = 0, h2dTime = 0, d2hTime = 0;
        auto result = fft_gpu_cufft(data, totalExec, kernelTime, h2dTime, d2hTime);

        std::cout << "[RESULTS] ExecutionTime(run=" << run << "): " << totalExec << "ms\n";
        std::cout << "  (Details) Host->Device: " << h2dTime << "ms\n";
        std::cout << "  (Details) Kernel: " << kernelTime << "ms\n";
        std::cout << "  (Details) Device->Host: " << d2hTime << "ms\n";

        totalTimeSum += totalExec;
    }

    std::cout << "[RESULTS] TotalTime: " << (readTime + totalTimeSum) << "ms\n";

    return 0;
}