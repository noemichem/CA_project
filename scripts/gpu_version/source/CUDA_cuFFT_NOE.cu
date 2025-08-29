// Miglioramento nella gestione del tempo: il tempo di plan non deve essere incluso nel tempo di esecuzione del kernel

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
std::vector<std::complex<double>> fft_gpu_cufft(
    const std::vector<std::complex<double>>& input,
    float& h2dTime, float& kernelTime, float& d2hTime,
    cufftHandle plan, cuDoubleComplex* d_data, cuDoubleComplex* h_buffer)
{
    int n = input.size();
    size_t vector_size = n * sizeof(cuDoubleComplex);

    // Convert input to cuDoubleComplex
    std::vector<cuDoubleComplex> h_input(n);
    for (int i = 0; i < n; i++)
        h_input[i] = make_cuDoubleComplex(input[i].real(), input[i].imag());

    // --- Host->Device
    cudaEvent_t startH2D, stopH2D;
    CHECK_CUDA_ERROR(cudaEventCreate(&startH2D));
    CHECK_CUDA_ERROR(cudaEventCreate(&stopH2D));
    CHECK_CUDA_ERROR(cudaEventRecord(startH2D));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_input.data(), vector_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaEventRecord(stopH2D));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stopH2D));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&h2dTime, startH2D, stopH2D));

    // --- FFT kernel execution
    cudaEvent_t startKernel, stopKernel;
    CHECK_CUDA_ERROR(cudaEventCreate(&startKernel));
    CHECK_CUDA_ERROR(cudaEventCreate(&stopKernel));
    CHECK_CUDA_ERROR(cudaEventRecord(startKernel));

    CHECK_CUFFT_ERROR(cufftExecZ2Z(plan, d_data, d_data, CUFFT_FORWARD));

    CHECK_CUDA_ERROR(cudaEventRecord(stopKernel));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stopKernel));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime, startKernel, stopKernel));

    // --- Device->Host
    std::vector<cuDoubleComplex> h_output(n);
    cudaEvent_t startD2H, stopD2H;
    CHECK_CUDA_ERROR(cudaEventCreate(&startD2H));
    CHECK_CUDA_ERROR(cudaEventCreate(&stopD2H));
    CHECK_CUDA_ERROR(cudaEventRecord(startD2H));
    CHECK_CUDA_ERROR(cudaMemcpy(h_output.data(), d_data, vector_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaEventRecord(stopD2H));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stopD2H));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&d2hTime, startD2H, stopD2H));

    // Convert to std::complex<double>
    std::vector<std::complex<double>> output(n);
    for (int i = 0; i < n; i++)
        output[i] = { cuCreal(h_output[i]), cuCimag(h_output[i]) };

    // Cleanup local events
    cudaEventDestroy(startH2D); cudaEventDestroy(stopH2D);
    cudaEventDestroy(startKernel); cudaEventDestroy(stopKernel);
    cudaEventDestroy(startD2H); cudaEventDestroy(stopD2H);

    return output;
}

// --- MAIN ---
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <num_runs>\n";
        return 1;
    }

    const char* filename = argv[1];
    int num_runs = std::max(1, std::stoi(argv[2]));

    // --- Read input file
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
    std::cout << "[RESULTS] ReadingTime: " << readTime << " ms\n";

    int n = data.size();
    size_t vector_size = n * sizeof(cuDoubleComplex);

    // --- Allocate device memory once
    cuDoubleComplex* d_data;
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, vector_size));

    // --- Create FFT plan once (planning time is separate from execution)
    cufftHandle plan;
    auto plan_start = std::chrono::high_resolution_clock::now();
    CHECK_CUFFT_ERROR(cufftPlan1d(&plan, n, CUFFT_Z2Z, 1));
    auto plan_end = std::chrono::high_resolution_clock::now();
    float planTime = std::chrono::duration<float, std::milli>(plan_end - plan_start).count();
    std::cout << "[RESULTS] PlanningTime: " << planTime << " ms\n";

    // --- Run multiple times
    float totalExecSum = 0, h2dSum = 0, d2hSum = 0, kernelSum = 0;
    for (int run = 1; run <= num_runs; ++run) {
        float h2d = 0, d2h = 0, kernel = 0;
        auto result = fft_gpu_cufft(data, h2d, kernel, d2h, plan, d_data, nullptr);

        float total = h2d + kernel + d2h;
        std::cout << "[RESULTS] ExecutionTime(run=" << run << "): " << total << " ms\n";
        std::cout << "   (Details) H2D: " << h2d << " ms, Kernel: " << kernel << " ms, D2H: " << d2h << " ms\n";

        totalExecSum += total;
        h2dSum += h2d;
        d2hSum += d2h;
        kernelSum += kernel;
    }

    // --- Print averages
    std::cout << "[RESULTS] Averages over " << num_runs << " runs:\n";
    std::cout << "   Avg H2D: " << (h2dSum / num_runs) << " ms\n";
    std::cout << "   Avg Kernel: " << (kernelSum / num_runs) << " ms\n";
    std::cout << "   Avg D2H: " << (d2hSum / num_runs) << " ms\n";
    std::cout << "   Avg Total (no plan): " << (totalExecSum / num_runs) << " ms\n";
    std::cout << "   PlanningTime (one-time): " << planTime << " ms\n";
    std::cout << "   Total including planning (1 run): " 
              << planTime + (totalExecSum / num_runs) << " ms\n";

    // --- Cleanup
    CHECK_CUFFT_ERROR(cufftDestroy(plan));
    CHECK_CUDA_ERROR(cudaFree(d_data));

    return 0;
}
