// miglioramenti: float 32, pinned memory, stream, batch FFT, da testare

#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cuda_runtime.h>
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

// --- Host function: FFT with cuFFT (optimized) ---
std::vector<std::complex<float>> fft_gpu_cufft(
        const std::vector<std::complex<float>>& input,
        int batch, float& totalExecTime, float& kernelTime,
        float& h2dTime, float& d2hTime) {

    int n = input.size();  
    size_t vector_size = n * batch * sizeof(cufftComplex);

    // ⚡ 1. Allocazione Host in **Pinned Memory** (più veloce per trasferimenti H2D/D2H)
    cufftComplex* h_input;
    cufftComplex* h_output;
    CHECK_CUDA_ERROR(cudaMallocHost(&h_input, vector_size));  // pinned
    CHECK_CUDA_ERROR(cudaMallocHost(&h_output, vector_size));

    // Riempimento dati batchati (replico lo stesso input per semplicità)
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < n; i++) {
            int idx = b * n + i;
            h_input[idx].x = input[i].real();
            h_input[idx].y = input[i].imag();
        }
    }

    // ⚡ 2. Allocazione memoria Device
    cufftComplex* d_data;
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, vector_size));

    // ⚡ 3. Creazione Stream per overlap
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // ⚡ 4. Creazione Piano cuFFT con **batch**
    cufftHandle plan;
    CHECK_CUFFT_ERROR(cufftPlan1d(&plan, n, CUFFT_C2C, batch));
    CHECK_CUFFT_ERROR(cufftSetStream(plan, stream));  // lega il piano allo stream

    // Timing
    cudaEvent_t startH2D, stopH2D, startKernel, stopKernel, startD2H, stopD2H;
    cudaEventCreate(&startH2D); cudaEventCreate(&stopH2D);
    cudaEventCreate(&startKernel); cudaEventCreate(&stopKernel);
    cudaEventCreate(&startD2H); cudaEventCreate(&stopD2H);

    // --- Host->Device (asincrono, pinned + stream)
    cudaEventRecord(startH2D, stream);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_data, h_input, vector_size,
                                     cudaMemcpyHostToDevice, stream));
    cudaEventRecord(stopH2D, stream);

    // --- FFT execution (in stream, su batch intero)
    cudaEventRecord(startKernel, stream);
    CHECK_CUFFT_ERROR(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));
    cudaEventRecord(stopKernel, stream);

    // --- Device->Host (asincrono, pinned + stream)
    cudaEventRecord(startD2H, stream);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_output, d_data, vector_size,
                                     cudaMemcpyDeviceToHost, stream));
    cudaEventRecord(stopD2H, stream);

    // Sync finale
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Calcolo tempi
    cudaEventElapsedTime(&h2dTime, startH2D, stopH2D);
    cudaEventElapsedTime(&kernelTime, startKernel, stopKernel);
    cudaEventElapsedTime(&d2hTime, startD2H, stopD2H);
    totalExecTime = h2dTime + kernelTime + d2hTime;

    // Conversione output in std::complex<float>
    std::vector<std::complex<float>> output(n * batch);
    for (int i = 0; i < n * batch; i++)
        output[i] = { h_output[i].x, h_output[i].y };

    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_data);
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    cudaStreamDestroy(stream);

    return output;
}

// --- MAIN ---
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <threads_per_block> <input_file> [num_runs]\n";
        return 1;
    }

    int batch = std::max(1, std::stoi(argv[1]));
    const char* filename = argv[2];
    int num_runs = (argc >= 4) ? std::max(1, std::stoi(argv[3])) : 1;

    // File reading
    auto t_start = std::chrono::high_resolution_clock::now();
    std::ifstream ifs(filename);
    if (!ifs) { std::cerr << "Error opening file.\n"; return 1; }

    std::vector<std::complex<float>> data;
    float re, im;
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
        auto result = fft_gpu_cufft(data, batch, totalExec, kernelTime, h2dTime, d2hTime);

        std::cout << "[RESULTS] ExecutionTime(run=" << run << "): " << totalExec << "ms\n";
        std::cout << "  (Details) Host->Device: " << h2dTime << "ms\n";
        std::cout << "  (Details) Kernel: " << kernelTime << "ms\n";
        std::cout << "  (Details) Device->Host: " << d2hTime << "ms\n";

        totalTimeSum += totalExec;
    }

    std::cout << "[RESULTS] TotalTime: " << (readTime + totalTimeSum) << "ms\n";

    return 0;
}
