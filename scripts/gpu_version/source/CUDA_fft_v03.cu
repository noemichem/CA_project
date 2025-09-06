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
__constant__ float d_PI;

// --- Bit-reversal kernel ---
__global__ void bit_reverse_kernel(cuFloatComplex* data, size_t n, int logn) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    size_t rev;
    if (logn <= 32) {
        unsigned int idx32 = static_cast<unsigned int>(i);
        unsigned int r32 = __brev(idx32);
        rev = static_cast<size_t>(r32 >> (32 - logn));
    } else {
        unsigned long long idx64 = static_cast<unsigned long long>(i);
        unsigned long long r64 = __brevll(idx64);
        rev = static_cast<size_t>(r64 >> (64 - logn));
    }

    if (i < rev) {
        cuFloatComplex tmp = data[i];
        data[i] = data[rev];
        data[rev] = tmp;
    }
}

// --- FFT butterfly kernel per stage ---
__global__ void butterfly_kernel(cuFloatComplex* data, size_t n, size_t len) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = tid % (len / 2);
    size_t block = tid / (len / 2);
    size_t i = block * len;
    if (i + j + len / 2 >= n) return;

    float x = 2.0f * static_cast<float>(j) / static_cast<float>(len);
    float s, c;
    sincospif(x, &s, &c);
    cuFloatComplex w = make_cuFloatComplex(c, -s);

    cuFloatComplex u = data[i + j];
    cuFloatComplex v = cuCmulf(data[i + j + len / 2], w);

    data[i + j] = cuCaddf(u, v);
    data[i + j + len / 2] = cuCsubf(u, v);
}

// --- Host GPU FFT wrapper with timing ---
void fft_gpu(
    std::vector<std::complex<float>>& input,
    int threadsPerBlock,
    float& totalExecTime,
    float& kernelTime,
    float& h2dTime,
    float& d2hTime,
    float& bitreverseTime
) {
    size_t n = input.size();
    size_t buffer_size = n * sizeof(cuFloatComplex);
    cuFloatComplex* d_data;
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, buffer_size));

    std::vector<cuFloatComplex> h_input(n);
    for (size_t i = 0; i < n; ++i)
        h_input[i] = make_cuFloatComplex(input[i].real(), input[i].imag());

    // events
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Host -> Device
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_input.data(), buffer_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&h2dTime, start, stop));

    // Kernel execution
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    int logn = 0; while ((1u << logn) < n) ++logn;

    // bit-reversal
    cudaEvent_t startBitrev, stopBitrev;
    cudaEventCreate(&startBitrev);
    cudaEventCreate(&stopBitrev);
    cudaEventRecord(startBitrev);
    bit_reverse_kernel<<<blocks, threadsPerBlock>>>(d_data, n, logn);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    cudaEventRecord(stopBitrev);
    cudaEventSynchronize(stopBitrev);
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&bitreverseTime, startBitrev, stopBitrev));
    cudaEventDestroy(startBitrev);
    cudaEventDestroy(stopBitrev);

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (size_t len = 2; len <= n; len <<= 1) {
        size_t total_pairs = (n / len) * (len / 2);
        int blocks_b = (total_pairs + threadsPerBlock - 1) / threadsPerBlock;
        butterfly_kernel<<<blocks_b, threadsPerBlock>>>(d_data, n, len);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime, start, stop));

    // Device -> Host
    std::vector<cuFloatComplex> h_output(n);
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    CHECK_CUDA_ERROR(cudaMemcpy(h_output.data(), d_data, buffer_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&d2hTime, start, stop));

    // Convert results
    for (size_t i = 0; i < n; ++i)
        input[i] = { cuCrealf(h_output[i]), cuCimagf(h_output[i]) };

    totalExecTime = h2dTime + kernelTime + d2hTime + bitreverseTime;

    CHECK_CUDA_ERROR(cudaFree(d_data));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
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

    std::vector<std::complex<float>> data;
    float re, im;
    while (ifs >> re >> im) data.emplace_back(re, im);
    ifs.close();
    auto t_end_read = std::chrono::high_resolution_clock::now();
    float read_ms = std::chrono::duration<float, std::milli>(t_end_read - t_start_read).count();
    std::cout << "[RESULTS] ReadingTime: " << read_ms << "ms\n";

    if (data.empty()) { std::cerr << "No data read\n"; return 1; }
    if ((data.size() & (data.size()-1)) != 0) { std::cerr << "Input size must be power of 2\n"; return 1; }

    // --- Multiple runs ---
    float total_exec_ms = 0.0f;
    for (int run = 1; run <= num_runs; ++run) {
        float execTime=0, kernelTime=0, h2dTime=0, d2hTime=0, bitreverseTime=0;
        std::vector<std::complex<float>> inputCopy = data;
        fft_gpu(inputCopy, threadsPerBlock, execTime, kernelTime, h2dTime, d2hTime, bitreverseTime);

        std::cout << "[RESULTS] ExecutionTime(run=" << run << "): " << execTime << "ms\n";
        std::cout << "  (Details) Host->Device: " << h2dTime << "ms\n";
        std::cout << "  (Details) Kernel: " << kernelTime << "ms\n";
        std::cout << "  (Details) Device->Host: " << d2hTime << "ms\n";
        std::cout << "  (Details) Bit-reversal: " << bitreverseTime << "ms\n";

        total_exec_ms += execTime;
    }

    std::cout << "[RESULTS] TotalTime: " << (read_ms + total_exec_ms) << "ms\n";
    if (num_runs > 1) {
        std::cout << "[RESULTS] AverageExecutionTime: " << (total_exec_ms / num_runs) << "ms\n";
    }

    return 0;
}