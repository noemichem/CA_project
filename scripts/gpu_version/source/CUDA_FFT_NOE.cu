#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <chrono>
#include <iomanip>

using namespace std;

// --- CUDA error checking macro ---
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        cerr << "CUDA Error: " << cudaGetErrorString(err) \
             << " in file " << __FILE__ << " at line " << __LINE__ << endl; \
        exit(EXIT_FAILURE); \
    }

// --- Constant PI in GPU memory ---
__constant__ double d_PI;

// --- Kernel: performs one FFT stage ---
__global__ void fft_stage(cuFloatComplex* data, int n, int step) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int halfStep = step / 2;

    if (tid < n / 2) {
        int i = (tid / halfStep) * step + (tid % halfStep);

        cuFloatComplex u = data[i];
        cuFloatComplex v = data[i + halfStep];

        float angle = -2.0f * d_PI * (tid % halfStep) / step;
        cuFloatComplex w = make_cuFloatComplex(cosf(angle), sinf(angle));

        cuFloatComplex t = make_cuFloatComplex(
            v.x * w.x - v.y * w.y,
            v.x * w.y + v.y * w.x
        );

        data[i]            = make_cuFloatComplex(u.x + t.x, u.y + t.y);
        data[i + halfStep] = make_cuFloatComplex(u.x - t.x, u.y - t.y);
    }
}

// --- Host function: FFT GPU with detailed timing ---
void fft_gpu(vector<complex<float>>& input, int threadsPerBlock,
             float& totalExecTime, float& kernelTime,
             float& h2dTime, float& d2hTime) {

    const double h_PI = acos(-1.0);
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_PI, &h_PI, sizeof(double)));

    int N = input.size();
    size_t size = N * sizeof(cuFloatComplex);

    // Allocate device memory
    cuFloatComplex* d_data;
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, size));

    // Host->Device copy
    cudaEvent_t startH2D, stopH2D;
    CHECK_CUDA_ERROR(cudaEventCreate(&startH2D));
    CHECK_CUDA_ERROR(cudaEventCreate(&stopH2D));
    CHECK_CUDA_ERROR(cudaEventRecord(startH2D));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, input.data(), size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaEventRecord(stopH2D));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stopH2D));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&h2dTime, startH2D, stopH2D));

    // Kernel execution
    cudaEvent_t startKernel, stopKernel;
    CHECK_CUDA_ERROR(cudaEventCreate(&startKernel));
    CHECK_CUDA_ERROR(cudaEventCreate(&stopKernel));
    CHECK_CUDA_ERROR(cudaEventRecord(startKernel));

    for (int step = 2; step <= N; step <<= 1) {
        int butterflies = N / 2;
        int blocks = (butterflies + threadsPerBlock - 1) / threadsPerBlock;
        fft_stage<<<blocks, threadsPerBlock>>>(d_data, N, step);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    CHECK_CUDA_ERROR(cudaEventRecord(stopKernel));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stopKernel));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime, startKernel, stopKernel));

    // Device->Host copy
    vector<cuFloatComplex> h_output(N);
    cudaEvent_t startD2H, stopD2H;
    CHECK_CUDA_ERROR(cudaEventCreate(&startD2H));
    CHECK_CUDA_ERROR(cudaEventCreate(&stopD2H));
    CHECK_CUDA_ERROR(cudaEventRecord(startD2H));
    CHECK_CUDA_ERROR(cudaMemcpy(h_output.data(), d_data, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaEventRecord(stopD2H));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stopD2H));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&d2hTime, startD2H, stopD2H));

    // Convert back to std::complex
    for (int i = 0; i < N; i++) {
        input[i] = { h_output[i].x, h_output[i].y };
    }

    // Total execution
    totalExecTime = h2dTime + kernelTime + d2hTime;

    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaEventDestroy(startH2D));
    CHECK_CUDA_ERROR(cudaEventDestroy(stopH2D));
    CHECK_CUDA_ERROR(cudaEventDestroy(startKernel));
    CHECK_CUDA_ERROR(cudaEventDestroy(stopKernel));
    CHECK_CUDA_ERROR(cudaEventDestroy(startD2H));
    CHECK_CUDA_ERROR(cudaEventDestroy(stopD2H));
}

// --- MAIN ---
int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <threads_per_block> <input_file> [num_runs]\n";
        return 1;
    }

    int threadsPerBlock = stoi(argv[1]);
    const char* filename = argv[2];
    int numRuns = (argc >= 4) ? std::max(1, std::stoi(argv[3])) : 1; // default 1

    // Measure file reading time
    auto startRead = chrono::high_resolution_clock::now();
    ifstream ifs(filename);
    if (!ifs) { cerr << "Error opening file.\n"; return 1; }

    vector<complex<float>> data;
    float re, im;
    while (ifs >> re >> im) data.emplace_back(re, im);
    ifs.close();
    auto endRead = chrono::high_resolution_clock::now();
    float readTime = chrono::duration<float, milli>(endRead - startRead).count();

    cout << "[RESULTS] ReadingTime: " << readTime << "ms\n";

    if ((data.size() & (data.size() - 1)) != 0) {
        cerr << "FFT requires power-of-2 size.\n";
        return 1;
    }

    // Run FFT multiple times according to numRuns
    float totalTimeSum = 0;
    for (int run = 1; run <= numRuns; ++run) {
        float totalExec = 0, kernelTime = 0, h2dTime = 0, d2hTime = 0;
        fft_gpu(data, threadsPerBlock, totalExec, kernelTime, h2dTime, d2hTime);

        cout << "[RESULTS] ExecutionTime(run=" << run << "): " << totalExec << "ms\n";
        cout << "  (Details) Host->Device: " << h2dTime << "ms\n";
        cout << "  (Details) Kernel: " << kernelTime << "ms\n";
        cout << "  (Details) Device->Host: " << d2hTime << "ms\n";

        totalTimeSum += totalExec;
    }

    cout << "[RESULTS] TotalTime: " << (readTime + totalTimeSum) << "ms\n";

    return 0;
}