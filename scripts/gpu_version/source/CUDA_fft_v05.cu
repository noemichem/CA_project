#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <chrono>
#include <math_constants.h>

// --- CUDA Error Checking Macro ---
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " in file " << __FILE__ << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

// ------------------- GPU Utility Ottimizzata -------------------
__device__ __forceinline__ cuFloatComplex complex_mul(cuFloatComplex a, cuFloatComplex b) {
    // (a.x + i*a.y) * (b.x + i*b.y) = (a.x*b.x - a.y*b.y) + i*(a.x*b.y + a.y*b.x)
    float real = __fmaf_rn(-a.y, b.y, __fmul_rn(a.x, b.x));
    float imag = __fmaf_rn(a.y, b.x, __fmul_rn(a.x, b.y));
    return make_cuFloatComplex(real, imag);
}

__device__ __forceinline__ cuFloatComplex complex_add(cuFloatComplex a, cuFloatComplex b) {
    return make_cuFloatComplex(__fadd_rn(a.x, b.x), __fadd_rn(a.y, b.y));
}

__device__ __forceinline__ cuFloatComplex complex_sub(cuFloatComplex a, cuFloatComplex b) {
    return make_cuFloatComplex(__fsub_rn(a.x, b.x), __fsub_rn(a.y, b.y));
}

// ------------------- Bit Reversal -------------------
__global__ void bit_reverse_kernel(cuFloatComplex* data, size_t n, int logn) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    unsigned long long idx64 = static_cast<unsigned long long>(i);
    unsigned long long r64   = __brevll(idx64);
    size_t rev = static_cast<size_t>(r64 >> (64 - logn));

    if (i < rev) {
        cuFloatComplex tmp = data[i];
        data[i]  = data[rev];
        data[rev] = tmp;
    }
}

// ------------------- FFT Kernel Ottimizzato con Shared Memory Sicura -------------------
__global__ void fft_kernel(cuFloatComplex* data, size_t n, int logn) {
    extern __shared__ cuFloatComplex s_data[];

    size_t tid = threadIdx.x;
    size_t gid = blockIdx.x * blockDim.x + tid;
    if (gid >= n) return;

    // Carica dati in shared memory solo se block può contenere tutta la fase
    for (int s = 1; s <= logn; ++s) {
        size_t m = 1ULL << s;
        size_t half = m >> 1;

        if (m <= blockDim.x) {
            // Shared memory FFT
            if (tid < m) s_data[tid] = data[blockIdx.x * blockDim.x + tid];
            __syncthreads();

            size_t j = tid & (half - 1);
            size_t block_idx = (tid >> (s - 1)) * m;
            size_t index1 = block_idx + j;
            size_t index2 = index1 + half;

            if (index2 < n) {
                float j_f = static_cast<float>(j);
                float m_f = static_cast<float>(m);
                float angle = __fdividef(__fmul_rn(-2.0f * CUDART_PI_F, j_f), m_f);

                float s_val, c_val;
                __sincosf(angle, &s_val, &c_val);
                cuFloatComplex w = make_cuFloatComplex(c_val, s_val);

                cuFloatComplex u = s_data[index1];
                cuFloatComplex v = complex_mul(s_data[index2], w);

                s_data[index1] = complex_add(u, v);
                s_data[index2] = complex_sub(u, v);
            }
            __syncthreads();

            // Scrivi fase su memoria globale
            if (tid < m) data[blockIdx.x * blockDim.x + tid] = s_data[tid];

        } else {
            // FFT su memoria globale
            size_t j = tid & (half - 1);
            size_t block_idx = (tid >> (s - 1)) * m;
            size_t index1 = block_idx + j;
            size_t index2 = index1 + half;

            if (index2 < n) {
                float j_f = static_cast<float>(j);
                float m_f = static_cast<float>(m);
                float angle = __fdividef(__fmul_rn(-2.0f * CUDART_PI_F, j_f), m_f);

                float s_val, c_val;
                __sincosf(angle, &s_val, &c_val);
                cuFloatComplex w = make_cuFloatComplex(c_val, s_val);

                cuFloatComplex u = data[index1];
                cuFloatComplex v = complex_mul(data[index2], w);

                data[index1] = complex_add(u, v);
                data[index2] = complex_sub(u, v);
            }
            __syncthreads(); // sicuro per i thread nello stesso block
        }
    }
}

// ------------------- Host Wrapper -------------------
void fft_gpu(std::vector<std::complex<float>>& input, int threadsPerBlock,
             float& totalExecTime, float& kernelTime,
             float& h2dTime, float& d2hTime) {

    size_t n = input.size();
    int logn = 0;
    while ((1ULL << logn) < n) ++logn;

    size_t sizeBytes = n * sizeof(cuFloatComplex);

    cuFloatComplex* d_data = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, sizeBytes));

    // Convert input to cuFloatComplex
    std::vector<cuFloatComplex> h_cdata(n);
    for (size_t i = 0; i < n; ++i) {
        h_cdata[i] = make_cuFloatComplex(input[i].real(), input[i].imag());
    }

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Host -> Device
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_cdata.data(), sizeBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&h2dTime, start, stop));

    // Kernel execution
    dim3 block(threadsPerBlock);
    dim3 grid((n + block.x - 1) / block.x);
    size_t shared_mem_size = block.x * sizeof(cuFloatComplex);

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    bit_reverse_kernel<<<grid, block>>>(d_data, n, logn);
    cudaDeviceSynchronize();
    fft_kernel<<<grid, block, shared_mem_size>>>(d_data, n, logn);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime, start, stop));

    // Device -> Host
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    CHECK_CUDA_ERROR(cudaMemcpy(h_cdata.data(), d_data, sizeBytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&d2hTime, start, stop));

    // Convert back to std::complex
    for (size_t i = 0; i < n; ++i) {
        input[i] = { h_cdata[i].x, h_cdata[i].y };
    }

    totalExecTime = h2dTime + kernelTime + d2hTime;

    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
}

// ------------------- MAIN -------------------
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <threads_per_block> <input_file> [num_runs]\n";
        return 1;
    }

    int threadsPerBlock = std::stoi(argv[1]);
    const char* filename = argv[2];
    int numRuns = (argc >= 4) ? std::max(1, std::stoi(argv[3])) : 1;

    auto startRead = std::chrono::high_resolution_clock::now();
    std::ifstream ifs(filename);
    if (!ifs) { std::cerr << "Error opening file.\n"; return 1; }

    std::vector<std::complex<float>> data;
    float re, im;
    while (ifs >> re >> im) data.emplace_back(re, im);
    ifs.close();
    auto endRead = std::chrono::high_resolution_clock::now();
    float readTime = std::chrono::duration<float, std::milli>(endRead - startRead).count();
    std::cout << "[RESULTS] ReadingTime: " << readTime << "ms\n";

    if (data.empty() || (data.size() & (data.size() - 1)) != 0) {
        std::cerr << "Input data size must be a power-of-2.\n";
        return 1;
    }

    float totalTimeSum = 0;
    for (int run = 1; run <= numRuns; ++run) {
        float totalExec = 0, kernelTime = 0, h2dTime = 0, d2hTime = 0;
        std::vector<std::complex<float>> inputCopy = data;
        fft_gpu(inputCopy, threadsPerBlock, totalExec, kernelTime, h2dTime, d2hTime);

        std::cout << "[RESULTS] ExecutionTime(run=" << run << "): " << totalExec << "ms\n";
        std::cout << "  (Details) Host->Device: " << h2dTime << "ms\n";
        std::cout << "  (Details) Kernel: " << kernelTime << "ms\n";
        std::cout << "  (Details) Device->Host: " << d2hTime << "ms\n";

        totalTimeSum += totalExec;
    }

    std::cout << "[RESULTS] TotalTime: " << (readTime + totalTimeSum) << "ms\n";
    if (numRuns > 1) {
        std::cout << "[RESULTS] AverageExecutionTime: " << (totalTimeSum / numRuns) << "ms\n";
    }

    return 0;
}