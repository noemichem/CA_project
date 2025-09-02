#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <chrono>
#include <math_constants.h>

#define NUM_BANKS 32

// Mappatura 2D virtuale per evitare bank conflicts
#define CONFLICT_FREE_IDX(i) ( ((i) / NUM_BANKS) * (NUM_BANKS + 1) + ((i) % NUM_BANKS) )

// --- CUDA Error Checking Macro ---
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " in file " << __FILE__ << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

// --- GPU Utility (invariato) ---
__device__ __forceinline__ cuFloatComplex complex_mul(cuFloatComplex a, cuFloatComplex b) {
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

// ------------------- KERNEL FINALE CON LOOP UNROLLING (4 STADI) -------------------
__global__ void fft_final_kernel(cuFloatComplex* data, size_t n, int logn, int threadsPerBlock) {
    extern __shared__ cuFloatComplex s_data[];

    size_t tid = threadIdx.x;
    size_t gid = blockIdx.x * blockDim.x + tid;

    // FASE 1: Caricamento coalescente
    if (gid < n) {
        s_data[CONFLICT_FREE_IDX(tid)] = data[gid];
    }
    __syncthreads();

    // FASE 2: Bit-Reversal in Shared Memory
    unsigned int local_rev_tid = __brev(static_cast<unsigned int>(tid));
    unsigned int log_block_dim = __popc(__brev(blockDim.x - 1));
    local_rev_tid >>= (32 - log_block_dim);

    if (tid < local_rev_tid) {
        cuFloatComplex temp = s_data[CONFLICT_FREE_IDX(tid)];
        s_data[CONFLICT_FREE_IDX(tid)] = s_data[CONFLICT_FREE_IDX(local_rev_tid)];
        s_data[CONFLICT_FREE_IDX(local_rev_tid)] = temp;
    }
    __syncthreads();

    // FASE 3: Stadi srotolati
    // --- Stadio 1 (m=2) ---
    if (1 <= logn) {
        size_t idx1 = tid * 2;
        size_t idx2 = idx1 + 1;
        if (idx2 < threadsPerBlock) {
            cuFloatComplex t1 = s_data[CONFLICT_FREE_IDX(idx1)];
            cuFloatComplex t2 = s_data[CONFLICT_FREE_IDX(idx2)];
            s_data[CONFLICT_FREE_IDX(idx1)] = complex_add(t1, t2);
            s_data[CONFLICT_FREE_IDX(idx2)] = complex_sub(t1, t2);
        }
        __syncthreads();
    }

    // --- Stadio 2 (m=4) ---
    if (2 <= logn) {
        size_t m = 4;
        size_t half_m = m >> 1;
        size_t idx1 = (tid / half_m) * m + (tid % half_m);
        size_t idx2 = idx1 + half_m;
        if (idx2 < threadsPerBlock) {
            float angle = __fmul_rn(-CUDART_PI_F / static_cast<float>(half_m), static_cast<float>(tid % half_m));
            float s_val, c_val;
            __sincosf(angle, &s_val, &c_val);
            cuFloatComplex w = make_cuFloatComplex(c_val, s_val);
            cuFloatComplex t1 = s_data[CONFLICT_FREE_IDX(idx1)];
            cuFloatComplex t2 = complex_mul(s_data[CONFLICT_FREE_IDX(idx2)], w);
            s_data[CONFLICT_FREE_IDX(idx1)] = complex_add(t1, t2);
            s_data[CONFLICT_FREE_IDX(idx2)] = complex_sub(t1, t2);
        }
        __syncthreads();
    }

    // --- Stadio 3 (m=8) ---
    if (3 <= logn) {
        size_t m = 8;
        size_t half_m = m >> 1;
        size_t idx1 = (tid / half_m) * m + (tid % half_m);
        size_t idx2 = idx1 + half_m;
        if (idx2 < threadsPerBlock) {
            float angle = __fmul_rn(-CUDART_PI_F / static_cast<float>(half_m), static_cast<float>(tid % half_m));
            float s_val, c_val;
            __sincosf(angle, &s_val, &c_val);
            cuFloatComplex w = make_cuFloatComplex(c_val, s_val);
            cuFloatComplex t1 = s_data[CONFLICT_FREE_IDX(idx1)];
            cuFloatComplex t2 = complex_mul(s_data[CONFLICT_FREE_IDX(idx2)], w);
            s_data[CONFLICT_FREE_IDX(idx1)] = complex_add(t1, t2);
            s_data[CONFLICT_FREE_IDX(idx2)] = complex_sub(t1, t2);
        }
        __syncthreads();
    }

    // --- Stadio 4 (m=16) ---
    if (4 <= logn) {
        size_t m = 16;
        size_t half_m = m >> 1;
        size_t idx1 = (tid / half_m) * m + (tid % half_m);
        size_t idx2 = idx1 + half_m;
        if (idx2 < threadsPerBlock) {
            float angle = __fmul_rn(-CUDART_PI_F / static_cast<float>(half_m), static_cast<float>(tid % half_m));
            float s_val, c_val;
            __sincosf(angle, &s_val, &c_val);
            cuFloatComplex w = make_cuFloatComplex(c_val, s_val);
            cuFloatComplex t1 = s_data[CONFLICT_FREE_IDX(idx1)];
            cuFloatComplex t2 = complex_mul(s_data[CONFLICT_FREE_IDX(idx2)], w);
            s_data[CONFLICT_FREE_IDX(idx1)] = complex_add(t1, t2);
            s_data[CONFLICT_FREE_IDX(idx2)] = complex_sub(t1, t2);
        }
        __syncthreads();
    }

    // --- Loop per stadi rimanenti (s ≥ 5) ---
    for (int s = 5; s <= logn; ++s) {
        size_t m = 1ULL << s;
        size_t half_m = m >> 1;
        if (m <= threadsPerBlock) {
            size_t idx1 = (tid / half_m) * m + (tid % half_m);
            size_t idx2 = idx1 + half_m;
            if (idx2 < threadsPerBlock) {
                float angle = __fmul_rn(-CUDART_PI_F / static_cast<float>(half_m), static_cast<float>(tid % half_m));
                float s_val, c_val;
                __sincosf(angle, &s_val, &c_val);
                cuFloatComplex w = make_cuFloatComplex(c_val, s_val);
                cuFloatComplex t1 = s_data[CONFLICT_FREE_IDX(idx1)];
                cuFloatComplex t2 = complex_mul(s_data[CONFLICT_FREE_IDX(idx2)], w);
                s_data[CONFLICT_FREE_IDX(idx1)] = complex_add(t1, t2);
                s_data[CONFLICT_FREE_IDX(idx2)] = complex_sub(t1, t2);
            }
        }
        __syncthreads();
    }

    // Scrittura finale
    if (gid < n) {
        data[gid] = s_data[CONFLICT_FREE_IDX(tid)];
    }
}

// ------------------- Host Wrapper (invariato) -------------------
void fft_gpu(std::vector<std::complex<float>>& input, int threadsPerBlock,
             float& totalExecTime, float& kernelTime,
             float& h2dTime, float& d2hTime) {

    size_t n = input.size();
    int logn = 0;
    while ((1ULL << logn) < n) ++logn;
    size_t sizeBytes = n * sizeof(cuFloatComplex);

    cuFloatComplex* d_data = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, sizeBytes));

    std::vector<cuFloatComplex> h_cdata(n);
    for (size_t i = 0; i < n; ++i) {
        h_cdata[i] = make_cuFloatComplex(input[i].real(), input[i].imag());
    }

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_cdata.data(), sizeBytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&h2dTime, start, stop));

    dim3 block(threadsPerBlock);
    dim3 grid((n + threadsPerBlock - 1) / threadsPerBlock);
    
    size_t num_rows = (threadsPerBlock + NUM_BANKS - 1) / NUM_BANKS;
    size_t shared_mem_elements = num_rows * (NUM_BANKS + 1);
    size_t shared_mem_size_fft = shared_mem_elements * sizeof(cuFloatComplex);

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    fft_final_kernel<<<grid, block, shared_mem_size_fft>>>(d_data, n, logn, threadsPerBlock);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernelTime, start, stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    CHECK_CUDA_ERROR(cudaMemcpy(h_cdata.data(), d_data, sizeBytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&d2hTime, start, stop));

    for (size_t i = 0; i < n; ++i) { input[i] = { h_cdata[i].x, h_cdata[i].y }; }
    totalExecTime = h2dTime + kernelTime + d2hTime;

    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
}

// ------------------- MAIN (invariato) -------------------
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <threads_per_block> <input_file> [num_runs]\n";
        return 1;
    }

    size_t threadsPerBlock = std::stoi(argv[1]);
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