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

// Macro per mappare l'indice logico a quello fisico con padding
#define SHARED_IDX(i) ( (i) + ((i) / NUM_BANKS) )

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
    // Usiamo la shared memory per evitare accessi globali non coalescenti
    extern __shared__ cuFloatComplex s_data[];

    size_t tid = threadIdx.x;
    size_t block_offset = blockIdx.x * blockDim.x;
    size_t i = block_offset + tid;

    // 1. Caricamento Coalescente da Globale a Shared Memory
    if (i < n) {
        s_data[tid] = data[i];
    }
    __syncthreads();

    // Se il blocco è fuori dal range dell'array, termina
    if (block_offset >= n) return;

    // 2. Calcolo dell'indice bit-reversed e scambio in Shared Memory
    unsigned int block_tid = static_cast<unsigned int>(tid);
    unsigned int r = __brev(block_tid); // __brev opera su 32-bit
    // Dobbiamo allineare il reverse alla dimensione del blocco (che è una potenza di 2)
    unsigned int log_block_dim = __popc(__brev(blockDim.x - 1));
    size_t rev_local = r >> (32 - log_block_dim);

    // Esegui lo scambio solo se l'indice è minore, per evitare doppi scambi
    if (tid < rev_local) {
        cuFloatComplex temp = s_data[tid];
        s_data[tid] = s_data[rev_local];
        s_data[rev_local] = temp;
    }
    __syncthreads();

    // 3. Scrittura Coalescente da Shared a Globale
    // Ogni thread scrive la sua posizione originale, ma con il valore potenzialmente scambiato
    if (i < n) {
        data[i] = s_data[tid];
    }
}

// ------------------- FFT Kernel Ottimizzato con Padding -------------------
__global__ void fft_kernel(cuFloatComplex* data, size_t n, int logn, int threadsPerBlock) {
    // *** MODIFICA 1: Usa la dimensione "paddata" per l'array in shared memory ***
    extern __shared__ cuFloatComplex s_data[];

    size_t tid = threadIdx.x;
    size_t gid = blockIdx.x * blockDim.x + tid;
    
    // Carica i dati iniziali nella shared memory usando l'indice mappato
    if (gid < n) {
        s_data[SHARED_IDX(tid)] = data[gid];
    }
    
    // Il loop sulle fasi della FFT rimane concettualmente identico
    for (int s = 1; s <= logn; ++s) {
        size_t m = 1ULL << s;
        size_t half_m = m >> 1;
        __syncthreads(); // Sincronizza per assicurare che la fase precedente sia completa

        if (m <= threadsPerBlock) {
            // FFT in Shared Memory
            float angle_base = -CUDART_PI_F / static_cast<float>(half_m);

            // Calcolo "butterfly"
            size_t k = tid % half_m;
            size_t butterfly_group_base = (tid / half_m) * m;
            
            size_t idx1 = butterfly_group_base + k;
            size_t idx2 = idx1 + half_m;

            if (idx2 < threadsPerBlock) {
                float angle = angle_base * static_cast<float>(k);
                float s_val, c_val;
                __sincosf(angle, &s_val, &c_val);
                cuFloatComplex w = make_cuFloatComplex(c_val, s_val);
                
                // *** MODIFICA 2: Usa la macro SHARED_IDX per TUTTI gli accessi a s_data ***
                cuFloatComplex t1 = s_data[SHARED_IDX(idx1)];
                cuFloatComplex t2 = complex_mul(s_data[SHARED_IDX(idx2)], w);

                s_data[SHARED_IDX(idx1)] = complex_add(t1, t2);
                s_data[SHARED_IDX(idx2)] = complex_sub(t1, t2);
            }
        } else {
            // FFT in memoria globale (per fasi più grandi)
            __syncthreads(); // Assicura che la scrittura da shared memory sia completa
            
            size_t j = gid & (half_m - 1);
            size_t i = gid & ~(m - 1);
            
            size_t index1 = i + j;
            size_t index2 = index1 + half_m;

            if (index2 < n) {
                float angle = -2.0f * CUDART_PI_F * static_cast<float>(j) / static_cast<float>(m);
                float s_val, c_val;
                __sincosf(angle, &s_val, &c_val);
                cuFloatComplex w = make_cuFloatComplex(c_val, s_val);

                cuFloatComplex u = data[index1];
                cuFloatComplex v = complex_mul(data[index2], w);

                data[index1] = complex_add(u, v);
                data[index2] = complex_sub(u, v);
            }
        }
    }
    __syncthreads();

    // Scrivi il risultato finale da shared memory a globale
    if (gid < n) {
        data[gid] = s_data[SHARED_IDX(tid)];
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
    const size_t PADDING = threadsPerBlock / NUM_BANKS;
    const size_t PADDED_THREADS_PER_BLOCK = threadsPerBlock + PADDING;
    size_t shared_mem_size_fft = PADDED_THREADS_PER_BLOCK * sizeof(cuFloatComplex);

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    bit_reverse_kernel<<<grid, block>>>(d_data, n, logn);
    cudaDeviceSynchronize();
    fft_kernel<<<grid, block, shared_mem_size_fft>>>(d_data, n, logn, threadsPerBlock);
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