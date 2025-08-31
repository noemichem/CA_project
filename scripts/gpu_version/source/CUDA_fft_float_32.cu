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
        unsigned int r32 = __brev(idx32);                     // inverti i 32 bit
        rev = static_cast<size_t>(r32 >> (32 - logn));        // prendi i logn bit
    } else {
        unsigned long long idx64 = static_cast<unsigned long long>(i);
        unsigned long long r64 = __brevll(idx64);             // inverti i 64 bit
        rev = static_cast<size_t>(r64 >> (64 - logn));
    }

    // scambia i valori solo se i < rev (evita doppie permutazioni)
    if (i < rev) {
        cuFloatComplex tmp = data[i];
        data[i]  = data[rev];
        data[rev] = tmp;
    }
}


// --- FFT butterfly kernel per stage ---
// This version uses CUDA's sincospif() intrinsic to compute sine and cosine of π times
// a normalized argument. Computing sincospif() is often faster and more accurate than
// computing sinf() and cosf() separately, and it reduces register pressure.  For each
// butterfly, we compute the twiddle factor `w` as e^{-j 2π j / len}.  We compute
// sinf(2π j / len) and cosf(2π j / len) via sinpi/cospi by normalizing the angle
// to fractions of π.  Since sinpif(x) computes sinf(π·x), we set x=2*j/len.  The
// sine of the negative angle is obtained by negating the result.  This avoids
// computing trigonometric functions that are slower on many architectures.
__global__ void butterfly_kernel(cuFloatComplex* data, size_t n, size_t len) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = tid % (len / 2);
    size_t block = tid / (len / 2);
    size_t i = block * len;
    // Guard against out‑of‑bounds indices
    if (i + j + len / 2 >= n) return;

    // Compute the twiddle factor using sincospi.  The angle 2π*j/len is
    // rewritten as π * (2*j/len).  sincospif(x, &s, &c) computes s=sinf(π·x)
    // and c=cosf(π·x).  We then form w = cosf(2πj/len) + i·sinf(2πj/len), but
    // because the original FFT uses a negative sign (e^{-i2πj/len}), the
    // imaginary part is negated.
    float x = 2.0f * static_cast<float>(j) / static_cast<float>(len);
    float s, c;
    // Use CUDA intrinsic sincospi to compute sinf(pi * x) and cosf(pi * x)
    sincospif(x, &s, &c);
    cuFloatComplex w = make_cuFloatComplex(c, -s);

    cuFloatComplex u = data[i + j];
    cuFloatComplex v = cuCmulf(data[i + j + len / 2], w);

    data[i + j] = cuCaddf(u, v);
    data[i + j + len / 2] = cuCsubf(u, v);
}

// --- Host GPU FFT wrapper with timing ---
std::vector<std::complex<float>> fft_gpu(
    const std::vector<std::complex<float>>& input,
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

    const float h_PI = acos(-1.0f);
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_PI, &h_PI, sizeof(float)));

    size_t n = input.size();
    size_t buffer_size = n * sizeof(cuFloatComplex);

    cuFloatComplex* d_data;
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, buffer_size));

    std::vector<cuFloatComplex> h_input(n);
    for (size_t i = 0; i < n; ++i)
        h_input[i] = make_cuFloatComplex(input[i].real(), input[i].imag());

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
        // Each stage of the Cooley–Tukey FFT processes len-length blocks.  We compute
        // the number of butterfly operations (n / len) * (len / 2) and launch
        // enough threads to cover them.  The butterfly_kernel internally
        // computes the twiddle factors using sincospif(), so there is no need to
        // compute an angle here.  See butterfly_kernel for details.
        size_t total_pairs = (n / len) * (len / 2);
        int blocks_b = (total_pairs + threads - 1) / threads;
        butterfly_kernel<<<blocks_b, threads>>>(d_data, n, len);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    // --- Device -> Host transfer ---
    std::vector<cuFloatComplex> h_output(n);
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
    std::vector<std::complex<float>> output(n);
    for (size_t i = 0; i < n; ++i)
        output[i] = { cuCrealf(h_output[i]), cuCimagf(h_output[i]) };

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