#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <cuda_runtime.h>
#include <cuComplex.h>

// Macro for robust CUDA error checking
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " in file " << __FILE__ \
                  << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

// OPTIMIZATION: Declare PI in the GPU's constant memory.
// This memory is cached and optimized for read-only access by all threads.
__constant__ double d_PI;

/**
 * @brief CUDA kernel for DFT computation.
 * Each thread calculates a single element of the output vector.
 * The complexity is O(N^2), distributed among the GPU threads.
 * @param input Pointer to the input vector (complex numbers) on GPU memory.
 * @param output Pointer to the output vector (complex numbers) on GPU memory.
 * @param n Size of the input and output vectors.
 */
__global__ void dft_kernel(const cuDoubleComplex* input, cuDoubleComplex* output, int n) {
    // Calculate the global thread index
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread performs the calculation only if its index is within the range of N
    if (k < n) {
        cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);

        // Calculate the k-th element of the DFT: X[k] = sum_{t=0}^{n-1} x[t] * exp(-j*2*pi*k*t/n)
        for (int t = 0; t < n; ++t) {
            // Use the constant PI from the GPU's constant memory
            double angle = -2.0 * d_PI * k * t / n;
            // Calculate the "twiddle factor" W_n^kt = exp(-j*2*pi*k*t/n)
            cuDoubleComplex W = make_cuDoubleComplex(cos(angle), sin(angle));
            // Multiply input[t] by the twiddle factor
            cuDoubleComplex term = cuCmul(input[t], W);
            // Add the result to the sum
            sum = cuCadd(sum, term);
        }
        output[k] = sum;
    }
}

/**
 * @brief Host (CPU) wrapper function to execute the DFT on the GPU.
 * It handles memory allocation, data transfers, and the kernel launch.
 * It also measures and prints the execution times.
 * @param input Input vector (std::vector of std::complex).
 * @return A vector containing the DFT result.
 */
std::vector<std::complex<double>> dft_gpu(const std::vector<std::complex<double>>& input) {
    // CUDA events for high-precision performance measurement
    cudaEvent_t start_total, stop_total, start_kernel, stop_kernel;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_total));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_total));
    CHECK_CUDA_ERROR(cudaEventCreate(&start_kernel));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_kernel));

    // --- Start Total Timing ---
    CHECK_CUDA_ERROR(cudaEventRecord(start_total));

    // Initialize PI on the host and copy it to the GPU's constant memory
    const double h_PI = acos(-1.0);
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_PI, &h_PI, sizeof(double)));

    int n = input.size();
    size_t buffer_size = n * sizeof(cuDoubleComplex);

    // 1. Allocate memory on the GPU
    cuDoubleComplex* d_input;
    cuDoubleComplex* d_output;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, buffer_size));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, buffer_size));

    // Convert from std::complex (host) to cuDoubleComplex (for CUDA)
    std::vector<cuDoubleComplex> h_input(n);
    for (int i = 0; i < n; i++) {
        h_input[i] = make_cuDoubleComplex(input[i].real(), input[i].imag());
    }

    // 2. Copy data from Host (CPU) to Device (GPU)
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), buffer_size, cudaMemcpyHostToDevice));

    // --- Start Kernel Timing ---
    CHECK_CUDA_ERROR(cudaEventRecord(start_kernel));

    // 3. Execute the Kernel on the GPU
    int threadsPerBlock = 256; // A common and usually efficient value
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock; // Calculated to cover all N elements
    dft_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    
    // Check for any errors launched by the kernel
    CHECK_CUDA_ERROR(cudaGetLastError());

    // --- End Kernel Timing ---
    CHECK_CUDA_ERROR(cudaEventRecord(stop_kernel));

    // Host vector to receive the results
    std::vector<cuDoubleComplex> h_output(n);

    // 4. Copy results from Device (GPU) to Host (CPU)
    CHECK_CUDA_ERROR(cudaMemcpy(h_output.data(), d_output, buffer_size, cudaMemcpyDeviceToHost));

    // Convert the result from cuDoubleComplex to std::complex
    std::vector<std::complex<double>> output(n);
    for (int i = 0; i < n; i++) {
        output[i] = { cuCreal(h_output[i]), cuCimag(h_output[i]) };
    }

    // --- End Total Timing ---
    CHECK_CUDA_ERROR(cudaEventRecord(stop_total));
    
    // Synchronize to ensure all events are completed before reading the timers
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_total));

    // Calculate and print timings
    float ms_kernel = 0, ms_total = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms_kernel, start_kernel, stop_kernel));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms_total, start_total, stop_total));

    std::cout << "Input size N: " << n << std::endl;
    std::cout << "GPU kernel execution time (ms): " << ms_kernel << std::endl;
    std::cout << "Total time (Transfers + Kernel) (ms): " << ms_total << std::endl;

    // 5. Cleanup memory and events
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_total));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_total));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_kernel));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_kernel));

    return output;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>\n";
        std::cerr << "The input file must contain pairs of numbers (real and imaginary parts) separated by spaces, one per line.\n";
        return 1;
    }

    const char* filename = argv[1];
    std::ifstream ifs(filename);
    if (!ifs) {
        std::cerr << "Error: could not open file " << filename << "\n";
        return 1;
    }

    std::vector<std::complex<double>> data;
    double real, imag;
    while (ifs >> real >> imag) {
        data.emplace_back(real, imag);
    }
    ifs.close();

    if (data.empty()) {
        std::cerr << "Error: no data read from file " << filename << "\n";
        return 1;
    }

    // Execute the DFT on the GPU
    auto result = dft_gpu(data);

    // (Optional) Print the first few results for verification
    // std::cout << "\nFirst 5 results:\n";
    // for (int i = 0; i < std::min(5, (int)result.size()); ++i) {
    //     std::cout << "Result[" << i << "] = " << result[i].real() << " + " << result[i].imag() << "i\n";
    // }

    return 0;
}
