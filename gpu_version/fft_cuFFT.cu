#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h> // Header for the cuFFT library

// Macro for robust CUDA error checking
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " in file " << __FILE__ \
                  << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

// Macro for robust cuFFT error checking
#define CHECK_CUFFT_ERROR(status) \
    if (status != CUFFT_SUCCESS) { \
        std::cerr << "cuFFT Error in file " << __FILE__ \
                  << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

/**
 * @brief Host function to execute the FFT on the GPU using the cuFFT library.
 * This is the most optimized approach, leveraging NVIDIA's dedicated FFT algorithms.
 * @param input The input signal vector. Its size MUST be a power of 2 for best performance.
 * @return A vector containing the FFT result.
 */
std::vector<std::complex<double>> fft_gpu_cufft(const std::vector<std::complex<double>>& input) {
    cudaEvent_t start_total, stop_total, start_compute, stop_compute;
    CHECK_CUDA_ERROR(cudaEventCreate(&start_total));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_total));
    CHECK_CUDA_ERROR(cudaEventCreate(&start_compute));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop_compute));

    CHECK_CUDA_ERROR(cudaEventRecord(start_total));

    int n = input.size();
    size_t vector_size = n * sizeof(cuDoubleComplex);

    // 1. Allocate GPU memory
    cuDoubleComplex *d_data;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data, vector_size));

    // Convert input to cuDoubleComplex format
    std::vector<cuDoubleComplex> h_input(n);
    for (int i = 0; i < n; i++) {
        h_input[i] = make_cuDoubleComplex(input[i].real(), input[i].imag());
    }

    // 2. Copy input vector from Host to Device
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_input.data(), vector_size, cudaMemcpyHostToDevice));

    CHECK_CUDA_ERROR(cudaEventRecord(start_compute));

    // 3. Create a cuFFT plan
    cufftHandle plan;
    CHECK_CUFFT_ERROR(cufftPlan1d(&plan, n, CUFFT_Z2Z, 1)); // Z2Z: Double precision, Complex-to-Complex

    // 4. Execute the FFT
    // The transformation is done in-place, so d_data will be overwritten with the result.
    CHECK_CUFFT_ERROR(cufftExecZ2Z(plan, d_data, d_data, CUFFT_FORWARD));

    CHECK_CUDA_ERROR(cudaEventRecord(stop_compute));

    // 5. Copy result from Device to Host
    std::vector<cuDoubleComplex> h_output(n);
    CHECK_CUDA_ERROR(cudaMemcpy(h_output.data(), d_data, vector_size, cudaMemcpyDeviceToHost));

    // Convert result back to std::complex
    std::vector<std::complex<double>> output(n);
    for (int i = 0; i < n; i++) {
        output[i] = { cuCreal(h_output[i]), cuCimag(h_output[i]) };
    }

    CHECK_CUDA_ERROR(cudaEventRecord(stop_total));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop_total));

    float ms_compute = 0, ms_total = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms_compute, start_compute, stop_compute));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms_total, start_total, stop_total));

    std::cout << "--- cuFFT Version ---" << std::endl;
    std::cout << "Input size N: " << n << std::endl;
    std::cout << "GPU computation time (FFT execution) (ms): " << ms_compute << std::endl;
    std::cout << "Total time (Transfers + Computation) (ms): " << ms_total << std::endl;

    // 6. Cleanup
    CHECK_CUFFT_ERROR(cufftDestroy(plan));
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_total));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_total));
    CHECK_CUDA_ERROR(cudaEventDestroy(start_compute));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop_compute));

    return output;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>\n";
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
    
    // For best performance, FFT input size should be a power of 2.
    size_t n = data.size();
    if ((n > 0) && ((n & (n - 1)) != 0)) {
        std::cout << "Warning: Input size " << n << " is not a power of 2. cuFFT performance may be suboptimal." << std::endl;
    }


    // Execute the FFT on the GPU using cuFFT
    auto result = fft_gpu_cufft(data);

    return 0;
}
