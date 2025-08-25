// Compile with:
//     nvcc -O3 -std=c++17 fft_gpu_custom.cu
// Run:
//     ./a.out <input_file>
// Input format: each line has two doubles: <real> <imag>
// Notes:
//  - Implements custom iterative FFT with bit-reverse and butterfly kernels.
//  - For educational use: not as optimized as cuFFT.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <complex>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <ctime>

#include <cuda_runtime.h>

static constexpr double PI = acos(-1);

struct Complex {
    double x, y;
    __host__ __device__ Complex(double r=0, double i=0): x(r), y(i) {}
    __host__ __device__ Complex operator+(const Complex& b) const { return Complex(x+b.x, y+b.y); }
    __host__ __device__ Complex operator-(const Complex& b) const { return Complex(x-b.x, y-b.y); }
    __host__ __device__ Complex operator*(const Complex& b) const { return Complex(x*b.x - y*b.y, x*b.y + y*b.x); }
};

// === Kernel: bit-reversal ===
__global__ void bit_reverse_kernel(Complex* data, size_t n, int logn) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    size_t rev = 0, x = i;
    for (int b = 0; b < logn; ++b) {
        rev = (rev << 1) | (x & 1);
        x >>= 1;
    }
    if (i < rev) {
        Complex tmp = data[i];
        data[i] = data[rev];
        data[rev] = tmp;
    }
}

// === Kernel: butterfly per livello ===
__global__ void butterfly_kernel(Complex* data, size_t n, size_t len, double ang) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = tid % (len/2);
    size_t block = tid / (len/2);
    size_t i = block * len;
    if (i + j + len/2 >= n) return;

    Complex w(cos(ang * j), sin(ang * j));
    Complex u = data[i + j];
    Complex v = data[i + j + len/2] * w;
    data[i + j] = u + v;
    data[i + j + len/2] = u - v;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>\n";
        return 1;
    }
    const char* filename = argv[1];

    // Lettura
    auto start_read_sys = std::chrono::system_clock::now();
    std::ifstream ifs(filename);
    if (!ifs) { std::cerr << "Error opening file: " << filename << "\n"; return 1; }

    std::vector<Complex> h_data;
    double re, im;
    while (ifs >> re >> im) h_data.emplace_back(re, im);
    ifs.close();
    size_t n = h_data.size();
    if ((n & (n - 1)) != 0) { std::cerr << "FFT requires power of 2. Got " << n << "\n"; return 1; }
    int logn = 0; while ((1u << logn) < n) ++logn;

    auto end_read_sys = std::chrono::system_clock::now();
    auto duration_read = std::chrono::duration_cast<std::chrono::milliseconds>(end_read_sys - start_read_sys);
    std::cout << "Durata Lettura: " << duration_read.count() << " ms\n";

    // Alloca device buffer
    Complex* d_data = nullptr;
    cudaMalloc(&d_data, n * sizeof(Complex));
    cudaMemcpy(d_data, h_data.data(), n*sizeof(Complex), cudaMemcpyHostToDevice);

    // Kernel config
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // === Bit-reverse ===
    bit_reverse_kernel<<<blocks, threads>>>(d_data, n, logn);
    cudaDeviceSynchronize();

    // === Butterflies per livello ===
    for (size_t len = 2; len <= n; len <<= 1) {
        double ang = -2 * PI / len;
        size_t total_pairs = (n/len) * (len/2);
        int blocks_b = (total_pairs + threads - 1) / threads;
        butterfly_kernel<<<blocks_b, threads>>>(d_data, n, len, ang);
        cudaDeviceSynchronize();
    }

    // Copia indietro
    cudaMemcpy(h_data.data(), d_data, n*sizeof(Complex), cudaMemcpyDeviceToHost);

    // Stampa i primi 4 risultati
    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < std::min<size_t>(4, n); ++i) {
        std::cout << "Out[" << i << "]: (" << h_data[i].x << ", " << h_data[i].y << ")\n";
    }

    cudaFree(d_data);
    return 0;
}
