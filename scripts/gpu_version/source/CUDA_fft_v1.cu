#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include <cuComplex.h>

using namespace std;

static double PI = acos(-1);

// === Kernel Bit-reversal ===
__global__ void bit_reverse(cuDoubleComplex* a, int n, int logn) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    unsigned int rev = 0;
    unsigned int x = i;
    for (int b = 0; b < logn; b++) {
        rev = (rev << 1) | (x & 1);
        x >>= 1;
    }
    if (i < rev) {
        cuDoubleComplex tmp = a[i];
        a[i] = a[rev];
        a[rev] = tmp;
    }
}

// === Kernel Butterfly for a given stage ===
__global__ void fft_stage(cuDoubleComplex* a, int n, int len, double ang) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n/len * (len/2)) return;

    int block = i / (len/2);
    int j = i % (len/2);
    int start = block * len;

    cuDoubleComplex w = make_cuDoubleComplex(cos(j*ang), sin(j*ang));

    cuDoubleComplex u = a[start + j];
    cuDoubleComplex v = cuCmul(a[start + j + len/2], w);

    a[start + j] = cuCadd(u, v);
    a[start + j + len/2] = cuCsub(u, v);
}

// === GPU FFT (with CUDA events for timing) ===
void fft_cuda(vector<complex<double>>& data,
              int threadsPerBlock,
              float &h2dTime, float &kernelTime, float &d2hTime) {
    int n = data.size();
    int logn = 0;
    while ((1 << logn) < n) ++logn;

    // Alloc device memory
    cuDoubleComplex* d_data;
    cudaMalloc(&d_data, n * sizeof(cuDoubleComplex));

    vector<cuDoubleComplex> h_data(n);
    for (int i = 0; i < n; i++)
        h_data[i] = make_cuDoubleComplex(data[i].real(), data[i].imag());

    // Events
    cudaEvent_t startH2D, stopH2D;
    cudaEvent_t startKernel, stopKernel;
    cudaEvent_t startD2H, stopD2H;
    cudaEventCreate(&startH2D);
    cudaEventCreate(&stopH2D);
    cudaEventCreate(&startKernel);
    cudaEventCreate(&stopKernel);
    cudaEventCreate(&startD2H);
    cudaEventCreate(&stopD2H);

    // === Host -> Device ===
    cudaEventRecord(startH2D);
    cudaMemcpy(d_data, h_data.data(), n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaEventRecord(stopH2D);
    cudaEventSynchronize(stopH2D);
    cudaEventElapsedTime(&h2dTime, startH2D, stopH2D);

    // === Kernel execution ===
    cudaEventRecord(startKernel);
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    // bit reversal
    bit_reverse<<<numBlocks, threadsPerBlock>>>(d_data, n, logn);
    cudaDeviceSynchronize();

    // stages
    for (int len = 2; len <= n; len <<= 1) {
        double ang = -2 * PI / len;
        int work_items = n / len * (len/2);
        int numBlocksStage = (work_items + threadsPerBlock - 1) / threadsPerBlock;
        fft_stage<<<numBlocksStage, threadsPerBlock>>>(d_data, n, len, ang);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stopKernel);
    cudaEventSynchronize(stopKernel);
    cudaEventElapsedTime(&kernelTime, startKernel, stopKernel);

    // === Device -> Host ===
    cudaEventRecord(startD2H);
    cudaMemcpy(h_data.data(), d_data, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    cudaEventRecord(stopD2H);
    cudaEventSynchronize(stopD2H);
    cudaEventElapsedTime(&d2hTime, startD2H, stopD2H);

    for (int i = 0; i < n; i++)
        data[i] = complex<double>(cuCreal(h_data[i]), cuCimag(h_data[i]));

    cudaFree(d_data);

    cudaEventDestroy(startH2D);
    cudaEventDestroy(stopH2D);
    cudaEventDestroy(startKernel);
    cudaEventDestroy(stopKernel);
    cudaEventDestroy(startD2H);
    cudaEventDestroy(stopD2H);
}

// === MAIN ===
int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <threads_per_block> <input_file> [num_runs]\n";
        return 1;
    }

    int threadsPerBlock = stoi(argv[1]);
    const char* filename = argv[2];
    int num_runs = (argc >= 4) ? stoi(argv[3]) : 1;

    cout << fixed << setprecision(4);

    // === Reading ===
    auto start_read = chrono::high_resolution_clock::now();
    ifstream ifs(filename);
    if (!ifs) {
        cerr << "Error opening file: " << filename << "\n";
        return 1;
    }

    vector<complex<double>> data;
    double real, imag;
    while (ifs >> real >> imag) data.emplace_back(real, imag);
    ifs.close();

    if (data.empty()) {
        cerr << "No data read from file.\n";
        return 1;
    }

    size_t n = data.size();
    if ((n & (n - 1)) != 0) {
        cerr << "FFT requires input size to be a power of 2. Got " << n << ".\n";
        return 1;
    }

    auto end_read = chrono::high_resolution_clock::now();
    auto duration_read = chrono::duration<double, std::milli>(end_read - start_read);
    cout << "[RESULTS] ReadingTime: " << duration_read.count() << "ms" << endl;

    // === Executions ===
    for (int r = 0; r < num_runs; r++) {
        vector<complex<double>> temp = data;

        auto start_exec = chrono::high_resolution_clock::now();
        float h2dTime, kernelTime, d2hTime;
        fft_cuda(temp, threadsPerBlock, h2dTime, kernelTime, d2hTime);
        auto end_exec = chrono::high_resolution_clock::now();

        auto duration_exec = chrono::duration<double, std::milli>(end_exec - start_exec);
        cout << "[RESULTS] ExecutionTime(run=" << (r+1) << "): " << duration_exec.count() << "ms\n";
        cout << "  (Details) Host->Device: " << h2dTime << "ms\n";
        cout << "  (Details) Kernel: " << kernelTime << "ms\n";
        cout << "  (Details) Device->Host: " << d2hTime << "ms\n";
    }

    auto end_all = chrono::high_resolution_clock::now();
    auto duration_total = chrono::duration<double, std::milli>(end_all - start_read);
    cout << "[RESULTS] TotalTime: " << duration_total.count() << "ms" << endl;

    return 0;
}