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

// === GPU FFT (timing with bit-reversal separated) ===
void fft_cuda(vector<complex<double>>& data,
              int threadsPerBlock,
              float &totalExecTime, float &gpuTime,
              float &kernelTime, float &bitreverseTime,
              float &h2dTime, float &d2hTime) {
    int n = data.size();
    int logn = 0;
    while ((1 << logn) < n) ++logn;

    cuDoubleComplex* d_data;
    cudaMalloc(&d_data, n * sizeof(cuDoubleComplex));

    vector<cuDoubleComplex> h_data(n);
    for (int i = 0; i < n; i++)
        h_data[i] = make_cuDoubleComplex(data[i].real(), data[i].imag());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // === Host -> Device ===
    cudaEventRecord(start);
    cudaMemcpy(d_data, h_data.data(), n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2dTime, start, stop);

    // === Bit-reversal ===
    cudaEventRecord(start);
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    bit_reverse<<<numBlocks, threadsPerBlock>>>(d_data, n, logn);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&bitreverseTime, start, stop);

    // === FFT Stages ===
    cudaEventRecord(start);
    for (int len = 2; len <= n; len <<= 1) {
        double ang = -2 * PI / len;
        int work_items = n / len * (len/2);
        int numBlocksStage = (work_items + threadsPerBlock - 1) / threadsPerBlock;
        fft_stage<<<numBlocksStage, threadsPerBlock>>>(d_data, n, len, ang);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernelTime, start, stop);

    // === Device -> Host ===
    cudaEventRecord(start);
    cudaMemcpy(h_data.data(), d_data, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2hTime, start, stop);

    for (int i = 0; i < n; i++)
        data[i] = complex<double>(cuCreal(h_data[i]), cuCimag(h_data[i]));

    gpuTime       = bitreverseTime + kernelTime;
    totalExecTime = h2dTime + gpuTime + d2hTime;

    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// === MAIN ===
int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <threads_per_block> <input_file> [num_runs]\n";
        return 1;
    }

    int threadsPerBlock = stoi(argv[1]);
    const char* filename = argv[2];
    int numRuns = (argc >= 4) ? max(1, stoi(argv[3])) : 1;

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
    float readTime = chrono::duration<float, milli>(end_read - start_read).count();
    cout << "[RESULTS] ReadingTime: " << readTime << "ms\n";

    float totalTimeSum = 0;
    for (int run = 1; run <= numRuns; run++) {
        float totalExec=0, gpuTime=0, kernelTime=0, bitreverseTime=0, h2dTime=0, d2hTime=0;
        vector<complex<double>> temp = data;

        fft_cuda(temp, threadsPerBlock,
                 totalExec, gpuTime,
                 kernelTime, bitreverseTime,
                 h2dTime, d2hTime);

        cout << "[RESULTS] ExecutionTime(run=" << run << "): " << totalExec << "ms\n";
        cout << "  (Details) Host->Device: " << h2dTime << "ms\n";
        cout << "  (Details) Kernel: " << kernelTime << "ms\n";
        cout << "  (Details) Device->Host: " << d2hTime << "ms\n";
        cout << "  (Details) Bit-reversal: " << bitreverseTime << "ms\n";

        totalTimeSum += totalExec;
    }

    cout << "[RESULTS] TotalTime: " << (readTime + totalTimeSum) << "ms\n";
    if (numRuns > 1) {
        cout << "[RESULTS] AverageExecutionTime: " << (totalTimeSum / numRuns) << "ms\n";
    }

    return 0;
}
