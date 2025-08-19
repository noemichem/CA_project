// Ottimizzazione di DFT: uso FFT

#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>

using namespace std;

const double PI = acos(-1);

// === KERNEL FFT: esegue un singolo "stage" ===
__global__ void fft_stage(cuFloatComplex* data, int n, int step) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int halfStep = step / 2;

    if (tid < n / 2) {
        int i = (tid / halfStep) * step + (tid % halfStep);

        cuFloatComplex u = data[i];
        cuFloatComplex v = data[i + halfStep];

        float angle = -2.0f * M_PI * (tid % halfStep) / step;
        cuFloatComplex w = make_cuFloatComplex(cosf(angle), sinf(angle));

        cuFloatComplex t = make_cuFloatComplex(
            v.x * w.x - v.y * w.y,
            v.x * w.y + v.y * w.x
        );

        data[i]           = make_cuFloatComplex(u.x + t.x, u.y + t.y);
        data[i + halfStep]= make_cuFloatComplex(u.x - t.x, u.y - t.y);
    }
}

// === GPU FFT Iterativa ===
void fft_gpu(vector<complex<float>>& input, int threadsPerBlock,
             float& avgTimeMs, float& gflops) {
    int N = input.size();
    size_t size = N * sizeof(cuFloatComplex);

    // Copia dati su device
    cuFloatComplex *d_data;
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, input.data(), size, cudaMemcpyHostToDevice);

    // Timing con CUDA events
    const int NUM_RUNS = 5;
    float totalTime = 0.0f;

    for (int run = 0; run < NUM_RUNS; run++) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // FFT iterativa per log2(N) stadi
        for (int step = 2; step <= N; step <<= 1) {
            int butterflies = N / 2;
            int blocks = (butterflies + threadsPerBlock - 1) / threadsPerBlock;
            fft_stage<<<blocks, threadsPerBlock>>>(d_data, N, step);
            cudaDeviceSynchronize();
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        totalTime += ms;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    avgTimeMs = totalTime / NUM_RUNS;

    // Calcolo GFLOPS (circa 5*N*log2(N) operazioni)
    double ops = 5.0 * N * log2((double)N);
    gflops = (ops / (avgTimeMs / 1000.0)) / 1e9;

    // Pulizia
    cudaFree(d_data);
}

// === MAIN ===
int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <threads_per_block> <input_file>\n";
        return 1;
    }

    int threadsPerBlock = stoi(argv[1]);
    const char* filename = argv[2];

    ifstream ifs(filename);
    if (!ifs) {
        cerr << "Error opening file.\n";
        return 1;
    }

    vector<complex<float>> data;
    float real, imag;
    while (ifs >> real >> imag) {
        data.emplace_back(real, imag);
    }
    ifs.close();

    // Controllo se N è potenza di 2
    int N = data.size();
    if ((N & (N - 1)) != 0) {
        cerr << "FFT requires power-of-2 size.\n";
        return 1;
    }

    // Lancia FFT GPU
    float avgTimeMs, gflops;
    fft_gpu(data, threadsPerBlock, avgTimeMs, gflops);

    // Output dei risultati (per CSV)
    cout << N << "," << threadsPerBlock << ","
         << avgTimeMs << "," << gflops << endl;

    return 0;
}

