#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <omp.h>

using namespace std;

static double PI = acos(-1);

// === Iterative FFT with bit-reversal and butterfly in a single parallel region ===
void fft_iterative_merged(vector<complex<double>>& a, int n_threads) {
    size_t n = a.size();
    int logn = 0;
    while ((size_t(1) << logn) < n) ++logn;

    omp_set_num_threads(n_threads);

    #pragma omp parallel
    {
        // Bit-reversal phase
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            size_t rev = 0;
            size_t x = i;
            for (int b = 0; b < logn; ++b) {
                rev = (rev << 1) | (x & 1);
                x >>= 1;
            }
            if (i < rev) swap(a[i], a[rev]);
        }

        // FFT levels loop (butterflies)
        for (size_t len = 2; len <= n; len <<= 1) {
            double ang = -2 * PI / len;
            complex<double> wlen(cos(ang), sin(ang));

            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; i += len) {
                complex<double> w(1);
                for (size_t j = 0; j < len/2; ++j) {
                    complex<double> u = a[i + j];
                    complex<double> v = a[i + j + len/2] * w;
                    a[i + j] = u + v;
                    a[i + j + len/2] = u - v;
                    w *= wlen;
                }
            }
        }
    }
}

// === MAIN ===
int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <num_threads> <input_file> [num_runs]\n";
        return 1;
    }

    int n_threads = stoi(argv[1]);
    const char* filename = argv[2];
    int num_runs = (argc >= 4) ? stoi(argv[3]) : 1;

    cout << fixed << setprecision(4);

    // === Reading Phase ===
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
    auto duration_read = chrono::duration_cast<chrono::milliseconds>(end_read - start_read);
    cout << "[RESULTS] ReadingTime: " << duration_read.count() << "ms" << endl;

    // === Algorithm Executions ===
    for (int r = 0; r < num_runs; ++r) {
        vector<complex<double>> temp_data = data; // copia per evitare modifiche successive
        auto start_exec = chrono::high_resolution_clock::now();
        fft_iterative_merged(temp_data, n_threads);
        auto end_exec = chrono::high_resolution_clock::now();

        auto duration_exec = chrono::duration_cast<chrono::milliseconds>(end_exec - start_exec);

        cout << "[RESULTS] ExecutionTime(run=" << (r+1) << "): " << duration_exec.count() << "ms" << endl;
    }
    
    auto end_all = chrono::high_resolution_clock::now();
    auto duration_total = chrono::duration_cast<chrono::milliseconds>(end_all - start_read);
    cout << "[RESULTS] TotalTime: " << duration_total.count() << "ms" << endl;

    return 0;
}