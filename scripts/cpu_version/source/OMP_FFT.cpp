#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <omp.h>

using namespace std;

static double PI = acos(-1);

// === Iterative FFT with bit-reversal and butterfly in a single parallel region ===
void fft_iterative_merged(vector<complex<double>>& a, int n_threads) {
    size_t n = a.size();
    // compute log2(n)
    int logn = 0;
    while ((size_t(1) << logn) < n) ++logn;

    // Set the global number of threads
    omp_set_num_threads(n_threads);

    #pragma omp parallel
    {
        // 1) Bit-reversal phase (parallelized)
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            size_t rev = 0;
            size_t x = i;
            for (int b = 0; b < logn; ++b) {
                rev = (rev << 1) | (x & 1);
                x >>= 1;
            }
            if (i < rev) {
                swap(a[i], a[rev]);
            }
        }
        // Implicit barrier: all threads finish bit-reversal

        // 2) FFT levels loop (butterflies)
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
        cerr << "Usage: " << argv[0] << " <num_threads> <input_file>\n";
        return 1;
    }

    int n_threads = stoi(argv[1]);
    const char* filename = argv[2];

    // === Reading Phase ===
    auto start_read_sys = chrono::system_clock::now();
    time_t start_read_c = chrono::system_clock::to_time_t(start_read_sys);
    cout << "Start Reading: "
         << put_time(localtime(&start_read_c), "%Y-%m-%d %H:%M:%S")
         << '\n';

    ifstream ifs(filename);
    if (!ifs) {
        cerr << "Error opening file: " << filename << "\n";
        return 1;
    }

    vector<complex<double>> data;
    double real, imag;
    while (ifs >> real >> imag) {
        data.emplace_back(real, imag);
    }
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

    auto end_read_sys = chrono::system_clock::now();
    time_t end_read_c = chrono::system_clock::to_time_t(end_read_sys);
    cout << "End Reading:    "
         << put_time(localtime(&end_read_c), "%Y-%m-%d %H:%M:%S")
         << '\n';

    auto duration_read = chrono::duration_cast<chrono::milliseconds>(end_read_sys - start_read_sys);
    cout << "Reading Time:   " << duration_read.count() << " ms\n";

    // === FFT Processing Phase ===
    auto start_fft_sys = chrono::system_clock::now();
    time_t start_fft_c = chrono::system_clock::to_time_t(start_fft_sys);
    cout << "Start FFT:      "
        << put_time(localtime(&start_fft_c), "%Y-%m-%d %H:%M:%S")
        << '\n'
        << "Number of threads: " << n_threads << '\n';

    fft_iterative_merged(data, n_threads);

    auto end_fft_sys = chrono::system_clock::now();
    time_t end_fft_c = chrono::system_clock::to_time_t(end_fft_sys);
    cout << "End FFT:        "
        << put_time(localtime(&end_fft_c), "%Y-%m-%d %H:%M:%S")
        << '\n';

    auto duration_fft = chrono::duration_cast<chrono::milliseconds>(end_fft_sys - start_fft_sys);
    cout << "FFT Time:       " << duration_fft.count() << " ms\n";

    // === Total Time ===
    auto duration_total = chrono::duration_cast<chrono::milliseconds>(end_fft_sys - start_read_sys);
    cout << "Total Time:     " << duration_total.count() << " ms\n";

    return 0;
}