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

const double PI = acos(-1);

// === Bit reversal permutation ===
void bit_reverse(vector<complex<double>>& a) {
    size_t n = a.size();
    size_t j = 0;
    for (size_t i = 1; i < n; ++i) {
        size_t bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j)
            swap(a[i], a[j]);
    }
}

// === Iterative radix-2 FFT with optional OpenMP parallelization ===
void fft_iterative(vector<complex<double>>& a, int n_threads) {
    size_t n = a.size();
    bit_reverse(a);

    #pragma omp parallel num_threads(n_threads)
    {
        for (size_t len = 2; len <= n; len <<= 1) {
            double ang = -2 * PI / len;
            complex<double> wlen(cos(ang), sin(ang));

            #pragma omp for schedule(static) nowait
            for (size_t i = 0; i < n; i += len) {
                complex<double> w(1);
                for (size_t j = 0; j < len / 2; ++j) {
                    complex<double> u = a[i + j];
                    complex<double> v = a[i + j + len / 2] * w;
                    a[i + j] = u + v;
                    a[i + j + len / 2] = u - v;
                    w *= wlen;
                }
            }
            // implicit barrier qui solo se serve (rimuovendo nowait se c'è dipendenza)
        }
    }
}


// === MAIN ===
int main(int argc, char* argv[]) {
    
    auto start_sys = std::chrono::system_clock::now();
    std::time_t start_c = std::chrono::system_clock::to_time_t(start_sys);
    std::cout << "Inizio esecuzione: "
              << std::put_time(std::localtime(&start_c), "%Y-%m-%d %H:%M:%S")
              << '\n';

    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <num_threads> <input_file>\n";
        return 1;
    }

    int n_threads = stoi(argv[1]);
    const char* filename = argv[2];

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

    // Controllo se la lunghezza è potenza di 2
    size_t n = data.size();
    if ((n & (n - 1)) != 0) {
        cerr << "FFT requires input size to be a power of 2. Got " << n << ".\n";
        return 1;
    }

    // Esegui FFT
    fft_iterative(data, n_threads);

    auto end_sys = std::chrono::system_clock::now();
    std::time_t end_c = std::chrono::system_clock::to_time_t(end_sys);
    std::cout << "Fine esecuzione:   "
              << std::put_time(std::localtime(&end_c), "%Y-%m-%d %H:%M:%S")
              << '\n';

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_sys - start_sys);
    std::cout << "Durata totale:     "
              << elapsed.count() << " ms\n";


    return 0;
}