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

static constexpr double PI = acos(-1);

// === FFT iterativa con bit-reverse e butterfly in un'unica regione parallela ===
void fft_iterative_merged(vector<complex<double>>& a, int n_threads) {
    size_t n = a.size();
    // calcolo log2(n)
    int logn = 0;
    while ((size_t(1) << logn) < n) ++logn;

    // Imposto il numero di thread globale
    omp_set_num_threads(n_threads);

    #pragma omp parallel
    {
        // 1) Fase di bit-reverse parallelizzata
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
        // Barriera implicita: tutti i thread terminano la bit-reverse

        // 2) Ciclo sui livelli di FFT (butterflies)
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
            // Barriera implicita al termine di ogni livello
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

    // === Fase di Lettura ===
    auto start_read_sys = chrono::system_clock::now();
    time_t start_read_c = chrono::system_clock::to_time_t(start_read_sys);
    cout << "Inizio Lettura: "
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
    cout << "Fine Lettura:    "
         << put_time(localtime(&end_read_c), "%Y-%m-%d %H:%M:%S")
         << '\n';

    auto duration_read = chrono::duration_cast<chrono::milliseconds>(end_read_sys - start_read_sys);
    cout << "Durata Lettura:  " << duration_read.count() << " ms\n";

    for (size_t i = 1; i < 21; ++i) {
        

        // === Fase di Elaborazione FFT ===
        auto start_fft_sys = chrono::system_clock::now();
        time_t start_fft_c = chrono::system_clock::to_time_t(start_fft_sys);
        cout << "Inizio FFT:      "
            << put_time(localtime(&start_fft_c), "%Y-%m-%d %H:%M:%S")
            << '\n'
            << "No. di thread: " << i << '\n';

        fft_iterative_merged(data, i);

        auto end_fft_sys = chrono::system_clock::now();
        time_t end_fft_c = chrono::system_clock::to_time_t(end_fft_sys);
        cout << "Fine FFT:        "
            << put_time(localtime(&end_fft_c), "%Y-%m-%d %H:%M:%S")
            << '\n';

        auto duration_fft = chrono::duration_cast<chrono::milliseconds>(end_fft_sys - start_fft_sys);
        cout << "Durata FFT:      " << duration_fft.count() << " ms\n";

        // === Tempo Totale ===
        auto duration_total = chrono::duration_cast<chrono::milliseconds>(end_fft_sys - start_read_sys);
        cout << "Durata Totale:   " << duration_total.count() << " ms\n";
    }
    return 0;
}
