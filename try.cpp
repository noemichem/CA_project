#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>

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

// === Iterative radix-2 FFT, single-threaded ===
void fft_iterative(vector<complex<double>>& a) {
    size_t n = a.size();
    bit_reverse(a);

    for (size_t len = 2; len <= n; len <<= 1) {
        double ang = -2 * PI / len;
        complex<double> wlen(cos(ang), sin(ang));

        for (size_t i = 0; i < n; i += len) {
            complex<double> w(1);
            for (size_t j = 0; j < len / 2; ++j) {
                complex<double> u = a[i + j];
                complex<double> v = a[i + j + len / 2] * w;
                a[i + j]         = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

// === MAIN ===
int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_file>\n";
        return 1;
    }

    const char* filename = argv[1];
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

    // Controllo che la lunghezza sia potenza di 2
    size_t n = data.size();
    if ((n & (n - 1)) != 0) {
        cerr << "FFT requires input size to be a power of 2. Got " << n << ".\n";
        return 1;
    }

    // Calcola la FFT
    fft_iterative(data);

    // Stampa i primi 10 risultati
    cout << "Primi 10 risultati FFT:\n";
    for (int i = 0; i < min<size_t>(10, data.size()); ++i) {
        cout << "  [" << i << "] "
             << data[i].real() << " + "
             << data[i].imag() << "i\n";
    }

    return 0;
}
