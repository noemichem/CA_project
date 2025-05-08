#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <omp.h>

const double PI = std::acos(-1);

// FFT ricorsiva senza alcuna normalizzazione
void FFT(std::vector<std::complex<double>>& a) {
    size_t n = a.size();
    if (n <= 1) return;

    // split pari/dispari
    std::vector<std::complex<double>> even(n/2), odd(n/2);
    for (size_t i = 0; i < n/2; ++i) {
        even[i] = a[2*i];
        odd[i]  = a[2*i + 1];
    }
    
    // Parallelize le chiamate ricorsive se grande
    #pragma omp task shared(even) if(n > 1024)
    FFT(even);
    #pragma omp task shared(odd) if(n > 1024)
    FFT(odd);

    #pragma omp taskwait

    // ricombino con i twiddle factors senza scalare
    #pragma omp parallel for schedule(dynamic)
    for (size_t k = 0; k < n/2; ++k) {
        auto t = std::polar(1.0, -2 * PI * k / n) * odd[k];
        a[k]       = even[k] + t;
        a[k + n/2] = even[k] - t;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <num_threads> <input_file>\n";
        return 1;
    }

    int n_threads = std::stoi(argv[1]);
    const char* filename = argv[2];

    omp_set_num_threads(n_threads);

    std::ifstream ifs(filename);
    if (!ifs) {
        std::cerr << "Error opening file: " << filename << "\n";
        return 1;
    }

    std::vector<std::complex<double>> data;
    double real, imag;
    while (ifs >> real >> imag) {
        data.emplace_back(real, imag);
    }
    ifs.close();

    if (data.empty()) {
        std::cerr << "No data read from file: " << filename << "\n";
        return 1;
    }

    // Esegui FFT in regione parallel
    #pragma omp parallel
    {
        #pragma omp single
        FFT(data);
    }

    // Stampa dei primi 10 output senza alcuna divisione
    std::cout << "Primi 10 risultati FFT senza normalizzazione:\n";
    for (int i = 0; i < std::min<size_t>(10, data.size()); ++i) {
        std::cout << "  [" << i << "] "
                  << data[i].real() << " + "
                  << data[i].imag() << "i\n";
    }

    return 0;
}
