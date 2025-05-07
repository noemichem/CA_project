#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <pthread.h>

const double PI = acos(-1);
using Complex = std::complex<double>;
using Signal = std::vector<Complex>;

// Parametri per passare ai thread
struct FFTArgs {
    Signal* data;
    int depth;
    int max_depth;
};

// FFT ricorsiva (versione sequenziale)
void fft_recursive(Signal& a, int depth, int max_depth);

// Thread entry point
void* fft_thread_entry(void* arg) {
    FFTArgs* args = static_cast<FFTArgs*>(arg);
    fft_recursive(*args->data, args->depth, args->max_depth);
    delete args;
    return nullptr;
}

// Funzione ricorsiva che gestisce pthread
void fft_recursive(Signal& a, int depth, int max_depth) {
    size_t n = a.size();
    if (n <= 1) return;

    Signal even(n / 2), odd(n / 2);
    for (size_t i = 0; i < n / 2; ++i) {
        even[i] = a[i * 2];
        odd[i] = a[i * 2 + 1];
    }

    if (depth < max_depth) {
        pthread_t t1, t2;

        // Alloca argomenti per entrambi i thread
        FFTArgs* arg1 = new FFTArgs{&even, depth + 1, max_depth};
        FFTArgs* arg2 = new FFTArgs{&odd, depth + 1, max_depth};

        pthread_create(&t1, nullptr, fft_thread_entry, arg1);
        pthread_create(&t2, nullptr, fft_thread_entry, arg2);

        pthread_join(t1, nullptr);
        pthread_join(t2, nullptr);
    } else {
        fft_recursive(even, depth + 1, max_depth);
        fft_recursive(odd, depth + 1, max_depth);
    }

    for (size_t k = 0; k < n / 2; ++k) {
        Complex t = std::polar(1.0, -2 * PI * k / n) * odd[k];
        a[k] = even[k] + t;
        a[k + n / 2] = even[k] - t;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <num_threads> <input_file>\n";
        return 1;
    }

    int n_threads = std::stoi(argv[1]);
    const char* filename = argv[2];

    std::ifstream ifs(filename);
    if (!ifs) {
        std::cerr << "Error opening file: " << filename << "\n";
        return 1;
    }

    std::vector<Complex> data;
    double real, imag;
    while (ifs >> real >> imag) {
        data.emplace_back(real, imag);
    }
    ifs.close();

    if (data.empty()) {
        std::cerr << "No data read from file: " << filename << "\n";
        return 1;
    }

    if ((data.size() & (data.size() - 1)) != 0) {
        std::cerr << "FFT requires input size to be a power of 2. Got " << data.size() << "\n";
        return 1;
    }

    // Calcola profondità massima di threading (log2(n_threads))
    int max_depth = static_cast<int>(std::log2(n_threads));

    fft_recursive(data, 0, max_depth);

    return 0;
}
