#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <pthread.h>
#include <atomic>

const double PI = std::acos(-1);
using Complex = std::complex<double>;

// Dati passati a ogni thread
struct ThreadData {
    const Complex* input;       // puntatore all'array di input
    Complex* output;            // puntatore all'array di output
    int n;                      // dimensione della DFT
    std::atomic<int>* counter;  // contatore atomico per scheduling dinamico
};

// Funzione eseguita da ciascun thread
void* dft_worker(void* arg) {
    ThreadData* td = static_cast<ThreadData*>(arg);
    const Complex* in = td->input;
    Complex* out = td->output;
    int n = td->n;
    std::atomic<int>& ctr = *td->counter;

    int k;
    while ((k = ctr.fetch_add(1, std::memory_order_relaxed)) < n) {
        // Calcolo del coefficiente W_k = exp(-2πi k / n)
        double angle_step = -2 * PI * k / n;
        Complex Wk = { std::cos(angle_step), std::sin(angle_step) };
        Complex cur(1.0, 0.0);
        Complex sum(0.0, 0.0);

        // Somma per t = 0..n-1
        for (int t = 0; t < n; ++t) {
            sum += in[t] * cur;
            cur *= Wk;
        }
        out[k] = sum;
    }

    return nullptr;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Uso: " << argv[0] << " <num_threads> <input_file>\n";
        return 1;
    }

    int num_threads = std::stoi(argv[1]);
    const char* filename = argv[2];

    // Lettura del file di input
    std::ifstream ifs(filename);
    if (!ifs) {
        std::cerr << "Errore nell'apertura di " << filename << "\n";
        return 1;
    }
    std::vector<Complex> data;
    double real, imag;
    while (ifs >> real >> imag) {
        data.emplace_back(real, imag);
    }
    ifs.close();

    int n = static_cast<int>(data.size());
    if (n == 0) {
        std::cerr << "Nessun dato letto da " << filename << "\n";
        return 1;
    }

    // Preparazione dell'output e del contatore atomico
    std::vector<Complex> output(n);
    std::atomic<int> counter(0);

    // Creazione dei thread
    std::vector<pthread_t> threads(num_threads);
    std::vector<ThreadData> tdata(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        tdata[i] = { data.data(), output.data(), n, &counter };
        if (pthread_create(&threads[i], nullptr, dft_worker, &tdata[i]) != 0) {
            std::cerr << "Errore nella creazione del thread " << i << "\n";
            return 1;
        }
    }

    // Join dei thread
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }

    // Stampa dei primi 10 risultati
    std::cout << "Primi 10 risultati DFT (pthread, scheduling dinamico):\n";
    for (int i = 0; i < std::min(10, n); ++i) {
        std::cout << "  [" << i << "] "
                  << output[i].real() << " + "
                  << output[i].imag() << "i\n";
    }

    return 0;
}
