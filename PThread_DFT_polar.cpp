#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <pthread.h>
#include <errno.h>
#include <atomic>

const double PI = acos(-1);
typedef std::complex<double> Complex;

struct ThreadData {
    int thread_id;
    int n_threads;
    int n;
    const Complex* input;
    Complex* result;
    Complex* partial_sums; // flattened [n_threads][n]
    std::atomic<int>* counter;
};

void* dft_thread(void* arg) {
    auto* data = static_cast<ThreadData*>(arg);
    int id = data->thread_id;
    int n = data->n;
    const Complex* in = data->input;
    Complex* partial = data->partial_sums + id * n;
    std::atomic<int>& counter = *data->counter;

    int total = n * n;
    int idx;
    while ((idx = counter.fetch_add(1, std::memory_order_relaxed)) < total) {
        int k = idx / n;
        int t = idx % n;
        double angle = -2 * PI * t * k / n;
        partial[k] += in[t] * std::polar(1.0, angle);
    }
    return nullptr;
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
    while (ifs >> real >> imag) data.emplace_back(real, imag);
    ifs.close();

    int n = static_cast<int>(data.size());
    if (n == 0) {
        std::cerr << "No data read from file: " << filename << "\n";
        return 1;
    }

    std::vector<Complex> result(n);
    std::vector<Complex> partial_sums(n_threads * n);
    for (auto& v : partial_sums) v = Complex(0.0, 0.0);

    std::atomic<int> counter(0);
    std::vector<pthread_t> threads(n_threads);
    std::vector<ThreadData> td(n_threads);

    for (int i = 0; i < n_threads; ++i) {
        td[i] = {i, n_threads, n, data.data(), result.data(), partial_sums.data(), &counter};
        pthread_create(&threads[i], nullptr, dft_thread, &td[i]);
    }
    for (int i = 0; i < n_threads; ++i) pthread_join(threads[i], nullptr);

    // Reduction: sum partial sums into result
    for (int k = 0; k < n; ++k) {
        Complex sum(0.0, 0.0);
        for (int i = 0; i < n_threads; ++i) {
            sum += partial_sums[i * n + k];
        }
        result[k] = sum;
    }

    return 0;
}
