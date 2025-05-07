#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <omp.h>

const double PI = acos(-1);

std::vector<std::complex<double>> dft_parallel(const std::vector<std::complex<double>>& input, int n_threads) {
    int n = static_cast<int>(input.size());
    std::vector<std::complex<double>> output(n);

    omp_set_num_threads(n_threads);
    // Parallelize both loops to improve load balancing
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int k = 0; k < n; ++k) {
        for (int t = 0; t < n; ++t) {
            double angle = -2 * PI * t * k / n;
            output[k] += (input[t] * std::polar(1.0, angle));
        }
    }
    return output;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <num_threads> <input_file>\n";
        return 1;
    }

    int n_threads = std::stoi(argv[1]);
    const char* filename = argv[2];
    const char* output_filename = "output.txt";

    std::ifstream ifs(filename);
    if (!ifs) {
        std::cout << "Sono qui1" << "\n";
        std::cerr << "Error opening file: " << filename << "\n";
        return 1;
    }


    // Read all samples (real and imaginary parts) until EOF
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


    auto start = std::chrono::high_resolution_clock::now();
    auto result = dft_parallel(data, n_threads);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed Time: " << elapsed.count() << "s\n";


    return 0;
}
