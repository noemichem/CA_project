#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <fstream>
#include <omp.h>

const double PI = acos(-1);

// DFT diretta (complessità O(n^2))
std::vector<std::complex<double>> dft(const std::vector<std::complex<double>>& input) {
    int n = input.size();
    std::vector<std::complex<double>> output(n);

    #pragma omp parallel for
    for (int k = 0; k < n; ++k) {
        std::complex<double> sum(0.0, 0.0);
        for (int t = 0; t < n; ++t) {
            double angle = -2 * PI * t * k / n;
            sum += input[t] * std::polar(1.0, angle);
        }
        output[k] = sum;
    }
    return output;
}

int main(int argc, char* argv[]) {

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <num_threads> <input_file>\n";
        return 1;
    }

    int n_threads = std::stoi(argv[1]);
    std::string input_filename = argv[2];

    omp_set_num_threads(n_threads);

    std::vector<std::complex<double>> input;
    std::ifstream infile(input_filename);

    if (!infile) {
        std::cerr << "Errore nell'aprire il file: " << input_filename << "\n";
        return 1;
    }

    double real, imag;
    while (infile >> real >> imag) {
        input.emplace_back(real, imag);
    }

    auto output = dft(input);

    return 0;
}