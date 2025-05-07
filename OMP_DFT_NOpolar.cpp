#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <omp.h>

const double PI = acos(-1);

std::vector<std::complex<double>> dft_fast(const std::vector<std::complex<double>>& input, int n_threads) {
    int n = input.size();
    std::vector<std::complex<double>> output(n);

    omp_set_num_threads(n_threads);
    // Parallelize both loops to improve load balancing
    #pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < n; ++k) {
        // Calcolo una sola volta il “passo” di rotazione per questo k
        double angle_step = -2 * PI * k / n;
        std::complex<double> W = { std::cos(angle_step), std::sin(angle_step) };
        std::complex<double> cur(1.0, 0.0);

        std::complex<double> sum(0.0, 0.0);
        for (int t = 0; t < n; ++t) {
            sum += input[t] * cur;
            cur *= W;     // aggiorno la rotazione senza richiami a sin/cos
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
    const char* filename = argv[2];
    const char* output_filename = "output.txt";

    std::ifstream ifs(filename);
    if (!ifs) {
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

    auto result = dft_fast(data, n_threads);

    // Stampa dei primi 10 output
    std::cout << "Primi 10 risultati DFT (no polar):\n";
    for (int i = 0; i < std::min(10, (int)result.size()); ++i) {
        std::cout << "  [" << i << "] "
                  << result[i].real() << " + "
                  << result[i].imag() << "i\n";
    }

    return 0;
}
