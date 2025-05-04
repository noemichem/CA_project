#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <random>
#include <chrono>

const double PI = acos(-1);

// DFT diretta (complessità O(n^2))
std::vector<std::complex<double>> dft(const std::vector<std::complex<double>>& input) {
    int n = input.size();
    std::vector<std::complex<double>> output(n);

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
    // Dimensione del vettore di input (default 5000, modificabile da linea di comando)
    int n = 50000;
    if (argc > 1) {
        n = std::stoi(argv[1]);
    }

    // Generazione di valori complessi random
    std::vector<std::complex<double>> input;
    input.reserve(n);

    std::mt19937_64 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < n; ++i) {
        double real = dist(rng);
        double imag = dist(rng);
        input.emplace_back(real, imag);
    }

    // Calcolo della DFT
    auto output = dft(input);

    (void)output; // Evita warning su variabile inutilizzata
    return 0;
}


    std::cout << "DFT result:\n";
    for (const auto& val : output) {
        std::cout << val << '\n';
    }



    // cronometro per misurare il tempo di esecuzione della DFT
    auto start = std::chrono::high_resolution_clock::now();
    
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "DFT time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
            << " microseconds\n";


    return 0;
}
