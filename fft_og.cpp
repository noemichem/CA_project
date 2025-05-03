#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

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

int main() {
    // Esempio di input: 4 numeri reali (0, 1, 2, 3)
    std::vector<std::complex<double>> input = {0, 1, 2, 3};
    auto output = dft(input);

    std::cout << "DFT result:\n";
    for (const auto& val : output) {
        std::cout << val << '\n';
    }

    return 0;
}
