#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <random>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_size> <output_file>\n";
        return 1;
    }

    int n = std::stoi(argv[1]);
    std::string filename = argv[2];

    std::ofstream outfile(filename);
    if (!outfile) {
        std::cerr << "Failed to open output file: " << filename << "\n";
        return 1;
    }

    std::mt19937_64 rng(42); // fixed seed for reproducibility
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < n; ++i) {
        double real = dist(rng);
        double imag = dist(rng);
        outfile << real << " " << imag << "\n";
    }

    outfile.close();
    std::cout << "Generated " << n << " complex numbers in " << filename << "\n";

    return 0;
}