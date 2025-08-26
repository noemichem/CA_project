#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <omp.h>

using namespace std;

const double PI = acos(-1);

// === Optimized Parallel DFT ===
vector<complex<double>> dft_fast(const vector<complex<double>>& input, int n_threads) {
    int n = input.size();
    vector<complex<double>> output(n);

    omp_set_num_threads(n_threads);

    #pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < n; ++k) {
        double angle_step = -2 * PI * k / n;
        complex<double> W(cos(angle_step), sin(angle_step));
        complex<double> cur(1.0, 0.0);

        complex<double> sum(0.0, 0.0);
        for (int t = 0; t < n; ++t) {
            sum += input[t] * cur;
            cur *= W;
        }
        output[k] = sum;
    }
    return output;
}

// === MAIN ===
int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <num_threads> <input_file> [num_runs]\n";
        return 1;
    }

    int n_threads = stoi(argv[1]);
    const char* filename = argv[2];
    int num_runs = (argc >= 4) ? stoi(argv[3]) : 1;

    // === Reading Phase ===
    auto start_read = chrono::high_resolution_clock::now();

    ifstream ifs(filename);
    if (!ifs) {
        cerr << "Error opening file: " << filename << "\n";
        return 1;
    }

    vector<complex<double>> data;
    double real, imag;
    while (ifs >> real >> imag) {
        data.emplace_back(real, imag);
    }
    ifs.close();

    if (data.empty()) {
        cerr << "No data read from file: " << filename << "\n";
        return 1;
    }

    auto end_read = chrono::high_resolution_clock::now();
    auto duration_read = chrono::duration_cast<chrono::milliseconds>(end_read - start_read);

    cout << fixed << setprecision(4);
    cout << "[RESULTS] ReadingTime: " << duration_read.count() << "ms" << endl;

    // === DFT Processing Phase ===
    for (int r = 0; r < num_runs; ++r) {
        auto start_dft = chrono::high_resolution_clock::now();
        auto result = dft_fast(data, n_threads);
        auto end_dft = chrono::high_resolution_clock::now();

        auto duration_dft = chrono::duration_cast<chrono::milliseconds>(end_dft - start_dft);

        cout << "[RESULTS] ExecutionTime(run=" << (r+1) << "): " << duration_dft.count() << "ms" << endl;
    }

    auto end_all = chrono::high_resolution_clock::now();
    auto duration_total = chrono::duration_cast<chrono::milliseconds>(end_all - start_read);
    cout << "[RESULTS] TotalTime: " << duration_total.count() << "ms" << endl;

    return 0;
}