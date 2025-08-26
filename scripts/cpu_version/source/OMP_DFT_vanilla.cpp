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

// === Parallel DFT ===
vector<complex<double>> dft_parallel(const vector<complex<double>>& input, int n_threads) {
    int n = static_cast<int>(input.size());
    vector<complex<double>> output(n, {0.0, 0.0});

    omp_set_num_threads(n_threads);

    #pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < n; ++k) {
        for (int t = 0; t < n; ++t) {
            double angle = -2 * PI * t * k / n;
            output[k] += input[t] * polar(1.0, angle);
        }
    }
    return output;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <num_threads> <input_file> [num_runs]\n";
        return 1;
    }

    int n_threads = stoi(argv[1]);
    const char* filename = argv[2];
    int num_runs = (argc >= 4) ? stoi(argv[3]) : 1;

    cout << fixed << setprecision(4);

    // === Reading Phase ===
    auto start_read = chrono::high_resolution_clock::now();

    ifstream ifs(filename);
    if (!ifs) {
        cerr << "Error opening file: " << filename << "\n";
        return 1;
    }

    vector<complex<double>> data;
    double real, imag;
    while (ifs >> real >> imag) data.emplace_back(real, imag);
    ifs.close();

    if (data.empty()) {
        cerr << "No data read from file.\n";
        return 1;
    }

    auto end_read = chrono::high_resolution_clock::now();
    auto duration_read = chrono::duration_cast<chrono::milliseconds>(end_read - start_read);
    cout << "[RESULTS] ReadingTime: " << duration_read.count() << "ms" << endl;

    // === Algorithm Executions ===
    for (int r = 0; r < num_runs; ++r) {
        auto start_exec = chrono::high_resolution_clock::now();
        auto result = dft_parallel(data, n_threads);
        auto end_exec = chrono::high_resolution_clock::now();

        auto duration_exec = chrono::duration_cast<chrono::milliseconds>(end_exec - start_exec);

        cout << "[RESULTS] ExecutionTime(run=" << (r+1) << "): " << duration_exec.count() << "ms" << endl;
    }

    auto end_all = chrono::high_resolution_clock::now();
    auto duration_total = chrono::duration_cast<chrono::milliseconds>(end_all - start_read);
    cout << "[RESULTS] TotalTime: " << duration_total.count() << "ms" << endl;

    return 0;
}