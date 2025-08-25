#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <ctime>
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

// === MAIN ===
int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <num_threads> <input_file>\n";
        return 1;
    }

    int n_threads = stoi(argv[1]);
    const char* filename = argv[2];

    // === Reading Phase ===
    auto start_read_sys = chrono::system_clock::now();
    time_t start_read_c = chrono::system_clock::to_time_t(start_read_sys);
    cout << "Start Reading: "
         << put_time(localtime(&start_read_c), "%Y-%m-%d %H:%M:%S")
         << '\n';

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

    auto end_read_sys = chrono::system_clock::now();
    time_t end_read_c = chrono::system_clock::to_time_t(end_read_sys);
    cout << "End Reading:    "
         << put_time(localtime(&end_read_c), "%Y-%m-%d %H:%M:%S")
         << '\n';

    auto duration_read = chrono::duration_cast<chrono::milliseconds>(end_read_sys - start_read_sys);
    cout << "Reading Time:   " << duration_read.count() << " ms\n";

    // === DFT Processing Phase ===
    auto start_dft_sys = chrono::system_clock::now();
    time_t start_dft_c = chrono::system_clock::to_time_t(start_dft_sys);
    cout << "Start DFT:      "
         << put_time(localtime(&start_dft_c), "%Y-%m-%d %H:%M:%S")
         << '\n'
         << "Number of threads: " << n_threads << '\n';

    auto result = dft_parallel(data, n_threads);

    auto end_dft_sys = chrono::system_clock::now();
    time_t end_dft_c = chrono::system_clock::to_time_t(end_dft_sys);
    cout << "End DFT:        "
         << put_time(localtime(&end_dft_c), "%Y-%m-%d %H:%M:%S")
         << '\n';

    auto duration_dft = chrono::duration_cast<chrono::milliseconds>(end_dft_sys - start_dft_sys);
    cout << "DFT Time:       " << duration_dft.count() << " ms\n";

    // === Total Time ===
    auto duration_total = chrono::duration_cast<chrono::milliseconds>(end_dft_sys - start_read_sys);
    cout << "Total Time:     " << duration_total.count() << " ms\n";

    return 0;
}
