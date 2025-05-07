#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>

using namespace std;
using namespace std::chrono;

using Complex = complex<double>;
using Signal = vector<Complex>;

const double PI = acos(-1);

void FFT(Signal& a) {
    size_t n = a.size();
    if (n <= 1) return;

    Signal even(n/2), odd(n/2);
    for (size_t i = 0; i < n / 2; ++i) {
        even[i] = a[i*2];
        odd[i] = a[i*2 + 1];
    }
    

    FFT(even);
    FFT(odd);

    for (size_t k = 0; k < n / 2; ++k) {
        Complex t = polar(1.0, -2 * PI * k / n) * odd[k];
        a[k] = even[k] + t;
        a[k + n/2] = even[k] - t;
    }
}

int main() {
    const size_t N = 1024;
    Signal input(N);
    for (size_t i = 0; i < N; ++i)
        input[i] = sin(2 * PI * i / N); // segnale test

    auto start = high_resolution_clock::now();
    FFT(input);
    auto end = high_resolution_clock::now();

    cout << "FFT computation took: "
         << duration_cast<milliseconds>(end - start).count()
         << " ms\n";

    return 0;
}
