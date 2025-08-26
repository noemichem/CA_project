# CPU & GPU DFT/FFT Benchmarking ![Python](https://img.shields.io/badge/python-3.8%2B-blue) ![C++](https://img.shields.io/badge/C%2B%2B-17-brightgreen) ![MIT License](https://img.shields.io/badge/license-MIT-lightgrey)

A complete framework for benchmarking **Discrete Fourier Transform (DFT) and Fast Fourier Transform (FFT)** on CPU and GPU. Compare performance, scalability, and speedup of different implementations with single runs or multiruns using Hydra.

---

## Quick Start (3 Steps)

### ⚠️ Important Note

All scripts **must be executed from the root folder of the project**.
Even if it might technically work from other directories, the project has been tested only when run from the root folder.

### 1️⃣ Clone & Setup

```bash
git clone <repo_url>
cd <project_root>
pip install pandas matplotlib hydra-core omegaconf
```

### 2️⃣ Generate Data

```bash
python data/scripts/generate_complex.py 1024
python data/scripts/generate_pow2_complex.py 10
```

### 3️⃣ Compile & Run

**CPU Compilation:**

```powershell
cd scripts/cpu_version
.\OMP_compile.ps1
cd ../..  # Return to root folder
```

**GPU Compilation:**

```powershell
cd scripts/gpu_version
.\CUDA_compile.ps1
cd ../..  # Return to root folder
```

**Single Run:**

```bash
# CPU
python scripts/run_cpp.py OMP_DFT_optimized.exe 8 data/numbers_1024.txt 1

# GPU
python scripts/run_cpp.py CUDA_fft.exe 256 data/numbers_1048576.txt 1 --cuda
```

**Hydra-based Multirun:**

```bash
python scripts/run_cpp_hydra.py -m
```

> Always execute these commands **from the project root** to ensure correct file paths and log saving.

---

## Repository Structure

```
+---config
|       config.yaml         # Hydra parameters for single/multirun
+---data
|   |   numbers_*.txt
|   \---scripts
|           generate_complex.py
|           generate_pow2_complex.py
+---logs
|   +---multirun
|   \---outputs
+---results
|   +---plots
|   +---scripts
|           CPU_plot_results.ipynb
|   \---tables
|           CPU_details.csv
|           CPU_mean.csv
|           GPU_details.csv
\---scripts
    |   run_cpp.py
    |   run_cpp_hydra.py
    +---cpu_version
    |   |   OMP_compile.ps1
    |   +---build
    |   \---source
    \---gpu_version
        |   CUDA_compile.ps1
        |   CUDA_profile.ps1
        +---build
        +---profiling
        \---source
```

---

## CPU Implementation

* Multi-threaded DFT/FFT using OpenMP.
* Source: `scripts/cpu_version/source/*.cpp`
* Libraries:

```cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <omp.h>
```

* Compile: `OMP_compile.ps1`
* Execute:

```bash
<executable_cpu> <num_threads> <input_file> [num_runs]
```

---

## GPU Implementation

* CUDA-based DFT/FFT and cuFFT.
* Source: `scripts/gpu_version/source/*.cu`
* Libraries:

```cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <chrono>
#include <cufft.h>
```

* Compile: `CUDA_compile.ps1`
* Profiling: `CUDA_profile.ps1` → `.ncu-rep` reports in `profiling/`
* Execute:

```bash
<executable_gpu> <threads_per_block> <input_file> [num_runs]
```

---

## Analyze Results

* Use `results/scripts/CPU_plot_results.ipynb` to generate plots of execution times and speedups.

* CSV results:

  * CPU: `Num Execution,Num Threads,Input File,Run,Execution Time (ms),Executable`
  * GPU: `Num Execution,Threads per Block,Input File,Run,Execution Time (ms),Executable`

* Kernel: Anaconda3 + Python 3.11

* Required Python libraries: `pandas`, `matplotlib`, `hydra-core`, `omegaconf`

---

## License

This project is licensed under the **MIT License** – see [LICENSE](LICENSE) for details.
