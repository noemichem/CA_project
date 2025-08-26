import subprocess
import csv
import os
import logging
import hydra
import re
from omegaconf import DictConfig

# Paths for storing CSV results
CPU_CSV = "results/tables/CPU_details.csv"
GPU_CSV = "results/tables/GPU_details.csv"

# ---------------------------
# Logging configuration
# ---------------------------
log = logging.getLogger(__name__)

# ---------------------------
# Run C++ executable
# ---------------------------
def run_executable(executable, num_threads, input_file, inner_runs):
    """
    Run the C++ program with given parameters:
    executable num_threads input_file inner_runs
    inner_runs = how many times the algorithm is repeated internally
    """
    cmd = [executable, str(num_threads), input_file, str(inner_runs)]

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception as e:
        log.error(f"Execution error: {e}")
        return None

    # Log stdout and stderr
    if result.stdout:
        for line in result.stdout.splitlines():
            log.info(f"[C++ STDOUT] {line}")
    if result.stderr:
        log.error(f"[C++ STDERR] {result.stderr.strip()}")

    if result.returncode != 0:
        log.error(f"Program exited with code {result.returncode}")
        return None

    # Parse execution times from output
    output = result.stdout.splitlines()
    times = {"ReadingTime": None, "TotalTime": None, "ExecutionTimes": []}

    for line in output:
        if line.startswith("[RESULTS] ReadingTime:"):
            m = re.search(r"([\d.]+)ms", line)
            if m:
                times["ReadingTime"] = float(m.group(1))
        elif line.startswith("[RESULTS] ExecutionTime"):
            m_run = re.search(r"run=(\d+)", line)
            m_val = re.search(r"([\d.]+)ms", line)
            if m_run and m_val:
                run_id = int(m_run.group(1))
                val = float(m_val.group(1))
                times["ExecutionTimes"].append((run_id, val))
        elif line.startswith("[RESULTS] TotalTime:"):
            m = re.search(r"([\d.]+)ms", line)
            if m:
                times["TotalTime"] = float(m.group(1))

    return times

# ---------------------------
# Save results to CSV
# ---------------------------
def save_to_csv(times, executable, num_threads, input_file, inner_runs, device, num_execution):
    """
    Save the parsed execution times to the CSV file.
    Handles CPU and GPU (CUDA) separately.
    """
    os.makedirs("results/tables", exist_ok=True)
    is_cuda = device == "cuda"

    csv_path = GPU_CSV if is_cuda else CPU_CSV
    exe_name = os.path.basename(executable)
    file_name = os.path.basename(input_file)

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        # Write header if file does not exist
        if not file_exists:
            header = ["Num Execution", "Threads per Block" if is_cuda else "Num Threads",
                      "Input File", "Run", "Execution Time (ms)", "Executable"]
            writer.writerow(header)

        # Write each execution time
        for run_id, val in sorted(times["ExecutionTimes"]):
            threads_value = None if is_cuda and "cuFFT" in exe_name else num_threads
            row = [num_execution, threads_value, file_name, run_id, val, exe_name]
            writer.writerow(row)

    log.info(f"{len(times['ExecutionTimes'])} results saved in {csv_path} (Num Execution = {num_execution})")

# ---------------------------
# Hydra main function
# ---------------------------
@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Load parameters from YAML
    executable = cfg.cpp_program.executable_path
    threads = cfg.settings.num_threads
    device = cfg.settings.device
    input_file = cfg.settings.input_file
    inner_runs = cfg.settings.num_run

    log.info(f"Running: {executable} with {threads} threads{' per block' if device == 'cuda' else ''}, file {input_file}, inner_runs={inner_runs}")

    # Determine num_execution by reading the existing CSV
    if device == "cuda":
        csv_path = GPU_CSV
    elif device == "cpu":
        csv_path = CPU_CSV
    else:
        log.error(f"Device '{device}' not recognized. Use 'cpu' or 'cuda'.")
        return

    if not os.path.exists(csv_path):
        num_execution = 1
    else:
        with open(csv_path, "r") as fr:
            lines = list(csv.reader(fr))
            last_num_exec = max(int(row[0]) for row in lines[1:])
            num_execution = last_num_exec + 1

    # Execute once (Hydra will handle sweeping combinations)
    times = run_executable(executable, threads, input_file, inner_runs)
    if times:
        save_to_csv(times, executable, threads, input_file, inner_runs, device, num_execution)
    else:
        log.error("Execution failed, no results saved.")

if __name__ == "__main__":
    main()