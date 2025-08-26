import argparse
import subprocess
import csv
import os
import re

# Paths for storing CSV results
CPU_CSV = "results/tables/CPU_details.csv"
GPU_CSV = "results/tables/GPU_details.csv"

# ---------------------------
# Function to run the C++ executable
# ---------------------------
def run_executable(executable, num_threads, input_file, num_runs=1):
    """
    Run the C++ executable with the given parameters:
      executable num_threads input_file num_runs
    num_runs = number of times the algorithm is repeated internally
    """
    cmd = [executable, str(num_threads), input_file, str(num_runs)]
    print(f"Executing: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    except Exception as e:
        print("Execution error:", e)
        return None

    # Print stdout and stderr
    if result.stdout:
        for line in result.stdout.splitlines():
            print("[C++ STDOUT]", line)
    if result.stderr:
        print("[C++ STDERR]", result.stderr.strip())

    if result.returncode != 0:
        print(f"Program exited with code {result.returncode}")
        return None

    # Parse output for timings
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
# Function to save results to CSV
# ---------------------------
def save_to_csv(times, executable, num_threads, input_file, is_cuda=False):
    """
    Save parsed execution times to the CSV file.
    Handles CPU and GPU (CUDA) separately.
    """
    os.makedirs("results/tables", exist_ok=True)
    
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
            num_exec = 1
        else:
            # Read last Num Execution from CSV
            with open(csv_path, "r") as fr:
                lines = list(csv.reader(fr))
                if len(lines) > 1:
                    last_num_exec = max(int(row[0]) for row in lines[1:])
                    num_exec = last_num_exec + 1
                else:
                    num_exec = 1

        # Write rows for each execution run
        for run_id, val in sorted(times["ExecutionTimes"]):
            if is_cuda:
                # For cuFFT, Threads per Block = None
                threads_value = None if "cuFFT" in exe_name else num_threads
                row = [num_exec, threads_value, file_name, run_id, val, exe_name]
            else:
                row = [num_exec, num_threads, file_name, run_id, val, exe_name]
            writer.writerow(row)

    print(f"{len(times['ExecutionTimes'])} results saved in {csv_path} (Num Execution = {num_exec})")

# ---------------------------
# Main CLI entry point
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wrapper for CPU/CUDA executables with CSV timing output")
    parser.add_argument("executable", help="Path to the executable")
    parser.add_argument("num_threads", type=int, help="Number of threads (CPU) or threads per block (CUDA)")
    parser.add_argument("input_file", help="Input file")
    parser.add_argument("num_runs", type=int, help="Number of internal runs")
    parser.add_argument("--cuda", action="store_true", help="Run as CUDA/cuFFT executable")

    args = parser.parse_args()

    times = run_executable(args.executable, args.num_threads, args.input_file, args.num_runs)
    if times:
        save_to_csv(times, args.executable, args.num_threads, args.input_file, is_cuda=args.cuda)