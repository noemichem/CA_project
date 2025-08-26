import subprocess
import csv
import os
import logging
import time
import hydra
import re
from omegaconf import DictConfig

CPU_CSV = "results/tables/CPU_details.csv"
GPU_CSV = "results/tables/GPU_details.csv"

# ---------------------------
# Logging configuration
# ---------------------------
LOG_DIR = "log"
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, "execution.log")

log = logging.getLogger(__name__)

log.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# File handler
file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Add handlers to logger
log.addHandler(console_handler)
log.addHandler(file_handler)


# ---------------------------
# Function to run C++ program
# ---------------------------
def run_executable(executable, num_threads, input_file, num_runs):

    cmd = [executable, str(num_threads), input_file, str(num_runs)]
    print(f"Eseguendo: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    except Exception as e:
        print("Errore durante l'esecuzione:", e)
        return None

    # Stampa output
    if result.stdout:
        for line in result.stdout.splitlines():
            print("[C++ STDOUT]", line)
    if result.stderr:
        print("[C++ STDERR]", result.stderr.strip())

    if result.returncode != 0:
        print(f"Programma terminato con codice {result.returncode}")
        return None

    # Parsing output
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
# Function to save to CSV
# ---------------------------
def save_to_csv(times, executable, num_threads, input_file, num_runs, device):
    os.makedirs("results/tables", exist_ok=True)

    if device == "cuda":
        is_cuda = True
    
    csv_path = GPU_CSV if is_cuda else CPU_CSV
    exe_name = os.path.basename(executable)
    file_name = os.path.basename(input_file)

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        # Header
        if not file_exists:
            if is_cuda:
                header = ["Num Execution", "Threads per Block", "Input File", "Run", "Execution Time (ms)", "Executable"]
            else:
                header = ["Num Execution", "Num Threads", "Input File", "Run", "Execution Time (ms)", "Executable"]
            writer.writerow(header)
            num_exec = num_runs
        else:
            num_exec = num_runs

        # Scrivi righe
        for run_id, val in sorted(times["ExecutionTimes"]):
            if is_cuda:
                # Se l'eseguibile è cuFFT, Threads per Block = None
                threads_value = None if "cuFFT" in exe_name else num_threads
                row = [num_exec, threads_value, file_name, run_id, val, exe_name]
            else:
                row = [num_exec, num_threads, file_name, run_id, val, exe_name]
            writer.writerow(row)

    print(f"{len(times['ExecutionTimes'])} risultati salvati in {csv_path} (Num Execution = {num_exec})")


# ---------------------------
# Hydra main function
# ---------------------------
@hydra.main(config_path="", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main function to execute the C++ program multiple times and log results.
    """
    log.info("Using configuration:")
    
    # Load configuration from Hydra
    executable = cfg.cpp_program.executable_path
    threads = cfg.settings.num_threads
    device = cfg.settings.device
    input_file = cfg.settings.input_file
    iteration = cfg.settings.for_limit

    
    log.info(f"Running command: {' '.join([executable, str(threads), input_file])}")
    log.info(f"Using {threads} threads with input file: {input_file}")

    if device == "cuda":
        csv_path = GPU_CSV
        if not os.path.exists(csv_path):
            num_runs = 1
        else:
            with open(csv_path, "r") as fr:
                lines = list(csv.reader(fr))
                if len(lines) > 1:
                    last_num_exec = max(int(row[0]) for row in lines[1:])
                    num_runs = last_num_exec + 1
    else:
        csv_path = CPU_CSV
        if not os.path.exists(csv_path):
            num_runs = 1
        else:
            with open(csv_path, "r") as fr:
                lines = list(csv.reader(fr))
                if len(lines) > 1:
                    last_num_exec = max(int(row[0]) for row in lines[1:])
                    num_runs = last_num_exec + 1

    # Run the program multiple times (adjust range as needed)
    for i in range(iteration):  # Run 'iteration' times
        log.info(f"Run {i + 1}/{iteration}")
        times = run_executable(executable, threads, input_file, num_runs)
        save_to_csv(times, executable, threads, input_file, device)
    else:
        log.error("No successful executions.")

# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    main()