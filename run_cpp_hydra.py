import subprocess
import csv
import os
import logging
import time
import hydra
from omegaconf import DictConfig

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
def run_cpp_program(executable_path: str, num_threads: int, input_file: str) -> float:
    """
    Runs the C++ executable with the specified number of threads and input file.
    Measures and returns execution time in milliseconds.
    """
    command = [executable_path, str(num_threads), input_file]

    try:
        start_ns = time.perf_counter_ns()
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8', timeout=None)
        end_ns = time.perf_counter_ns()
        execution_time = (end_ns - start_ns) / 1_000_000  # convert to milliseconds
        log.info(f"Execution time: {execution_time:.4f} ms")
        return execution_time

    except subprocess.CalledProcessError as e:
        log.error(f"Error executing command: {e}")
        log.error(f"Return code: {e.returncode}")
        log.error(f"Standard output: {e.stdout}")
        log.error(f"Error output: {e.stderr}")
    except FileNotFoundError:
        log.error(f"Executable not found at '{executable_path}'")
    except Exception as e:
        log.error(f"Unexpected error: {e}")

    return None


# ---------------------------
# Function to save individual execution details to CSV
# ---------------------------
def save_details_csv(csv_file: str, results: list[float], num_threads: int, input_file: str, executable: str):
    """
    Saves the execution times of each run to a CSV file.
    """
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists or os.path.getsize(csv_file) == 0:
            writer.writerow(["Run Number", "Threads", "Input File", "Execution Time (ms)", "Executable"])
        for i, exec_time in enumerate(results):
            writer.writerow([i + 1, num_threads, input_file, f"{exec_time:.4f}", executable])


# ---------------------------
# Function to save average execution time to CSV
# ---------------------------
def save_average_csv(csv_file: str, average: float, num_threads: int, input_file: str, executable: str):
    """
    Saves the average execution time to a CSV file.
    """
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists or os.path.getsize(csv_file) == 0:
            writer.writerow(["Threads", "Input File", "Average Execution Time (ms)", "Executable"])
        writer.writerow([num_threads, input_file, f"{average:.4f}", executable])


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
    input_file = cfg.settings.input_file
    details_csv = cfg.output.details_csv
    average_csv = cfg.output.average_csv

    execution_times = []

    log.info(f"Running command: {' '.join([executable, str(threads), input_file])}")
    log.info(f"Using {threads} threads with input file: {input_file}")

    # Run the program multiple times (adjust range as needed)
    for i in range(10):  # Run 10 times
        log.info(f"Run {i + 1}/10")
        exec_time = run_cpp_program(executable, threads, input_file)
        if exec_time is not None:
            execution_times.append(exec_time)

    # Save results if at least one run succeeded
    if execution_times:
        average_time = sum(execution_times) / len(execution_times)
        log.info(f"Average execution time: {average_time:.4f} ms")
        save_details_csv(details_csv, execution_times, threads, input_file, executable)
        save_average_csv(average_csv, average_time, threads, input_file, executable)
    else:
        log.error("No successful executions.")

# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    main()