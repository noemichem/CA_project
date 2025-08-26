import argparse
import subprocess
import csv
import os
import re

CPU_CSV = "results/tables/CPU_details.csv"
GPU_CSV = "results/tables/GPU_details.csv"

def run_executable(executable, num_threads, input_file, num_runs=1, is_cuda=False):
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


def save_to_csv(times, executable, num_threads, input_file, is_cuda=False):
    os.makedirs("results/tables", exist_ok=True)
    
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
            num_exec = 1
        else:
            # Leggi ultima colonna Num Execution
            with open(csv_path, "r") as fr:
                lines = list(csv.reader(fr))
                if len(lines) > 1:
                    last_num_exec = max(int(row[0]) for row in lines[1:])
                    num_exec = last_num_exec + 1
                else:
                    num_exec = 1

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wrapper per eseguibili CPU/CUDA e salvataggio tempi su CSV")
    parser.add_argument("executable", help="Percorso dell'eseguibile")
    parser.add_argument("num_threads", type=int, help="Numero di thread (CPU) o Threads per Block (CUDA)")
    parser.add_argument("input_file", help="File di input")
    parser.add_argument("num_runs", type=int, help="Numero di esecuzioni")
    parser.add_argument("--cuda", action="store_true", help="Esegui come script CUDA/cuFFT")

    args = parser.parse_args()

    times = run_executable(args.executable, args.num_threads, args.input_file, args.num_runs, is_cuda=args.cuda)
    if times:
        save_to_csv(times, args.executable, args.num_threads, args.input_file, is_cuda=args.cuda)