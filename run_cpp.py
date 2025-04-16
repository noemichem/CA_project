import subprocess
import time
import sys
import csv
import os

def run_cpp_program(executable_path, input_file, num_threads, csv_file="risultati.csv"):
    # Costruisci il comando da eseguire
    command = [executable_path, str(num_threads), input_file]

    print(f"Esecuzione del programma con {num_threads} thread...")

    start_time = time.time()

    try:
        # Esegui il programma C++
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        end_time = time.time()
        execution_time = end_time - start_time

        print("Output del programma:")
        print(result.stdout)

        print(f"\nTempo di esecuzione: {execution_time:.4f} secondi")

        # Scrivi i risultati nel file CSV
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Numero Thread", "Tempo di Esecuzione (s)"])
            writer.writerow([num_threads, f"{execution_time:.4f}"])

    except subprocess.CalledProcessError as e:
        print("Errore nell'esecuzione del programma:")
        print(e.stderr)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso: python run_cpp.py <path_eseguibile> <file_input> <num_thread>")
        sys.exit(1)

    executable_path = sys.argv[1]
    input_file = sys.argv[2]
    num_threads = int(sys.argv[3])

    run_cpp_program(executable_path, input_file, num_threads)
