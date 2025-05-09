import subprocess
import csv
import os
import logging
import hydra
import time
import math
from omegaconf import DictConfig

log = logging.getLogger(__name__)

def next_power_of_two(n):
    return 1 if n == 0 else 2**math.ceil(math.log2(n))

def pad_input_file_if_needed(original_file: str, executable: str) -> str:
    if "OMP_FFT.exe" not in executable.lower():
        return original_file  # Nessun padding necessario

    temp_file = "temp_padded_input.txt"
    data = []

    with open(original_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                real, imag = map(float, parts)
                data.append((real, imag))

    original_len = len(data)
    padded_len = next_power_of_two(original_len)

    if original_len == padded_len:
        return original_file  # Già potenza di 2

    # Padding con zeri
    with open(temp_file, 'w', encoding='utf-8') as f:
        for real, imag in data:
            f.write(f"{real} {imag}\n")
        for _ in range(padded_len - original_len):
            f.write("0.0 0.0\n")

    log.info(f"File {original_file} padded da {original_len} a {padded_len} elementi")
    return temp_file

def run_cpp_program(executable_path: str, num_threads: int, input_file: str) -> float:
    padded_file = pad_input_file_if_needed(input_file, executable_path)

    command = [executable_path, str(num_threads), padded_file]

    try:
        start_ns = time.perf_counter_ns()
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8', timeout=None)
        end_ns = time.perf_counter_ns()
        execution_time = (end_ns - start_ns) / 1_000_000  # in millisecondi
        log.info(f"Tempo di esecuzione: {execution_time:.4f} ms")

        return execution_time

    except subprocess.CalledProcessError as e:
        log.error(f"Errore durante l'esecuzione del comando: {e}")
        log.error(f"Codice di ritorno: {e.returncode}")
        log.error(f"Output standard: {e.stdout}")
        log.error(f"Errore: {e.stderr}")
    except FileNotFoundError:
        log.error(f"Eseguibile non trovato a '{executable_path}'")
    except Exception as e:
        log.error(f"Errore inaspettato: {e}")

    return None

def salva_dettagli_csv(csv_file: str, results: list[float], num_threads: int, input_file: str, executable: str):
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists or os.path.getsize(csv_file) == 0:
            writer.writerow(["Numero Esecuzione", "Numero Thread", "File di Input", "Tempo di Esecuzione (ms)", "Eseguibile"])
        for i, exec_time in enumerate(results):
            writer.writerow([i + 1, num_threads, input_file, f"{exec_time:.4f}", executable])

def salva_media_csv(csv_file: str, media: float, num_threads: int, input_file: str, executable: str):
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists or os.path.getsize(csv_file) == 0:
            writer.writerow(["Numero Thread", "File di Input", "Media Tempo di Esecuzione (ms)", "Eseguibile"])
        writer.writerow([num_threads, input_file, f"{media:.4f}", executable])

@hydra.main(config_path="", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info("Configurazione utilizzata:")
    executable = cfg.cpp_program.executable_path
    threads = cfg.settings.num_threads
    input_file = cfg.settings.input_file
    dettagli_csv = cfg.output.dettagli_csv
    media_csv = cfg.output.media_csv

    execution_times = []

    log.info(f"Esecuzione del comando: {' '.join([executable, str(threads), input_file])}")
    log.info(f"Uso di {threads} thread con file di input: {input_file}")
    for i in range(1):
        log.info(f"Esecuzione numero {i+1}/10")
        exec_time = run_cpp_program(executable, threads, input_file)
        if exec_time is not None:
            execution_times.append(exec_time)

    if execution_times:
        media = sum(execution_times) / len(execution_times)
        log.info(f"Tempo medio di esecuzione: {media:.4f} millisecondi")
        salva_dettagli_csv(dettagli_csv, execution_times, threads, input_file, executable)
        salva_media_csv(media_csv, media, threads, input_file, executable)
    else:
        log.error("Nessuna esecuzione completata con successo.")

if __name__ == "__main__":
    main()
