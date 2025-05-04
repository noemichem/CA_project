import subprocess
import time
import csv
import os
import logging
import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)

def run_cpp_program(executable_path: str, num_threads: int, input_file: str, csv_file: str):
    command = [executable_path, str(num_threads), input_file]

    log.info(f"Esecuzione del comando: {' '.join(command)}")
    log.info(f"Uso di {num_threads} thread con file di input: {input_file}")

    start_time = time.time()

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8', timeout=None)
        end_time = time.time()
        execution_time = end_time - start_time

        log.info("Output del programma:")
        for line in result.stdout.splitlines():
            log.info(f"  [STDOUT] {line}")
        if result.stderr:
            log.warning("Output di STDERR:")
            for line in result.stderr.splitlines():
                log.warning(f"  [STDERR] {line}")

        output_csv_path = csv_file
        file_exists = os.path.isfile(output_csv_path)
        with open(output_csv_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists or os.path.getsize(output_csv_path) == 0:
                writer.writerow(["Numero Thread", "File di Input", "Tempo di Esecuzione (s)", "Eseguibile"])
            writer.writerow([num_threads, input_file, f"{execution_time:.4f}", executable_path])
        log.info(f"Risultati salvati in: {os.path.abspath(output_csv_path)}")

    except subprocess.CalledProcessError as e:
        log.error(f"Errore durante l'esecuzione del comando: {e}")
        log.error(f"Codice di ritorno: {e.returncode}")
        log.error(f"Output standard: {e.stdout}")
        log.error(f"Errore: {e.stderr}")
    except FileNotFoundError:
        log.error(f"Eseguibile non trovato a '{executable_path}'")
    except Exception as e:
        log.error(f"Errore inaspettato: {e}")

@hydra.main(config_path="", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info("Configurazione utilizzata:")
    executable = cfg.cpp_program.executable_path
    threads = cfg.settings.num_threads
    input_file = cfg.settings.input_file
    output_csv = cfg.output.csv_file

    run_cpp_program(
        executable_path=executable,
        num_threads=threads,
        input_file=input_file,
        csv_file=output_csv
    )

if __name__ == "__main__":
    main()