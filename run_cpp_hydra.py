import subprocess
import time
import csv
import os
import logging # Importa il modulo logging
import hydra
from omegaconf import DictConfig

# Configura un logger (Hydra lo gestirà automaticamente in parte)
log = logging.getLogger(__name__)

def run_cpp_program(executable_path: str, input_file: str, num_threads: int, csv_file: str):
    """
    Esegue un programma C++, misura il tempo di esecuzione e salva i risultati in un CSV.

    Args:
        executable_path: Percorso dell'eseguibile C++.
        input_file: Percorso del file di input per il programma C++.
        num_threads: Numero di thread da passare al programma C++.
        csv_file: Percorso del file CSV dove salvare i risultati.
    """
    # Costruisci il comando da eseguire
    command = [executable_path, str(num_threads), input_file]

    log.info(f"Esecuzione del comando: {' '.join(command)}")
    log.info(f"Uso di {num_threads} thread...")

    start_time = time.time()

    try:
        # Esegui il programma C++
        # check=True solleva CalledProcessError se il processo ritorna un codice != 0
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        end_time = time.time()
        execution_time = end_time - start_time

        log.info("Output del programma:")
        # Stampa l'output riga per riga per una migliore leggibilità nel log
        for line in result.stdout.splitlines():
             log.info(f"  [STDOUT] {line}")
        if result.stderr:
            log.warning("Output di STDERR (potrebbe contenere warning o messaggi informativi):")
            for line in result.stderr.splitlines():
                log.warning(f"  [STDERR] {line}")


        log.info(f"Tempo di esecuzione: {execution_time:.4f} secondi")

        # Scrivi i risultati nel file CSV
        # Hydra cambia la directory di lavoro, quindi usiamo hydra.utils.to_absolute_path
        # se vogliamo che il CSV sia sempre nello stesso posto rispetto alla locazione originale.
        # Altrimenti, verrà creato nella cartella di output di Hydra (outputs/...).
        # In questo esempio, lo creiamo nella cartella di output di Hydra.
        # output_csv_path = hydra.utils.to_absolute_path(csv_file) # Opzione per percorso assoluto
        output_csv_path = csv_file # Salva nella directory di output di Hydra

        file_exists = os.path.isfile(output_csv_path)
        # Usiamo 'a' (append) per aggiungere righe alle esecuzioni precedenti (utile con multirun)
        with open(output_csv_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Scrivi l'intestazione solo se il file non esiste o è vuoto
            if not file_exists or os.path.getsize(output_csv_path) == 0:
                writer.writerow(["Numero Thread", "Tempo di Esecuzione (s)", "File Input", "Eseguibile"])
            writer.writerow([num_threads, f"{execution_time:.4f}", input_file, executable_path])
        log.info(f"Risultati aggiunti a: {os.path.abspath(output_csv_path)}") # Mostra il percorso assoluto

    except subprocess.CalledProcessError as e:
        log.error("Errore nell'esecuzione del programma C++:")
        log.error(f"Comando: {' '.join(e.cmd)}")
        log.error(f"Codice di Ritorno: {e.returncode}")
        log.error("Output STDERR:")
        for line in e.stderr.splitlines():
            log.error(f"  [STDERR] {line}")
        # Potresti voler propagare l'errore o uscire
        # raise e # Rilancia l'eccezione se necessario
    except FileNotFoundError:
        log.error(f"Errore: Eseguibile non trovato a '{executable_path}'. Verifica il percorso.")
    except Exception as e:
        log.error(f"Errore inaspettato durante l'esecuzione: {e}")


# --- Configurazione Hydra ---
# config_path: Specifica la directory relativa allo script dove cercare i file di configurazione.
# config_name: Specifica il nome del file di configurazione principale (senza estensione .yaml).
# version_base=None: Necessario per compatibilità con le versioni più recenti di Hydra
#                    per mantenere il comportamento di cambio directory predefinito.
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Funzione principale orchestrata da Hydra.

    Args:
        cfg: Oggetto di configurazione caricato da Hydra (da config.yaml).
    """
    # Stampa la configurazione utilizzata (utile per il debug)
    log.info("Configurazione utilizzata:")
    # OmegaConf.to_yaml(cfg) è un modo pulito per stampare la config
    # log.info(OmegaConf.to_yaml(cfg)) # Richiede: from omegaconf import OmegaConf

    # Accedi ai parametri di configurazione tramite l'oggetto cfg
    executable = cfg.cpp_program.executable_path
    input_f = cfg.cpp_program.input_file
    threads = cfg.settings.num_threads
    output_csv = cfg.output.csv_file

    # Esegui la logica principale
    run_cpp_program(
        executable_path=executable,
        input_file=input_f,
        num_threads=threads,
        csv_file=output_csv
    )

if __name__ == "__main__":
    main()