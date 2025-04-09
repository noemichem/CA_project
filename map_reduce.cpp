#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <algorithm>
#include <cctype>
#include <thread>
#include <mutex>

// Mutex per accesso sicuro alla struttura condivisa
std::mutex mtx;

// Funzione Map: conta ogni lettera in una porzione di testo
void map_chunk(const std::string& text_chunk, std::vector<std::pair<char, int>>& result) {
    std::vector<std::pair<char, int>> local_result;
    for (char c : text_chunk) {
        if (std::isalpha(c)) {
            c = std::tolower(c);
            local_result.push_back(std::make_pair(c, 1));
        }
    }

    // Protezione con mutex per scrivere nel risultato globale
    std::lock_guard<std::mutex> lock(mtx);
    result.insert(result.end(), local_result.begin(), local_result.end());
}

// Funzione Reduce: aggrega i risultati
std::map<char, int> reduce(const std::vector<std::pair<char, int>>& mapped_data) {
    std::map<char, int> letter_count;
    for (const auto& pair : mapped_data) {
        letter_count[pair.first] += pair.second;
    }
    return letter_count;
}

int main() {
    std::ifstream infile("LordOfTheRings_Italian.txt");
    if (!infile) {
        std::cerr << "Errore nell'apertura del file LordOfTheRings_Italian.txt" << std::endl;
        return 1;
    }

    std::string line;
    std::string input;

    while (std::getline(infile, line)) {
        input += line + '\n';
    }
    infile.close();

    // Numero di thread (puoi scegliere anche std::thread::hardware_concurrency())
    const int num_threads = 4;
    std::vector<std::thread> threads;
    std::vector<std::pair<char, int>> mapped_result;

    size_t chunk_size = input.size() / num_threads;

    // Lancio dei thread per la fase di Map
    for (int i = 0; i < num_threads; ++i) {
        size_t start = i * chunk_size;
        size_t end = (i == num_threads - 1) ? input.size() : start + chunk_size;
        std::string chunk = input.substr(start, end - start);
        threads.emplace_back(map_chunk, chunk, std::ref(mapped_result));
    }

    // Attesa della fine dei thread
    for (auto& t : threads) {
        t.join();
    }

    // Fase Reduce
    std::map<char, int> reduced = reduce(mapped_result);

    // Output finale
    for (const auto& entry : reduced) {
        std::cout << entry.first << ": " << entry.second << std::endl;
    }

    return 0;
}
