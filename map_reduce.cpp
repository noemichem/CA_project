#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <algorithm>
#include <cctype>
#include <string>

// Funzione Map: conta ogni lettera in una stringa e produce una lista di coppie (carattere, 1)
std::vector<std::pair<char, int>> map(const std::string& text) {
    std::vector<std::pair<char, int>> intermediate;
    for (char c : text) {
        if (std::isalpha(c)) {
            c = std::tolower(c);  // Normalizziamo le lettere
            intermediate.push_back(std::make_pair(c, 1));
        }
    }
    return intermediate;
}

// Funzione Reduce: raggruppa e somma tutte le occorrenze per ciascuna lettera
std::map<char, int> reduce(const std::vector<std::pair<char, int>>& mapped_data) {
    std::map<char, int> letter_count;
    for (const auto& pair : mapped_data) {
        letter_count[pair.first] += pair.second;
    }
    return letter_count;
}

int main() {
    std::ifstream infile("Seagul_Italian.txt");
    if (!infile) {
        std::cerr << "Errore nell'apertura del file input.txt" << std::endl;
        return 1;
    }

    std::string line;
    std::string input;

    while (std::getline(infile, line)) {
        input += line + '\n';  // Accumula tutto il testo
    }

    infile.close();

    // Step 1: Map
    std::vector<std::pair<char, int>> mapped = map(input);

    // Step 2: Reduce
    std::map<char, int> reduced = reduce(mapped);

    // Output dei risultati
    for (const auto& entry : reduced) {
        std::cout << entry.first << ": " << entry.second << std::endl;
    }

    return 0;
}