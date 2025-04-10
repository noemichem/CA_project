/*Take in input a file with words, divide it into 10 chuncks, then count the letters in each of the chunks, then put them together to output the number of each letter*/ 

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cctype>

class LetterCounter {
private:
    std::string content;
    std::vector<std::string> chunks;
    std::map<char, int> totalLetterCount;

public:
    LetterCounter(const std::string& filename) {
        readFile(filename);
        divideChunks(5);
        countLetters();
    }

    void readFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file) {
            std::cerr << "Could not open the file.\n";
            return;
        }

        std::string line;
        while (std::getline(file, line)) {
            content += line;
        }
        file.close();
    }

    void divideChunks(int numChunks) {
        size_t chunkSize = content.size() / numChunks;
        for (int i = 0; i < numChunks; ++i) {
            size_t start = i * chunkSize;
            if (i == numChunks - 1) {
                chunks.push_back(content.substr(start)); // last chunk gets the rest
            } else {
                chunks.push_back(content.substr(start, chunkSize));
            }
        }
    }

    void countLetters() {
        for (const auto& chunk : chunks) {
            for (char ch : chunk) {
                if (std::isalpha(ch)) {
                    ch = std::toupper(ch);
                    totalLetterCount[ch]++;
                }
            }
        }
    }

    void printTotalLetterCounts() const {
        std::cout << "Total letter counts (A-Z):\n";
        for (char ch = 'A'; ch <= 'Z'; ++ch) {
            auto it = totalLetterCount.find(ch);
            if (it != totalLetterCount.end()) {
                std::cout << ch << ": " << it->second << "\n";
            }
        }
    }
};

int main() {
    LetterCounter analyzer("Seagul_Italian.txt");
    analyzer.printTotalLetterCounts();
    return 0;
}


