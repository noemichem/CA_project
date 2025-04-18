#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <cctype>
#include <algorithm>

std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
std::mutex mtx;

class LetterCounter {
private:
    std::string content;
    std::vector<std::string> chunks;
    std::map<char, int> totalLetterCount;

    const int fixedChunks = 5;

public:
    LetterCounter(const std::string& filename, int numThreads) {
        startTimer();
        readFile(filename);
        divideChunks(fixedChunks);
        countLettersMultiThread(numThreads);
        stopTimer();
    }

    void startTimer() {
        startTime = std::chrono::high_resolution_clock::now();
    }

    void stopTimer() {
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        std::cout << "Execution time: " << duration << " ms\n";
    }

    void readFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file) {
            std::cerr << "Could not open the file.\n";
            exit(1);
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
                chunks.push_back(content.substr(start));
            }
            else {
                chunks.push_back(content.substr(start, chunkSize));
            }
        }
    }

    void processChunkRange(int startIdx, int endIdx) {
        std::map<char, int> localCount;

        for (int i = startIdx; i <= endIdx; ++i) {
            for (char ch : chunks[i]) {
                unsigned char uch = static_cast<unsigned char>(ch);
                if (std::isalpha(uch)) {
                    ch = std::toupper(uch);
                    localCount[ch]++;
                }
            }
        }

        std::lock_guard<std::mutex> lock(mtx);
        for (const auto& pair : localCount) {
            totalLetterCount[pair.first] += pair.second;
        }
    }

    void countLettersMultiThread(int numThreads) {
        int actualThreads = std::min(numThreads, fixedChunks);
        int chunksPerThread = fixedChunks / actualThreads;
        int remainder = fixedChunks % actualThreads;

        std::vector<std::thread> threads;
        int currentChunk = 0;

        for (int i = 0; i < actualThreads; ++i) {
            int chunksThisThread = chunksPerThread + (i < remainder ? 1 : 0);
            int start = currentChunk;
            int end = currentChunk + chunksThisThread - 1;

            threads.emplace_back(&LetterCounter::processChunkRange, this, start, end);
            currentChunk = end + 1;
        }

        for (auto& t : threads) {
            t.join();
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

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <num_threads> <input_file>\n";
        return 1;
    }

    int numThreads = std::stoi(argv[1]);
    std::string filename = argv[2];

    LetterCounter analyzer(filename, numThreads);
    analyzer.printTotalLetterCounts();

    return 0;
}
