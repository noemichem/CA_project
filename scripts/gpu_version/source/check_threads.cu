#include <iostream>
#include <cuda_runtime.h>

int main() {
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    std::cout << "Name: " << p.name << "\n";
    std::cout << "MultiProcessorCount: " << p.multiProcessorCount << "\n";
    std::cout << "MaxThreadsPerMultiProcessor: " << p.maxThreadsPerMultiProcessor << "\n";
    std::cout << "MaxThreadsPerBlock: " << p.maxThreadsPerBlock << "\n";
    std::cout << "WarpSize: " << p.warpSize << "\n";
    std::cout << "RegsPerBlock: " << p.regsPerBlock << "\n";
    std::cout << "SharedMemPerBlock: " << p.sharedMemPerBlock << "\n";
    return 0;
}
