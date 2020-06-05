//
// Compile:
//
// $ nvcc list_gpus.cu -o list_gpus
//
//

#include <cuda.h>
#include <curand_kernel.h>
#include <stdio.h>


int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    int devidx, ccmaj, ccmin, sms;
    size_t globmem, BinM = 1024*1024;
    char *name;
    
    for (devidx = 0; devidx < deviceCount; ++devidx) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, devidx);

        name =  deviceProp.name;
        ccmaj = deviceProp.major;  // Compute Capability major number
        ccmin = deviceProp.minor;  // Compute Capability minor number
        sms =   deviceProp.multiProcessorCount; // Streaming Multiprocessors #
        globmem = deviceProp.totalGlobalMem/BinM; // Global Memory, MBytes
        
        printf("DevIdx %d, %s, CompCapbl %d.%d, SM# %d, GlobMem %ld MiB\n", 
               devidx, name, ccmaj, ccmin, sms, globmem);
    }    
}
