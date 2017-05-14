#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t __rst = call; \
        if (__rst != cudaSuccess) { \
            fprintf(stderr, "cuda %s failed: %s\n", #call, \
                    cudaGetErrorString(__rst)); \
            __builtin_trap(); \
        } \
    } while(0)

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("arch=compute_%d%d,code=sm_%d%d\n",
            prop.major, prop.minor,
            prop.major, prop.minor);
}
