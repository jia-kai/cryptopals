#pragma once

#include <memory>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(_expr) do { \
    cudaError_t _err = (_expr); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "cuda call failed: err=%s(%d) expr=%s at %s:%d\n", \
                cudaGetErrorString(_err), static_cast<int>(_err), #_expr, \
                __FILE__, __LINE__); \
        abort(); \
    } \
} while(0)

class CUDAMemReleaser {
    public:
        void operator()(void *ptr) {
            if (ptr) {
                CUDA_CHECK(cudaFree(ptr));
            }
        }
};

//! allocate array containing \p nr_elem elements of type T on CUDA
template<typename T>
std::unique_ptr<T, CUDAMemReleaser> cuda_new_arr(size_t nr_elem) {
    void *ptr;
    CUDA_CHECK(cudaMalloc(&ptr, sizeof(T) * nr_elem));
    return std::unique_ptr<T, CUDAMemReleaser>(static_cast<T*>(ptr));
}

// vim: ft=cuda syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
