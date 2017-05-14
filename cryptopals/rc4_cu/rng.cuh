#pragma once

#include <stdint.h>

// see https://en.wikipedia.org/wiki/Xorshift
__host__ __device__ __forceinline__ uint64_t xorshift128plus(uint64_t *s) {
    uint64_t x = s[0];
    uint64_t const y = s[1];
    s[0] = y;
    x ^= x << 23; // a
    s[1] = x ^ y ^ (x >> 17) ^ (y >> 26); // b, c
    return s[1] + y;
}

// vim: ft=cuda syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
