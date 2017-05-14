#pragma once

#include <stdint.h>

template<uint32_t KEY_LEN>
__device__ __forceinline__ void rc4_key_sched(
        uint8_t *state, const uint8_t *key) {
    for (int i = 0; i < 256; ++ i) {
        state[i] = i;
    }
    uint8_t j = 0;
    for (int i = 0; i < 256; ++ i) {
        j = j + state[i] + key[i % KEY_LEN];
        uint8_t tmp = state[i];
        state[i] = state[j];
        state[j] = tmp;
    }
}

__device__ __forceinline__ uint8_t rc4_next(
        uint8_t *state, uint8_t &i, uint8_t &j) {
    i += 1;
    j += state[i];
    uint8_t tmp = state[i];
    state[i] = state[j];
    state[j] = tmp;
    return state[static_cast<uint8_t>(state[i] + state[j])];
}

// vim: ft=cuda syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}
