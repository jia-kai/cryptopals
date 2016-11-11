#include <stddef.h>
#include <stdint.h>

static inline size_t _do_popcount(const uint8_t* arr, size_t len) {
    size_t ret = 0, i, len_ul = len / sizeof(unsigned long);
    const unsigned long *pul = (const unsigned long*)arr;
    for (i = 0; i < len_ul; ++ i) {
        ret += __builtin_popcountl(pul[i]);
    }
    for (i = len_ul * sizeof(unsigned long); i < len; ++ i) {
        ret += __builtin_popcount(arr[i]);
    }
    return ret;
}
