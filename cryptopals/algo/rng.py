# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractproperty
import numpy as np

class MersenneTwister(metaclass=ABCMeta):
    """see https://en.wikipedia.org/wiki/Mersenne_Twister"""

    @abstractproperty
    def dtype(self):
        """data type, word size"""

    @abstractproperty
    def _nmr(self):
        """
        * n: degree of recurrence
        * m: middle word, an offset used in the recurrence relation defining
          the series x, 1 <= m < n
        * r: separation point of one word, or the number of bits of the lower
          bitmask, 0 <= r < w
        """

    @abstractproperty
    def _abcstudl(self):
        """
        * a: coefficients of the rational normal form twist matrix
        * b, c: TGFSR(R) tempering bitmasks
        * s, t: TGFSR(R) tempering bit shifts
        * u, d, l: additional Mersenne Twister tempering bit shifts/masks
        """

    @abstractproperty
    def _f(self):
        """constant in init"""

    _state = None
    _state_ret = None
    _index = None

    def __init__(self, seed):
        n, m, r = self._nmr
        w = np.dtype(self.dtype).itemsize * 8
        f = self.dtype(self._f)
        self._index = n
        self._state = state = np.empty(n, dtype=self.dtype)
        state[0] = seed
        for i in range(1, n):
            state[i] = f * (state[i-1] ^ (state[i-1] >> (w-2))) + i

    def __call__(self):
        n, m, r = self._nmr
        if self._index >= n:
            self._twist()
        ret = self._state_ret[self._index]
        self._index += 1
        return ret

    def _twist(self):
        n, m, r = self._nmr
        a, b, c, s, t, u, d, l = map(self.dtype, self._abcstudl)
        lower_mask = self.dtype((1 << r) - 1)
        upper_mask = ~lower_mask
        state = self._state
        for i in range(n):
            x = (state[i] & upper_mask) + (state[(i+1)%n] & lower_mask)
            xA = x >> 1
            if x & 1:
                xA ^= a
            state[i] = state[(i + m) % n] ^ xA

        self._state_ret = state ^ ((state >> u) & d)
        state = self._state_ret
        state ^= (state << s) & b
        state ^= (state << t) & c
        state ^= state >> l

        self._index = 0


class MT19937(MersenneTwister):
    dtype = np.uint32
    _f = 1812433253
    _nmr = (624, 397, 31)
    _abcstudl = (0x9908B0DF,
                 0x9D2C5680, 0xEFC60000,
                 7, 15,
                 11, 0xFFFFFFFF, 18)
