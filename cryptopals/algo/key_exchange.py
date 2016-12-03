# -*- coding: utf-8 -*-

"""key-exchange algorithms"""

from .numt import powmod, ceil_div
import numpy as np

class DiffieHellman:
    class Session:
        _dh = None
        _priv = None

        def __init__(self, dh):
            self._dh = dh
            t = 0
            for i in range(ceil_div(dh._p.bit_length(), 32)):
                t = (t << 32) | np.random.randint(1 << 32)
            self._priv = t % dh._p

        @property
        def pubkey(self):
            return powmod(self._dh._g, self._priv, self._dh._p)

        def get_session_key(self, peer_pub):
            return powmod(peer_pub, self._priv, self._dh._p)

    _p = None
    _g = None
    def __init__(self, p, g):
        """p is a prime, and g is a primitive root modulo p"""
        self._p = p
        self._g = g

    def make_session(self):
        return self.Session(self)
