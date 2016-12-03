# -*- coding: utf-8 -*-

from ..utils import as_bytes, as_np_bytearr
from ..bytearr import Bytearr

import numpy as np

from abc import ABCMeta, abstractproperty, abstractmethod
import struct

class _MD64Base:
    """base class for implementing a hash function with Merkle–Damgård
    construction"""

    _io_dtype = None

    @abstractproperty
    def _init_state(self):
        """default initial state"""

    @abstractproperty
    def _endian(self):
        """'>' or '<' to mark endian"""

    @classmethod
    def _lrotate(cls, n, b):
        return (n << np.uint32(b)) | (n >> np.uint32(32 - b))

    @abstractmethod
    def _compress(self, block, state):
        """compress a single 64-byte block

        :returnL new state
        """

    def __init__(self):
        self._io_dtype = np.dtype(self._endian + 'I')

    @property
    def io_dtype(self):
        """numpy dtype for binary I/O"""
        return self._io_dtype

    def pad(self, message, nbytes_off=0):
        """add padding used in Merkle–Damgård construction to 64 bytes

        :param nbytes_off: number of bytes to be assumed before given messsage
        :return: bytes
        """
        message = as_bytes(message)
        nbytes = len(message) + nbytes_off

        message += b'\x80'
        message += b'\x00' * ((56 - (nbytes + 1)) % 64)
        message += struct.pack(self._endian + 'Q', nbytes * 8)

        return message

    def __call__(self, message, state=None, nbytes_off=0, outhex=False):
        """compute hash result for given message

        :param state: overwrite initial state
        :param nbytes_off: see :meth:`pad`
        :param outhex: if True, return a hex str; otherwise return digest bytes
        """
        warnings_filters = np.warnings.filters[:]
        np.warnings.simplefilter("ignore", RuntimeWarning)

        if state is None:
            state = self._init_state
        state = np.array(state, dtype=np.uint32)
        message = self.pad(message, nbytes_off)
        assert len(message) % 64 == 0

        for i in range(0, len(message), 64):
            block = np.fromstring(message[i:i+64], dtype=self._io_dtype)
            block = block.astype(np.uint32)
            state = self._compress(block, state)

        np.warnings.filters[:] = warnings_filters
        ret = state.astype(self._io_dtype).tobytes()
        if outhex:
            ret = Bytearr(ret).to_hex()
        return ret


class _SHA1Impl(_MD64Base):
    _init_state = (0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0)
    _endian = '>'

    @classmethod
    def _compress(cls, block, state):
        # modified from https://github.com/ajalt/python-sha1
        lrot = cls._lrotate

        w = np.zeros(80, dtype=np.uint32)
        w[:16] = block
        for j in range(16, 80):
            w[j] = lrot(w[j-3] ^ w[j-8] ^ w[j-14] ^ w[j-16], 1)

        a, b, c, d, e = state

        for i in range(80):
            if 0 <= i <= 19:
                f = (b & c) | (~b & d)
                k = 0x5A827999
            elif 20 <= i <= 39:
                f = b ^ c ^ d
                k = 0x6ED9EBA1
            elif 40 <= i <= 59:
                f = (b & c) | (b & d) | (c & d)
                k = 0x8F1BBCDC
            elif 60 <= i <= 79:
                f = b ^ c ^ d
                k = 0xCA62C1D6

            a, b, c, d, e = (
                lrot(a, 5) + f + e + np.uint32(k) + w[i],
                a, lrot(b, 30), c, d)

        state += a, b, c, d, e
        return state


class _MD4Impl(_MD64Base):
    _init_state = _SHA1Impl._init_state[:4]
    _endian = '<'

    @classmethod
    def _compress(cls, block, state):
        # modified from http://www.acooke.org/cute/PurePython0.html
        assert len(block) == 16
        lrot = cls._lrotate

        def f(x, y, z):
            return x & y | ~x & z
        def g(x, y, z):
            return x & y | x & z | y & z
        def h(x, y, z):
            return x ^ y ^ z

        def f1(a, b, c, d, k, s):
            return lrot(a + f(b, c, d) + block[k], s)
        def f2(a, b, c, d, k, s):
            return lrot(a + g(b, c, d) + block[k] + np.uint32(0x5a827999), s)
        def f3(a, b, c, d, k, s):
            return lrot(a + h(b, c, d) + block[k] + np.uint32(0x6ed9eba1), s)

        a, b, c, d = state
        a = f1(a,b,c,d, 0, 3)
        d = f1(d,a,b,c, 1, 7)
        c = f1(c,d,a,b, 2,11)
        b = f1(b,c,d,a, 3,19)
        a = f1(a,b,c,d, 4, 3)
        d = f1(d,a,b,c, 5, 7)
        c = f1(c,d,a,b, 6,11)
        b = f1(b,c,d,a, 7,19)
        a = f1(a,b,c,d, 8, 3)
        d = f1(d,a,b,c, 9, 7)
        c = f1(c,d,a,b,10,11)
        b = f1(b,c,d,a,11,19)
        a = f1(a,b,c,d,12, 3)
        d = f1(d,a,b,c,13, 7)
        c = f1(c,d,a,b,14,11)
        b = f1(b,c,d,a,15,19)

        a = f2(a,b,c,d, 0, 3)
        d = f2(d,a,b,c, 4, 5)
        c = f2(c,d,a,b, 8, 9)
        b = f2(b,c,d,a,12,13)
        a = f2(a,b,c,d, 1, 3)
        d = f2(d,a,b,c, 5, 5)
        c = f2(c,d,a,b, 9, 9)
        b = f2(b,c,d,a,13,13)
        a = f2(a,b,c,d, 2, 3)
        d = f2(d,a,b,c, 6, 5)
        c = f2(c,d,a,b,10, 9)
        b = f2(b,c,d,a,14,13)
        a = f2(a,b,c,d, 3, 3)
        d = f2(d,a,b,c, 7, 5)
        c = f2(c,d,a,b,11, 9)
        b = f2(b,c,d,a,15,13)

        a = f3(a,b,c,d, 0, 3)
        d = f3(d,a,b,c, 8, 9)
        c = f3(c,d,a,b, 4,11)
        b = f3(b,c,d,a,12,15)
        a = f3(a,b,c,d, 2, 3)
        d = f3(d,a,b,c,10, 9)
        c = f3(c,d,a,b, 6,11)
        b = f3(b,c,d,a,14,15)
        a = f3(a,b,c,d, 1, 3)
        d = f3(d,a,b,c, 9, 9)
        c = f3(c,d,a,b, 5,11)
        b = f3(b,c,d,a,13,15)
        a = f3(a,b,c,d, 3, 3)
        d = f3(d,a,b,c,11, 9)
        c = f3(c,d,a,b, 7,11)
        b = f3(b,c,d,a,15,15)

        state += a, b, c, d
        return state

def hmac(key, message, hash_impl, block_size=64, **kwargs):
    key = as_bytes(key)
    if len(key) > block_size:
        key = hash_impl(key)
    if len(key) < block_size:
        key += b'\x00' * (block_size - len(key))

    key = as_np_bytearr(key)
    assert key.size == block_size
    o_key_pad = as_bytes(key ^ 0x5c)
    i_key_pad = as_bytes(key ^ 0x36)
    return hash_impl(o_key_pad + hash_impl(i_key_pad + as_bytes(message)),
                     **kwargs)

sha1 = _SHA1Impl()
md4 = _MD4Impl()
