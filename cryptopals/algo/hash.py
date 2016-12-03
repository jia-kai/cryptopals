# -*- coding: utf-8 -*-

from ..utils import as_bytes

import numpy as np

import struct

dt_bigu32 = np.dtype('>I')

def md_padding(message, nbytes_off=0):
    """padding used in Merkle–Damgård construction

    :param nbytes_off: number of bytes to be assumed before given messsage
    :return: bytes
    """
    message = as_bytes(message)
    nbytes = len(message) + nbytes_off

    message += b'\x80'
    message += b'\x00' * ((56 - (nbytes + 1)) % 64)
    message += struct.pack('>Q', nbytes * 8)

    return message

def sha1(message,
         state=(0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0),
         nbytes_off=0):
    """
    SHA-1 Hashing Function
    A custom SHA-1 hashing function implemented entirely in Python.

    :param message: The input message string to hash.
    :param state: initial SHA-1 internal state
    :return: SHA-1 digest bytes
    """
    # modified from https://github.com/ajalt/python-sha1

    warnings_filters = np.warnings.filters[:]
    np.warnings.simplefilter("ignore", RuntimeWarning)

    def left_rotate(n, b):
        return (n << np.uint32(b)) | (n >> np.uint32(32 - b))

    state = np.array(state, dtype=np.uint32)
    message = md_padding(message, nbytes_off)
    assert len(message) % 64 == 0

    for i in range(0, len(message), 64):
        w = np.zeros(80, dtype=np.uint32)
        w[:16] = np.fromstring(message[i:i+64], dtype=dt_bigu32)
        for j in range(16, 80):
            w[j] = left_rotate(w[j-3] ^ w[j-8] ^ w[j-14] ^ w[j-16], 1)

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
                left_rotate(a, 5) + f + e + np.uint32(k) + w[i],
                a, left_rotate(b, 30), c, d)

        state += a, b, c, d, e

    np.warnings.filters[:] = warnings_filters
    return state.astype(dt_bigu32)

