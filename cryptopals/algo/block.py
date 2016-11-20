# -*- coding: utf-8 -*-

"""block ciphers"""

from ..utils import as_bytes
from ..bytearr import Bytearr

import numpy as np

import itertools

def pkcs7_pad(data, block=16):
    """PKCS#7 padding

    :rtype: :class:`bytes`
    """
    data = as_bytes(data)
    n = block - len(data) % block
    return data + bytes(itertools.repeat(n, n))

def pkcs7_unpad(data, block=16):
    """PKCS#7 unpadding

    :rtype: :class:`bytes`
    """
    data = as_bytes(data)
    return data[:-data[-1]]


def cbc_encrypt(iv, data, block_enc, block_size=16):
    """encrypt in CBC mode

    :rtype: :class:`.Bytearr`
    """

    iv = Bytearr(iv, allow_borrow=True).np_data
    data = Bytearr(data, allow_borrow=True).np_data.reshape(-1, block_size)
    ret = np.empty_like(data)
    for idx, plain in enumerate(data):
        iv = Bytearr(
            block_enc(as_bytes(plain ^ iv)), allow_borrow=True).np_data
        ret[idx] = iv
    return Bytearr(ret.flatten(), allow_borrow=True)

def cbc_decrypt(iv, data, block_dec, block_size=16):
    """transform deciphered block data to get original plaintext

    :rtype: :class:`.Bytearr`
    """
    iv = Bytearr(iv, allow_borrow=True).np_data
    ciphertext = Bytearr(data, allow_borrow=True)
    plaintext = Bytearr(block_dec(ciphertext.to_bytes())).np_data.reshape(
        -1, block_size)
    plaintext[0] ^= iv
    plaintext[1:] ^= ciphertext.np_data.reshape(-1, block_size)[:-1]
    return Bytearr(plaintext.flatten(), allow_borrow=True)

def aes_ecb(key):
    """helper for create a :mode:`cryptography` AES cipher in ECB mode"""
    from cryptography.hazmat.primitives.ciphers import (
        Cipher, algorithms, modes)
    from cryptography.hazmat.backends import default_backend
    key = as_bytes(key)
    backend = default_backend()
    return Cipher(algorithms.AES(key), modes.ECB(), backend=backend)
