# -*- coding: utf-8 -*-

"""block ciphers"""

from ..utils import as_bytes, as_np_bytearr, CipherError
from ..bytearr import Bytearr

import numpy as np

import collections
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
    if len(data) % block:
        raise CipherError(
            'bad data size: block={} data={}'.format(block, len(data)))
    nr = data[-1]
    if not (nr > 0 and set(data[-nr:]) == {nr}):
        raise CipherError('bad padding(block={}): {}'.format(block, data))
    return data[:-nr]


def cbc_encrypt(iv, data, block_enc, block_size=16):
    """encrypt in CBC mode

    :rtype: :class:`.Bytearr`
    """

    iv = as_np_bytearr(iv)
    data = as_np_bytearr(data).reshape(-1, block_size)
    ret = np.empty_like(data)
    for idx, plain in enumerate(data):
        iv = as_np_bytearr(block_enc(as_bytes(plain ^ iv)))
        ret[idx] = iv
    return Bytearr(ret.flatten(), allow_borrow=True)

def cbc_decrypt(iv, data, block_dec, block_size=16):
    """transform deciphered block data to get original plaintext

    :rtype: :class:`.Bytearr`
    """
    iv = as_np_bytearr(iv)
    ciphertext = as_np_bytearr(data)
    plaintext = as_np_bytearr(block_dec(ciphertext.tostring())).reshape(
        -1, block_size)
    plaintext[0] ^= iv
    plaintext[1:] ^= ciphertext.reshape(-1, block_size)[:-1]
    return Bytearr(plaintext.flatten(), allow_borrow=True)

def aes_ecb(key):
    """helper for create a :mode:`cryptography` AES cipher in ECB mode"""
    from cryptography.hazmat.primitives.ciphers import (
        Cipher, algorithms, modes)
    from cryptography.hazmat.backends import default_backend
    key = as_bytes(key)
    backend = default_backend()
    return Cipher(algorithms.AES(key), modes.ECB(), backend=backend)

def ctr_encrypt(block_cipher, data, nonce=0, dtype=np.uint64):
    """CTR mode that turns a block cipher into a key stream; note that data may
    be modified inplace

    :param block_cipher: callable to encrypt a single block that
        constitutes nonce and counter
    :rtype: numpy.ndarary[np.uint8]
    """

    assert isinstance(block_cipher, collections.Callable)
    state = np.zeros(2, dtype=dtype)
    state[0] = nonce
    bs = state.nbytes
    data = as_np_bytearr(data)
    for i in range(0, len(data), bs):
        end = min(i + bs, len(data))
        data[i:end] ^= as_np_bytearr(block_cipher(state.tostring()))[:end-i]
        state[1] += 1
    return data
