# -*- coding: utf-8 -*-

from .utils import (challenge, assert_eq, open_resource, CipherError, as_bytes,
                    summarize_str)
from .algo.block import (aes_ecb, pkcs7_pad, pkcs7_unpad, cbc_encrypt,
                         cbc_decrypt, as_np_bytearr, ctr_encrypt)
from .bytearr import Bytearr

import numpy as np

@challenge
def ch17():
    class Server:
        _data = None
        _encr_impl = None
        _decr_impl = None

        def __init__(self):
            self._data = data = []
            with open_resource() as fin:
                for line in fin:
                    data.append(Bytearr.from_base64(line))
            enc = aes_ecb(np.random.bytes(16))
            self._encr_impl = enc.encryptor().update
            self._decr_impl = enc.decryptor().update

        def query(self):
            iv = Bytearr(np.random.bytes(16), allow_borrow=True)
            data = self._data[np.random.randint(len(self._data))]
            data = pkcs7_pad(data)
            return iv, cbc_encrypt(iv, data, self._encr_impl).np_data

        def check(self, iv, ciphertext):
            data = cbc_decrypt(iv, ciphertext, self._decr_impl)
            try:
                pkcs7_unpad(data)
                return True
            except CipherError:
                return False

    srv = Server()
    iv, ciphertext = srv.query()
    block_size = 16
    assert len(ciphertext) % block_size == 0

    def decr_block(ciphertext):
        # decrypt a single block

        def check_single(iv):
            # assume (iv ^ plain)[-1] is '\x01'
            plain = np.empty_like(iv)
            plain[-1] = iv[-1] ^ 1
            for pad_size in range(2, block_size + 1):
                iv[-pad_size+1:] = plain[-pad_size+1:] ^ pad_size
                found = False
                for test in range(256):
                    iv[-pad_size] = test
                    if srv.check(iv, ciphertext):
                        found = True
                        break
                if not found:
                    return
                plain[-pad_size] = iv[-pad_size] ^ pad_size
            return plain

        while True:
            iv = as_np_bytearr(np.random.bytes(16))
            if not srv.check(iv, ciphertext):
                continue
            ret = check_single(iv)
            if ret is not None:
                return ret

    plain = []
    for i in range(0, len(ciphertext), block_size):
        cur = decr_block(ciphertext[i:i+block_size])
        plain.extend(map(int, iv ^ cur))
        iv = ciphertext[i:i+block_size]
    return pkcs7_unpad(plain).decode('ascii')

@challenge
def ch18():
    data = ('L77na/nrFsKvynd6HzOoG7GHTLXsTVu9qvY'
            '/2syLXzhPweyyMTJULu/6/kXX0KSvoOLSFQ==')
    return as_bytes(ctr_encrypt(
        aes_ecb('YELLOW SUBMARINE').encryptor().update,
        Bytearr.from_base64(data))).decode('ascii')

@challenge
def ch19():
    data = []
    with open_resource() as fin:
        for line in fin:
            data.append(Bytearr.from_base64(line).np_data)

    key = []
    for i in range(0, max(map(len, data))):
        cipher = np.array([j[i] for j in data if len(j) > i])
        if len(cipher) <= len(data) // 2:
            break
        keys = np.arange(256, dtype=np.uint8)[:, np.newaxis]
        cand = cipher[np.newaxis] ^ keys
        score = np.logical_and(cand >= ord('a'), cand <= ord('z')).sum(axis=1)
        key.append(np.argmax(score))

    key.append(ord('g') ^ 103); key.append(ord('h') ^ 104)
    key.append(ord('n') ^ 110); key.append(ord('d') ^ 100)
    key.append(ord('d') ^ 100)
    key.extend(as_np_bytearr('ead') ^ [101, 97, 100])
    key.extend(as_np_bytearr('n,') ^ [110, 44])
    key = as_np_bytearr(key)
    plain = []
    for i in data:
        plain.extend(key[:len(i)] ^ i)
        plain.append(ord(' '))
    return summarize_str(as_bytes(plain).decode('ascii'))
