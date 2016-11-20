# -*- coding: utf-8 -*-

from .bytearr import Bytearr
from .algo.block import (pkcs7_pad, pkcs7_unpad, cbc_decrypt, cbc_encrypt,
                         aes_ecb)
from .utils import challenge, assert_eq, open_resource, summarize_str

import numpy as np

@challenge
def ch09():
    a = "YELLOW SUBMARINE"
    b = b"YELLOW SUBMARINE\x04\x04\x04\x04"
    assert_eq(pkcs7_pad(a, 20), b)

@challenge
def ch10():
    with open_resource() as fin:
        data = Bytearr.from_base64(fin.read())

    iv = np.zeros((16, ), dtype=np.uint8)
    cipher = aes_ecb("YELLOW SUBMARINE")
    plaintext = cbc_decrypt(iv, data, cipher.decryptor().update)
    ciphertext_check = cbc_encrypt(iv, plaintext, cipher.encryptor().update)
    assert_eq(ciphertext_check, data)
    return summarize_str(pkcs7_unpad(plaintext.to_str()))
