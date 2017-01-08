# -*- coding: utf-8 -*-

from .utils import (challenge, assert_eq, as_bytes, capture_stdout,
                    as_np_bytearr)
from .bytearr import Bytearr
from .algo.block import cbc_encrypt, aes_ecb, pkcs7_pad

@challenge
def ch49():
    # Too many assumptions without a concrete reference system (e.g. need
    # control on the first block, but does the bank system really allow this?)
    # And the whole challenge is still about xor ...
    return 'skipped'

@challenge
def ch50():
    # javascript is irrelevant; we use python for this demo

    class CBC_MAC:
        def __init__(self, key=b'YELLOW SUBMARINE', iv=b'\x00' * 16):
            self._key = key
            self._iv = iv
            self._encr_impl = aes_ecb(key).encryptor().update

        def __call__(self, msg):
            msg = pkcs7_pad(msg)
            return as_bytes(cbc_encrypt(self._iv, msg, self._encr_impl))[-16:]


    cbc_mac = CBC_MAC()
    assert_eq(Bytearr(cbc_mac("alert('MZA who was that?');\n")).to_hex(),
              '296b8d7cb78a243dda4d0a61d33bbdd1')

    code0 = "print('MZA who was that?');"
    mac = cbc_mac(code0)
    code1 = "print('Ayo, the Wu is back!');#"
    code1 += '#' * ((-len(code1)) % 16)
    code1 = code1.encode('utf-8')
    def get_suffix():
        # assume we know the key
        decr = aes_ecb(cbc_mac._key).decryptor().update
        def blk(out, iv):
            return as_bytes(as_np_bytearr(iv) ^ as_np_bytearr(decr(out)))
        pad = pkcs7_pad(b'')
        prev = blk(mac, pad)
        iv = cbc_encrypt(cbc_mac._iv, code1, cbc_mac._encr_impl)[-16:]
        return blk(prev, iv)

    code1 += get_suffix()
    assert_eq(cbc_mac(code1), mac)
    with capture_stdout() as s:
        exec(code1)
    s = s.getvalue()
    assert_eq(s, 'Ayo, the Wu is back!\n')
