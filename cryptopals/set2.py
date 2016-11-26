# -*- coding: utf-8 -*-

from .bytearr import Bytearr
from .algo.block import (pkcs7_pad, pkcs7_unpad, cbc_decrypt, cbc_encrypt,
                         aes_ecb)
from .utils import challenge, assert_eq, open_resource, summarize_str, as_bytes

import numpy as np

import functools
import itertools
from urllib.parse import urlencode, parse_qsl, quote

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

@challenge
def ch11():
    ECB = 0
    CBC = 1
    rng = np.random.RandomState()
    def encr_oracle(mode, inp):
        inp = (rng.bytes(rng.randint(5, 10)) + as_bytes(inp) +
               rng.bytes(rng.randint(5, 10)))
        inp = pkcs7_pad(inp)
        key = rng.bytes(16)
        enc = aes_ecb(key).encryptor().update
        if mode == ECB:
            return enc(inp)
        iv = rng.bytes(16)
        return cbc_encrypt(iv, inp, enc)

    def detector(encr):
        r = encr(b'\x00' * 48)
        return int(r[16:32] != r[32:48])

    for i in [ECB, CBC]:
        assert_eq(detector(functools.partial(encr_oracle, i)), i)

@challenge
def ch12():
    secret_b64 = (
        'Um9sbGluJyBpbiBteSA1LjAKV2l0aCBteSByYWctdG9wIGRvd24gc28gbXkg'
        'aGFpciBjYW4gYmxvdwpUaGUgZ2lybGllcyBvbiBzdGFuZGJ5IHdhdmluZyBq'
        'dXN0IHRvIHNheSBoaQpEaWQgeW91IHN0b3A/IE5vLCBJIGp1c3QgZHJvdmUg'
        'YnkK')

    def encr(data, *,
             secret=Bytearr.from_base64(secret_b64).to_bytes(),
             impl=aes_ecb(np.random.bytes(16)).encryptor().update):
        return impl(pkcs7_pad(as_bytes(data) + secret))

    def get_block_size():
        seq = ['\x00']
        prev = encr(seq)[:len(seq)]
        while True:
            seq.append(seq[-1])
            cur = encr(seq)[:len(seq)]
            if cur[:-1] == prev:
                return len(prev)
            prev = cur

    block_size = get_block_size()
    # check ECB
    ecb_chk = encr('\x00' * (block_size * 2))
    assert ecb_chk[:block_size] == ecb_chk[block_size:block_size*2]

    inferred_plain = [0] * block_size
    for start in itertools.count():
        prefix = block_size - 1 - start % block_size
        end = start + prefix + 1
        target = encr([0]*prefix)[end-block_size:end]
        prefix_seq = inferred_plain[-block_size+1:]
        prefix_seq.append(0)
        found = False
        for probe in range(256):
            prefix_seq[-1] = probe
            if encr(prefix_seq)[:block_size] == target:
                found = True
                inferred_plain.append(probe)
                break
        if not found:
            # we could not recover the second pkcs#7 padding character since
            # the first padding would change when adding prefix for it
            # So found would be False on second padding char
            assert inferred_plain[-1] == 1
            break
    plain = as_bytes(pkcs7_unpad(inferred_plain[block_size:]))
    assert_eq(plain, Bytearr.from_base64(secret_b64).to_bytes())
    return summarize_str(plain)

@challenge
def ch13():
    class Server:
        _encr_impl = None
        _decr_impl = None

        def __init__(self):
            enc = aes_ecb(np.random.bytes(16))
            self._encr_impl = enc.encryptor().update
            self._decr_impl = enc.decryptor().update

        def get_cookie(self, email):
            q = urlencode([('email', email), ('uid', '10'), ('role', 'user')])
            return self._encr_impl(pkcs7_pad(q))

        def parse_cookie(self, cookie):
            return {k.decode('ascii'): v.decode('ascii')
                    for k, v in
                    parse_qsl(pkcs7_unpad(self._decr_impl(cookie)),
                              strict_parsing=True)}

    srv = Server()
    blk_size = 16
    s0 = 'email=&uid=10&role='
    p0 = blk_size - len(s0) % blk_size
    up_to_role_eq = srv.get_cookie('A' * p0)[:len(s0)+p0]
    assert len(up_to_role_eq) % blk_size == 0
    admin_pad = 'admin'
    n = blk_size - len(admin_pad)
    while quote(chr(n)) != chr(n):
        n += blk_size
    admin_pad += chr(n) * n
    admin_enc = srv.get_cookie('A'*(blk_size - len('email=')) + admin_pad)[
        blk_size:blk_size+len(admin_pad)]
    evil = up_to_role_eq + admin_enc
    ret = srv.parse_cookie(evil)
    assert_eq(ret['role'], 'admin')
    return ret
