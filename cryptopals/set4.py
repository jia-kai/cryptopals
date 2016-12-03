# -*- coding: utf-8 -*-

from .utils import challenge, assert_eq, as_bytes
from .algo.hash import sha1, md4, hmac
from .bytearr import Bytearr

import functools
import numpy as np

@challenge
def ch25():
    # boring: (x ^ key) ^ (a ^ key) ^ a == x
    return 'skipped'

@challenge
def ch26():
    # boring ...
    return 'skipped'

@challenge
def ch27():
    # boring ... the fun part is to construct c1, 0, c1, but already spoiled in
    # the problem description
    return 'skipped'

@challenge
def ch28():
    import hashlib

    assert_eq(sha1("The quick brown fox jumps over the lazy dog", outhex=True),
              '2fd4e1c67a2d28fced849ee1bb76e7391b93eb12')

    for i in range(50):
        if not i:
            msg = b''
        else:
            msg = np.random.bytes(np.random.randint(0, 1024))
        h = hashlib.sha1()
        h.update(msg)
        assert_eq(h.digest(), sha1(msg))

@challenge
def ch29():
    return ch29_impl(sha1)

def ch29_impl(hash_impl):
    class Server:
        def __init__(self, keysize):
            self._key = np.random.bytes(keysize)

        def request(self, msg):
            return hash_impl(self._key + as_bytes(msg))

    ceil_div = lambda a, b: (a+b-1)//b
    round_up = lambda a, b: ceil_div(a, b) * b

    msg = (b"comment1=cooking%20MCs;userdata=foo;"
           b"comment2=%20like%20a%20pound%20of%20bacon")
    payload = b";admin=true"
    for keylen in range(65):
        srv = Server(keylen)
        mac0 = srv.request(msg)
        msg1 = hash_impl.pad(msg, keylen) + payload
        padded_size = round_up(len(msg) + keylen + 1 + 8, 64)
        state = np.fromstring(mac0, dtype=hash_impl.io_dtype)
        mac1 = hash_impl(payload, state=state, nbytes_off=padded_size)

        assert_eq(srv.request(msg1), mac1, keylen)

@challenge
def ch30():
    check = lambda a, b: assert_eq(md4(a, outhex=True), b, a)
    check('', '31d6cfe0d16ae931b73c59d7e0c089c0')
    check('a', 'bde52cb31de33e46245e05fbdbd6fb24')
    check('abc', 'a448017aaf21d8525fc10ae87aa6729d')
    check('message digest',
          'd9130a8164549fe818874806e1c7014b')
    check('abcdefghijklmnopqrstuvwxyz',
          'd79e1c308aa5bbcdeea8ed63df412da9')
    check('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789',
          '043f8582f241db351ce627e153e7f0e4')
    check('1234567890123456789012345678901234567890123456'
          '7890123456789012345678901234567890',
          'e33b4ddc9c38f2199c3e7b164fcc0536')

    ch29_impl(md4)

@challenge
def ch31():
    hmac_sha1 = functools.partial(hmac, hash_impl=sha1)
    check = lambda k, m, c: assert_eq(hmac_sha1(k, m, outhex=True), c, (k, m))
    check('', '', 'fbdb1d1b18aa6c08324b7d64b71fb76370690e1d')
    check('key', 'The quick brown fox jumps over the lazy dog',
          'de7c9b85b8b78aa6bc8a7a36f70a90701c9db4d9')

    # the challenge part is boring, having nothing to do with hmac; just guess
    # and try ...
    return 'skipped'

@challenge
def ch32():
    # boring again ...
    return 'skipped'
