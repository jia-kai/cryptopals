# -*- coding: utf-8 -*-

from .utils import challenge, assert_eq, as_bytes
from .algo.numt import powmod, invmod, bytes2int, int2bytes
from .algo.asym import RSA
from .bytearr import Bytearr

import numpy as np
import gmpy2

import re
import os

@challenge
def ch41():
    enc, dec = RSA.make_enc_dec_pair()
    msg = bytes2int(np.random.bytes(20))
    c0 = enc(msg)
    S = 2
    c1 = c0 * enc(S)

    m1 = dec(c1)
    m0 = invmod(S, enc._n, m1)

    assert_eq(m0, msg)

@challenge
def ch42():
    def make_pkcs15(msg, num=1):
        return b'\x01' + (b'\xff' * num) + b'\x00ASN.1' + as_bytes(msg)

    def pkcs15_check(padded, msg):
        pat = b'\x01\xff+\x00ASN.1' + re.escape(as_bytes(msg))
        return re.match(pat, as_bytes(padded)) is not None

    assert_eq(pkcs15_check(make_pkcs15('x'), 'x'), True)
    assert_eq(pkcs15_check(make_pkcs15('x', 2), 'x'), True)
    assert_eq(pkcs15_check(make_pkcs15('x', 0), 'x'), False)
    assert_eq(pkcs15_check(make_pkcs15('x'), 'y'), False)

    enc, _ = RSA.make_enc_dec_pair(bits=1024, e=3)

    msg = 'hi mom'
    msg_pad = make_pkcs15(msg)
    n = bytes2int(msg_pad)
    n_upper = n + 1
    while True:
        n3, exact = gmpy2.iroot(n, 3)
        n3 = int(n3)
        if not exact:
            n3 += 1
        n33 = n3**3
        assert n33 < enc._n
        if n33 < n_upper:
            break

        n <<= 8
        n_upper <<= 8

    assert_eq(pkcs15_check(int2bytes(enc(n3)), msg), True)

@challenge
def ch43():
    # I consider all brute-force methods half-boring, and this challenge only
    # plays with simple modular equations being the other half
    return 'skipped'

@challenge
def ch44():
    # g, y both being 0 or 1 mod p, boring
    return 'skipped'

@challenge
def ch45():
    if os.getenv('CRYPTOPALS_BIGTEST'):
        msg = Bytearr.from_base64(
            'VGhhdCdzIHdoeSBJIGZvdW5kIHlvdSBkb24ndCBwbGF5IGF'
            'yb3VuZCB3aXRoIHRoZSBGdW5reSBDb2xkIE1lZGluYQ==')
        enc, dec = RSA.make_enc_dec_pair(bits=1024)
    else:
        msg = 'halo'
        enc, dec = RSA.make_enc_dec_pair(bits=64)
    msg = as_bytes(msg)

    def parity(ciphertext, *, _dec=dec):
        return _dec(ciphertext) & 1
    del dec

    ciphertext = enc(bytes2int(msg))

    a = 0
    s = 1
    slog = 0
    snext = 2
    snext_enc_step = snext_enc = enc(snext)
    snext_enc *= ciphertext
    while enc._n >= snext:
        a <<= 1
        if parity(snext_enc):
            a += 1
        s = snext
        slog += 1
        snext <<= 1
        snext_enc = snext_enc_step * snext_enc % enc._n

    lo = a * enc._n >> slog
    hi = (a + 1) * enc._n >> slog
    diff = hi - lo
    assert diff >= 1
    if diff > 1:
        assert diff == 2
        mid = ((a<<1) + 1) * enc._n >> (slog + 1)
        if not parity(enc(snext) * ciphertext):
            hi = mid

    recovered_plain = int2bytes(hi)
    assert_eq(recovered_plain, msg)
    return recovered_plain.decode('ascii')
