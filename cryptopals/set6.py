# -*- coding: utf-8 -*-

from .utils import challenge, assert_eq, as_bytes, open_output
from .algo.numt import powmod, invmod, bytes2int, int2bytes, ceil_div
from .algo.asym import RSA
from .bytearr import Bytearr

import numpy as np
import gmpy2

import re
import functools
import os
import itertools

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
    # it is basic math
    return 'skipped'

@challenge
def ch45():
    # g, y both being 0 or 1 mod p, boring
    return 'skipped'

@challenge
def ch46():
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

@challenge
def ch47():
    return 'see ch48'

@challenge
def ch48():
    enc, dec = RSA.make_enc_dec_pair(bits=(
        256 if not os.getenv('CRYPTOPALS_BIGTEST')
        else 1024))

    # note that our rng generates n with small n % 8, so this attack would be
    # much faster than if were done in practice
    n = enc._n
    nbytes = ceil_div(n.bit_length(), 8)

    B = 1 << (8 * (nbytes - 2))
    def check_pkcs1(ciphertext, *, _dec=dec, _lo=B*2, _hi=B*3):
        p = _dec(ciphertext)
        return p >= _lo and p < _hi
    del dec

    msg = bytes2int(b'\x00\x02' + np.random.bytes(nbytes-2))
    c0 = enc(msg)
    assert check_pkcs1(c0)

    tot_nr_try = 0
    with open_output() as fout:
        plog = functools.partial(print, file=fout)

        intervals = [(B*2, B*3-1)]
        # s_0 m >= n
        s_prev = ceil_div(n, intervals[0][1])
        while not check_pkcs1(enc(s_prev) * c0):
            s_prev += 1
        plog('s0={}'.format(s_prev))

        for i in itertools.count(2):
            if len(intervals) == 1:
                (a, b), = intervals

                k = 4
                # Here is to ensure r >= max{r_prev} * k
                #
                # Increasing k results in larger r and thus larger s (s approx
                # r since (sm - rn) remains O(1)), thus resulting in quicker
                # decrease of interval size and smaller iteration steps (since
                # interval length is B/s); however total runtime seems to
                # remain quite constant
                r = (b * s_prev - 2 * B) // n * k
                for nr_try in itertools.count(1):
                    s = ceil_div(2*B + r*n, b)
                    s_hi = (3*B + r*n) // a
                    found = False
                    while s <= s_hi:
                        if check_pkcs1(enc(s) * c0):
                            found = True
                            break
                        s += 1
                    if found:
                        break
                    r += 1
                tot_nr_try += nr_try
                plog('{}: with one interval, nr_try={} s={}'.format(
                    i, nr_try, s))
            else:
                s = s_prev + 1
                nr_try = 0
                while not check_pkcs1(enc(s) * c0):
                    s += 1
                    nr_try += 1
                tot_nr_try += nr_try
                plog('{}: with multi intervals, nr_try={} s={}'.format(
                    i, nr_try, s))

            # the evidence 2B <= sm - rn <= 3B-1 with constraint 2B <= m < 3B
            # can give about sB/n intervals; new interval lengths is smaller
            # than distances between previous intervals, and also it can be
            # easily proved that no two new intervals can reside within in one
            # previous interval; so total number of intervals is bounded.

            new_intervals = []
            new_intervals_lens = []
            for a, b in intervals:
                rlo = ceil_div(s*a - 3*B + 1, n)
                rhi = (s*b - 2*B) // n
                r = rlo
                while r <= rhi:
                    ar = max(a, ceil_div(2*B + r*n, s))
                    br = min(b, (3*B - 1 + r*n) // s)
                    if ar <= br:
                        new_intervals.append((ar, br))
                        new_intervals_lens.append(br - ar + 1)
                    r += 1

            s_prev = s % n
            intervals = new_intervals
            nl = np.array(new_intervals_lens, dtype=np.float128)
            plog('{}: new_intervals={} length=(min={},max={},avg={})'.format(
                i, nl.size, nl.min(), nl.max(), nl.mean()))

            if new_intervals_lens == [1]:
                break

    (_, ans), = intervals
    assert_eq(msg, ans)
    return 'in {} iterations, tot_nr_try={}'.format(i, tot_nr_try)
