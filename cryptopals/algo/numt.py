# -*- coding: utf-8 -*-

"""number theory helper routines"""

import numpy as np

def powmod(a, p, m):
    """compute a ** b % m"""
    assert p >= 0 and m >= 2, (p, m)
    ans = 1
    cur = a % m
    while p:
        if p & 1:
            ans = (ans * cur) % m
        p >>= 1
        cur = (cur * cur) % m
    return ans

def hex2int(hv):
    """convert hex string to large int"""
    assert isinstance(hv, str)
    return int(hv, 16)

def int2hex(iv):
    """convert int value to hex"""
    return hex(iv)[2:]

def bytes2int(bv, endian='little'):
    """convert bytes to large int"""
    return int.from_bytes(bv, endian)

def int2bytes(iv, length=None, endian='little'):
    """convert bytes to large int"""
    assert isinstance(iv, int)
    if length is None:
        length = ceil_div(iv.bit_length(), 8)
    return iv.to_bytes(length, endian)

def ceil_div(a, b):
    """ceil(a / b) for int"""
    return (a + b - 1) // b

def round_up(a, b):
    """ceil_div(a, b) * b"""
    return ceil_div(a, b) * b

def large_randint(n, rng=np.random):
    """generate randint(n) for large n values"""
    assert isinstance(n, int)
    t = 0
    for i in range(ceil_div(n.bit_length(), 32)):
        t = (t << 32) | rng.randint(1 << 32)
    return t % n
