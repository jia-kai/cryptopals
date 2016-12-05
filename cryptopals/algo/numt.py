# -*- coding: utf-8 -*-

"""number theory helper routines"""

import numpy as np
import gmpy2

import math

def powmod(a, p, m):
    """compute a ** b % m"""
    return pow(a, p, m)

def hex2int(hv):
    """convert hex string to large int"""
    assert isinstance(hv, str)
    return int(hv, 16)

def int2hex(iv):
    """convert int value to hex"""
    return hex(iv)[2:]

def bytes2int(bv, endian='big'):
    """convert bytes to large int"""
    return int.from_bytes(bv, endian)

def int2bytes(iv, length=None, endian='big'):
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
    nbytes = n.bit_length() // 8 + 1
    return bytes2int(np.random.bytes(nbytes)) % n


def primep(n):
    return gmpy2.is_prime(n)

def gen_prime(bits, requirement=lambda x: True):
    """generate a prime of at least given bits long"""
    n = bytes2int(np.random.bytes(ceil_div(bits, 8)))
    while True:
        n = next_prime(n)
        if requirement(n):
            return n

        # apply Cramér's conjecture to jump over the gap
        n += max(2, int(n.bit_length() * math.log(2))**2)

def next_prime(n):
    return int(gmpy2.next_prime(n))

def egcd(a, b):
    """extended gcd

    :return: g, x, y such that x * a + y * b == g and x > 0
    """
    g, x, y = map(int, gmpy2.gcdext(a, b))
    x %= (b // g)
    y = (g - x * a) // b
    return g, x, y

def lcm(a, b):
    g, _, _ = egcd(a, b)
    return a * b // g

def invmod(a, m, b=1):
    """solve x such that a*x === b (mod m) and return the least positive x"""
    b %= m
    g, x, _ = egcd(a, m)
    k, r = divmod(b, g)
    assert r == 0, (a, m, b, g)
    if k != 1:
        x *= k
    return x % m

def solve_congruences(rems, mods):
    """solve for x such that x === rems[i] (mod mods[i])

    :return: (k, b) so that kt + b is a solution for any t
    """
    assert len(mods) == len(rems)

    cur_mod = mods[0]
    cur_rem = rems[0]

    for i in range(1, len(mods)):
        # assume x == k * cur_mod + cur_rem
        k = invmod(cur_mod, mods[i], rems[i] - cur_rem)
        cur_rem = k * cur_mod + cur_rem
        cur_mod = lcm(cur_mod, mods[i])
        cur_rem %= cur_mod

    return cur_mod, cur_rem
