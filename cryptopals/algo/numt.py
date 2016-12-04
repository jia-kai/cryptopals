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
    nbytes = n.bit_length() // 8 + 1
    return bytes2int(np.random.bytes(nbytes)) % n


class _PrimalityTestImpl:
    _presel = None
    """small primes for fast test"""

    def __init__(self, presel_size):
        self._presel = presel = [2]
        cur = 2
        while len(presel) < presel_size:
            is_p = False
            while not is_p:
                cur += 1
                is_p = True
                i = iter(presel)
                while True:
                    iv = next(i)
                    if cur % iv == 0:
                        is_p = False
                        break
                    if iv * iv >= cur:
                        break
            presel.append(cur)

    def __call__(self, n):
        for i in self._presel:
            if n % i == 0:
                return False
            if i * i >= n:
                return True

        # Miller–Rabin test
        d = n - 1
        s = 0
        while not (d & 1):
            s += 1
            d >>= 1
        mod_m1 = n - 1

        # with error rate 4***(-60)
        for _ in range(60):
            a = int(np.random.randint(1 << 30))
            ap = powmod(a, d, n)
            if ap == 1:
                continue

            is_comp = True
            for i in range(s):
                if ap == mod_m1:
                    is_comp = False
                    break
                ap = (ap * ap) % n

            if is_comp:
                return False

        return True

primep = _PrimalityTestImpl(100)

def gen_prime(bits):
    """generate a prime of at least given bits long"""
    n = bytes2int(np.random.bytes(ceil_div(bits, 8)))
    return next_prime(n)

def next_prime(n):
    if not (n & 1):
        n += 1
    while not primep(n):
        n += 2
    return n

def egcd(a, b):
    """extended gcd

    :return: g, x, y such that x * a + y * b == g and x > 0
    """
    if a == 0 or b == 0:
        return a | b, 1, 1

    k, r = divmod(a, b)
    g, x1, y1 = egcd(b, r)

    x, y = y1, x1 - k * y1
    x %= (b // g)
    y = (g - x * a) // b
    return g, x, y

def lcm(a, b):
    g, _, _ = egcd(a, b)
    return a * b // g

def invmod(a, m, b=1):
    """solve x such that a*x === b (mod m) and x > 0"""
    b %= m
    g, x, _ = egcd(a, m)
    k, r = divmod(b, g)
    assert r == 0, (a, m, b, g)
    if k != 1:
        x *= k
    return x

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

def main():
    from ..utils import assert_eq
    print('512-bit prime:', gen_prime(512))
    a = np.random.randint(2, 1<<16)
    b = np.random.randint(2, 1<<16)
    g, x, y = egcd(a, b)
    assert_eq(a * x + b * y, g)
    assert_eq(invmod(123, 4567) * 123 % 4567, 1)

if __name__ == '__main__':
    main()
