# -*- coding: utf-8 -*-

"""number theory helper routines"""

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

def ceil_div(a, b):
    """ceil(a / b) for int"""
    return (a + b - 1) // b

def round_up(a, b):
    """ceil_div(a, b) * b"""
    return ceil_div(a, b) * b
