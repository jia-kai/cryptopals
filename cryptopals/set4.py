# -*- coding: utf-8 -*-

from .utils import challenge, assert_eq
from .algo.hash import sha1

import numpy as np

@challenge
def ch25():
    # boring: (x ^ key) ^ (a ^ key) ^ a == x
    pass

@challenge
def ch26():
    # boring ...
    pass

@challenge
def ch27():
    # boring ... the fun part is to construct c1, 0, c1, but already spoiled in
    # the problem description
    pass

@challenge
def ch28():
    import hashlib
    msg = b'hello, world'

    h = hashlib.sha1()
    h.update(msg)
    assert_eq(h.digest(), sha1(msg))
