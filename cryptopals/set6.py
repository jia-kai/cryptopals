# -*- coding: utf-8 -*-

from .utils import challenge, assert_eq
from .algo.numt import powmod, invmod, bytes2int
from .algo.asym import RSA

import numpy as np

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
