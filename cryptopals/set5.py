# -*- coding: utf-8 -*-

from .utils import challenge, assert_eq
from .utils.emucs import run_cs_session
from .algo.key_exchange import NIST_PRIME, DiffieHellman, SRP

import numpy as np

import functools

@challenge
def ch33():
    dh = DiffieHellman(37, 5)
    sess0 = dh.make_session()
    sess1 = dh.make_session()
    assert_eq(sess0.get_session_key(sess1.pubkey),
              sess1.get_session_key(sess0.pubkey))


    p = NIST_PRIME
    g = 2

    dh = DiffieHellman(p, g)
    sess0 = dh.make_session()
    sess1 = dh.make_session()
    assert_eq(sess0.get_session_key(sess1.pubkey),
              sess1.get_session_key(sess0.pubkey))

@challenge
def ch34():
    # no interesting math involved
    return 'skipped'

@challenge
def ch35():
    # no interesting math involved
    # {1, p, p-1} ** n % p is too trival
    return 'skipped'

@challenge
def ch36_check_sha256():
    from .algo.hash import sha256
    import hashlib

    assert_eq(
        sha256('', outhex=True),
        'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855')

    for i in range(10):
        msg = np.random.bytes(np.random.randint(1, 2000))
        h = hashlib.sha256()
        h.update(msg)
        assert_eq(h.digest(), sha256(msg))

@challenge
def ch36():
    server, client = SRP.make_server_client_pair()
    run_cs_session(server, client)
    assert_eq(client.result, True)
    # in fact, both server and client computes S = g**((a+x*u)*b) % N
