# -*- coding: utf-8 -*-

from .utils import challenge, assert_eq, as_bytes
from .algo.hash import sha1, md_padding, dt_bigu32

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
    assert_eq(h.digest(), sha1(msg).tostring())

@challenge
def ch29():
    class Server:
        def __init__(self):
            self._key = np.random.bytes(np.random.randint(5, 60))

        @property
        def keylen(self):
            return len(self._key)

        def request(self, msg):
            return sha1(self._key + as_bytes(msg)).tostring()


    srv = Server()
    msg = (b"comment1=cooking%20MCs;userdata=foo;"
           b"comment2=%20like%20a%20pound%20of%20bacon")
    mac0 = srv.request(msg)
    payload = b";admin=true"
    msg1 = md_padding(msg, srv.keylen) + payload
    padded_size = ((len(msg) + srv.keylen-1)//64+1) * 64
    mac1 = sha1(payload, state=np.fromstring(mac0, dtype=dt_bigu32),
                nbytes_off=padded_size).tostring()

    assert_eq(srv.request(msg1), mac1 + b'x')
