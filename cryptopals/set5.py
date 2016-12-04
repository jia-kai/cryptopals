# -*- coding: utf-8 -*-

from .utils import challenge, assert_eq
from .utils.emucs import run_cs_session
from .algo.key_exchange import NIST_PRIME, DiffieHellman, SRP
from .algo.numt import bytes2int, int2bytes, solve_congruences
from .algo.hash import sha256, hmac
from .algo.asym import RSA

import numpy as np
import gmpy2

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

    del client.result
    client._password += b'x'
    run_cs_session(server, client)
    assert_eq(client.result, False)

@challenge
def ch37():
    # trival and boring
    # but why not replace salt with something derived from (salt, password) in
    # the final HMAC step?
    return 'skipped'

@challenge
def ch38():
    password_dict = [np.random.bytes(np.random.randint(5, 32))
                     for _ in range(32)]

    param = SRP.Param()
    param.k = 0 # simplify the protocal
    password = None
    async def fake_server(socket):
        nonlocal password
        salt = b's'
        _, A = await socket.recv()
        socket.send((salt, param.g))
        sig_cli = await socket.recv()
        await socket.send(True).wait()

        u = bytes2int(sha256(int2bytes(A) + int2bytes(param.g)))
        for guess_passwd in password_dict:
            x = bytes2int(sha256(salt + guess_passwd))
            S = (A * param.powmod_g(u * x)) % param.N
            K = sha256(int2bytes(S))
            if hmac(K, salt, sha256) == sig_cli:
                password = guess_passwd
                return

    client = SRP.Client(param, 'test',
                        password_dict[np.random.randint(len(password_dict))])
    run_cs_session(fake_server, client)
    assert_eq(password, client._password)

@challenge
def ch39():
    encr, decr = RSA.make_enc_dec_pair()
    s = b'hello, world'
    assert_eq(s, int2bytes(decr(encr(bytes2int(s)))))

@challenge
def ch40():
    msg = bytes2int(np.random.bytes(20))
    ciphertexts = []
    for i in range(3):
        encr, _ = RSA.make_enc_dec_pair(256, e=3)
        ciphertexts.append((encr(msg), encr._n))

    _, b = solve_congruences(*zip(*ciphertexts))
    m, exact = gmpy2.iroot(b, 3)
    assert_eq(int(m), msg)
