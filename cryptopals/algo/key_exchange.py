# -*- coding: utf-8 -*-

"""key-exchange algorithms, including password-authenticated key agreement
methods"""

from ..utils import as_bytes
from .numt import powmod, large_randint, hex2int, bytes2int, int2bytes
from .hash import sha256, hmac
import numpy as np

NIST_PRIME = hex2int(
    'ffffffffffffffffc90fdaa22168c234c4c6628b80dc1cd129024'
    'e088a67cc74020bbea63b139b22514a08798e3404ddef9519b3cd'
    '3a431b302b0a6df25f14374fe1356d6d51c245e485b576625e7ec'
    '6f44c42e9a637ed6b0bff5cb6f406b7edee386bfb5a899fa5ae9f'
    '24117c4b1fe649286651ece45b3dc2007cb8a163bf0598da48361'
    'c55d39a69163fa8fd24cf5f83655d23dca3ad961c62f356208552'
    'bb9ed529077096966d670c354e4abc9804f1746c08ca237327fff'
    'fffffffffffff')

class DiffieHellman:
    class Session:
        _dh = None
        _priv = None

        def __init__(self, dh):
            self._dh = dh
            self._priv = large_randint(dh._p)

        @property
        def pubkey(self):
            return powmod(self._dh._g, self._priv, self._dh._p)

        def get_session_key(self, peer_pub):
            return powmod(peer_pub, self._priv, self._dh._p)

    _p = None
    _g = None
    def __init__(self, p, g):
        """p is a prime, and g is a primitive root modulo p"""
        self._p = p
        self._g = g

    def make_session(self):
        return self.Session(self)


class SRP:
    """secure remote password"""

    class Param:
        N = NIST_PRIME
        """a large safe prime(i.e. N = 2q+1 where q is also prime)"""

        g = 2
        """primitive root of N"""

        k = 3
        """used to prevent a 2-for-1 guess when an active attacker impersonates
        the server"""

        password = None
        """aggreed password

        :type: bytes
        """

        def powmod(self, a, x):
            return powmod(a, x, self.N)
        def powmod_g(self, x):
            return powmod(self.g, x, self.N)

        def make_dh_session(self):
            return DiffieHellman(self.N, self.g).make_session()

    class Server:
        _param = None

        def __init__(self, param):
            assert isinstance(param, SRP.Param)
            self._param = param

        async def __call__(self, socket):
            param = self._param

            salt = np.random.bytes(np.random.randint(4, 16))
            x = bytes2int(sha256(salt + param.password))
            v = param.powmod_g(x)
            # password is not needed further

            login, A = await socket.recv()

            b = large_randint(param.N)
            B = (param.k * v + param.powmod_g(b)) % param.N
            socket.send((salt, B))

            u = bytes2int(sha256(int2bytes(A) + int2bytes(B)))
            S = param.powmod(A * param.powmod(v, u), b)
            K = sha256(int2bytes(S))

            sig_cli = await socket.recv()
            sig_srv = hmac(K, salt, sha256)
            await socket.send(sig_cli == sig_srv).wait()

    class Client:
        _param = None
        _login = None

        result = None

        def __init__(self, param, login):
            assert isinstance(param, SRP.Param)
            self._param = param
            self._login = login

        async def __call__(self, socket):
            param = self._param

            a = large_randint(param.N)
            A = param.powmod_g(a)
            socket.send((self._login, A))

            salt, B = await socket.recv()
            u = bytes2int(sha256(int2bytes(A) + int2bytes(B)))
            x = bytes2int(sha256(salt + param.password))
            S = param.powmod(B - param.k * param.powmod_g(x), a + u * x)
            K = sha256(int2bytes(S))

            socket.send(hmac(K, salt, sha256))
            self.result = await socket.recv()

    @classmethod
    def make_server_client_pair(cls, password='test', login='guest',
                                param=None):
        if param is None:
            param = SRP.Param()
        param.password = as_bytes(password)
        return cls.Server(param), cls.Client(param, login)
