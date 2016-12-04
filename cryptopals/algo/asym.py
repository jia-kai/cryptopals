# -*- coding: utf-8 -*-

"""asymmetric encryption algorithms"""

from .numt import gen_prime, next_prime, invmod, powmod

class RSA:
    class Opr:
        """encryption or decryption operator"""
        def __init__(self, e, n):
            self._e = e
            self._n = n

        def __call__(self, x):
            return powmod(x, self._e, self._n)

    @classmethod
    def make_enc_dec_pair(cls, bits=512, e=3):
        """make (encryptor, decryptor) opr pair"""
        while True:
            p = gen_prime(bits)
            if p % e != 1:
                break

        while True:
            q = gen_prime(bits)
            if q % e != 1:
                break

        n = p * q
        encr = cls.Opr(e, n)
        decr = cls.Opr(invmod(e, (p-1)*(q-1)), n)
        return encr, decr
