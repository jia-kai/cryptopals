# -*- coding: utf-8 -*-

from .utils import (challenge, assert_eq, open_resource, CipherError, as_bytes,
                    summarize_str)
from .algo.block import (aes_ecb, pkcs7_pad, pkcs7_unpad, cbc_encrypt,
                         cbc_decrypt, as_np_bytearr, ctr_encrypt)
from .algo.rng import MT19937
from .bytearr import Bytearr

import numpy as np

@challenge
def ch17():
    class Server:
        _data = None
        _encr_impl = None
        _decr_impl = None

        def __init__(self):
            self._data = data = []
            with open_resource() as fin:
                for line in fin:
                    data.append(Bytearr.from_base64(line))
            enc = aes_ecb(np.random.bytes(16))
            self._encr_impl = enc.encryptor().update
            self._decr_impl = enc.decryptor().update

        def query(self):
            iv = Bytearr(np.random.bytes(16), allow_borrow=True)
            data = self._data[np.random.randint(len(self._data))]
            data = pkcs7_pad(data)
            return iv, cbc_encrypt(iv, data, self._encr_impl).np_data

        def check(self, iv, ciphertext):
            data = cbc_decrypt(iv, ciphertext, self._decr_impl)
            try:
                pkcs7_unpad(data)
                return True
            except CipherError:
                return False

    srv = Server()
    iv, ciphertext = srv.query()
    block_size = 16
    assert len(ciphertext) % block_size == 0

    def decr_block(ciphertext):
        # decrypt a single block

        def check_single(iv):
            # assume (iv ^ plain)[-1] is '\x01'
            plain = np.empty_like(iv)
            plain[-1] = iv[-1] ^ 1
            for pad_size in range(2, block_size + 1):
                iv[-pad_size+1:] = plain[-pad_size+1:] ^ pad_size
                found = False
                for test in range(256):
                    iv[-pad_size] = test
                    if srv.check(iv, ciphertext):
                        found = True
                        break
                if not found:
                    return
                plain[-pad_size] = iv[-pad_size] ^ pad_size
            return plain

        while True:
            iv = as_np_bytearr(np.random.bytes(16))
            if not srv.check(iv, ciphertext):
                continue
            ret = check_single(iv)
            if ret is not None:
                return ret

    plain = []
    for i in range(0, len(ciphertext), block_size):
        cur = decr_block(ciphertext[i:i+block_size])
        plain.extend(map(int, iv ^ cur))
        iv = ciphertext[i:i+block_size]
    return pkcs7_unpad(plain).decode('ascii')

@challenge
def ch18():
    data = ('L77na/nrFsKvynd6HzOoG7GHTLXsTVu9qvY'
            '/2syLXzhPweyyMTJULu/6/kXX0KSvoOLSFQ==')
    return as_bytes(ctr_encrypt(
        aes_ecb('YELLOW SUBMARINE').encryptor().update,
        Bytearr.from_base64(data))).decode('ascii')

@challenge
def ch19():
    data, key = ch19_stat(lambda x: x // 2)

    key.append(ord('g') ^ 103); key.append(ord('h') ^ 104)
    key.append(ord('n') ^ 110); key.append(ord('d') ^ 100)
    key.append(ord('d') ^ 100)
    key.extend(as_np_bytearr('ead') ^ [101, 97, 100])
    key.extend(as_np_bytearr('n,') ^ [110, 44])
    key = as_np_bytearr(key)
    plain = []
    for i in data:
        plain.extend(key[:len(i)] ^ i)
        plain.append(ord(' '))
    return summarize_str(as_bytes(plain).decode('ascii'))

def ch19_stat(thresh):
    data = []
    with open_resource() as fin:
        for line in fin:
            data.append(Bytearr.from_base64(line).np_data)

    key = []
    thresh = thresh(len(data))
    for i in range(0, max(map(len, data))):
        cipher = np.array([j[i] for j in data if len(j) > i])
        if len(cipher) <= thresh:
            break
        keys = np.arange(256, dtype=np.uint8)[:, np.newaxis]
        cand = cipher[np.newaxis] ^ keys
        score = np.logical_and(cand >= ord('a'), cand <= ord('z')).sum(axis=1)
        key.append(np.argmax(score))
    return data, key

@challenge
def ch20():
    data, key = ch19_stat(lambda x: min(x // 3, 10))
    if False:
        k = len(key)
        for i in data:
            if len(i) >= len(key):
                print(as_bytes(as_np_bytearr(i[:k] ^ key[:k])), i[k:])
    key.extend(
        as_np_bytearr('rve the whole scenery') ^
        [114,118,101,32,116,104,101,32,119,104,111,108,101,32,115,99,101,
         110,101,114,121])

    plain = []
    for i in data:
        cur = as_np_bytearr(key[:len(i)] ^ i)
        #print(as_bytes(cur))
        plain.extend(cur)
        plain.append(ord(' '))
    return summarize_str(as_bytes(plain).decode('ascii'))

@challenge
def ch21():
    def _int32(x):
        # Get the 32 least significant bits.
        return int(0xFFFFFFFF & x)

    class MT19937Ref:
        def __init__(self, seed):
            # Initialize the index to 0
            self.index = 624
            self.mt = [0] * 624
            self.mt[0] = seed  # Initialize the initial state to the seed
            for i in range(1, 624):
                self.mt[i] = _int32(
                    1812433253 * (self.mt[i - 1] ^ self.mt[i - 1] >> 30) + i)

        def extract_number(self):
            if self.index >= 624:
                self.twist()

            y = self.mt[self.index]

            # Right shift by 11 bits
            y = y ^ y >> 11
            # Shift y left by 7 and take the bitwise and of 2636928640
            y = y ^ y << 7 & 2636928640
            # Shift y left by 15 and take the bitwise and of y and 4022730752
            y = y ^ y << 15 & 4022730752
            # Right shift by 18 bits
            y = y ^ y >> 18

            self.index = self.index + 1

            return _int32(y)

        def twist(self):
            for i in range(624):
                # Get the most significant bit and add it to the less
                # significant bits of the next number
                y = _int32((self.mt[i] & 0x80000000) +
                           (self.mt[(i + 1) % 624] & 0x7fffffff))
                self.mt[i] = self.mt[(i + 397) % 624] ^ y >> 1

                if y % 2 != 0:
                    self.mt[i] = self.mt[i] ^ 0x9908b0df
            self.index = 0

    r0 = MT19937(42)
    r1 = MT19937Ref(42)
    for i in range(5000):
        assert_eq(r0(), r1.extract_number(), i)

@challenge
def ch22():
    # just try some numbers before current timestamp
    pass

@challenge
def ch23():
    rng = MT19937(np.random.randint(2**32))
    cloned = MT19937(0)

    w = np.dtype(rng.dtype).itemsize * 8

    def to_bits_le(x):
        """convert x to little-endian bits"""
        ret = list(map(int, bin(x)[2:]))[::-1]
        ret += [0] * (w - len(ret))
        return ret

    def from_bits_le(bits):
        return rng.dtype(sum(j << i for i, j in enumerate(bits)))

    def inv_xsa(x, s, a):
        """return y such that x == y ^ ((y << s) & a); s can be negative"""
        if s < 0:
            ss = -s
            rev = lambda x: x[::-1]
        else:
            ss = s
            rev = lambda x: x
        ybits = rev(to_bits_le(x))
        abits = rev(to_bits_le(a))
        for i in range(ss, len(ybits)):
            ybits[i] ^= ybits[i - ss] & abits[i]
        y = from_bits_le(rev(ybits))
        if s < 0:
            assert (y ^ ((y >> ss) & a)) == x
        else:
            assert (y ^ ((y << ss) & a)) == x
        return y

    n, m, r = rng._nmr
    a, b, c, s, t, u, d, l = rng._abcstudl
    b, c, d = map(rng.dtype, (b, c, d))
    for i in range(n):
        x = rng()
        x = inv_xsa(x, -l, ~rng.dtype(0))
        x = inv_xsa(x, t, c)
        x = inv_xsa(x, s, b)
        x = inv_xsa(x, -u, d)
        cloned._state[i] = x

    for i in range(5000):
        assert_eq(cloned(), rng(), i)
