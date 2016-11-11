# -*- coding: utf-8 -*-

from .bytearr import Bytearr
from .utils import challenge, assert_eq, open_resource, popcount8

import numpy as np
import itertools

@challenge
def ch01():
    data = Bytearr.from_hex(
        '49276d206b696c6c696e6720796f757220627261696e206c'
        '696b65206120706f69736f6e6f7573206d757368726f6f6d')
    assert_eq(data.to_base64(),
              'SSdtIGtpbGxpbmcgeW91ciBicm'
              'FpbiBsaWtlIGEgcG9pc29ub3VzIG11c2hyb29t')

@challenge
def ch02():
    a = Bytearr.from_hex('1c0111001f010100061a024b53535009181c')
    b = Bytearr.from_hex('686974207468652062756c6c277320657965')
    c = Bytearr.from_hex('746865206b696420646f6e277420706c6179')
    assert_eq(a ^ b, c)

def crack_single_byte_xor(cipher, need_key=False):
    """:return: (score, plain)"""
    if isinstance(cipher, Bytearr):
        cipher = cipher.np_data
    assert isinstance(cipher, np.ndarray) and cipher.dtype == np.uint8
    keys = np.arange(256, dtype=np.uint8)[:, np.newaxis]
    cand = cipher[np.newaxis] ^ keys
    score = np.logical_and(cand >= ord('a'), cand <= ord('z')).sum(axis=1)
    key = np.argmax(score)
    ret = score[key], Bytearr(cand[key])
    if need_key:
        ret += (key, )
    return ret

@challenge
def ch03():
    cipher = Bytearr.from_hex('1b37373331363f78151b7f2b783431333d78'
                              '397828372d363c78373e783a393b3736')
    _, p = crack_single_byte_xor(cipher)
    return p.to_str()

@challenge
def ch04():
    best_score = 0
    with open_resource() as fin:
        for line_num, line in enumerate(fin):
            score, plain = crack_single_byte_xor(Bytearr.from_hex(line))
            if score > best_score:
                best_score = score
                best_plain = plain
                best_num = line_num + 1
    return best_num, best_plain.to_str()

@challenge
def ch05():
    plain = Bytearr("Burning 'em, if you ain't quick and nimble\n"
                    "I go crazy when I hear a cymbal")
    key = itertools.cycle(map(ord, 'ICE'))
    dest = Bytearr.from_hex(
        '0b3637272a2b2e63622c2e69692a23693a2a3c6324202d623d'
        '63343c2a26226324272765272a282b2f20430a652e2c652a31'
        '24333a653e2b2027630c692b20283165286326302e27282f')
    assert_eq(plain ^ key, dest)

@challenge
def ch06():
    assert_eq(
        popcount8(
            np.array(list(map(ord, 'this is a test')), dtype=np.uint8) ^
            np.array(list(map(ord, 'wokka wokka!!!')), dtype=np.uint8)
        ),
        37
    )
    with open_resource() as fin:
        data_br = Bytearr.from_base64(fin.read())
    data = data_br.np_data
    NR_PAIRS = 8
    scores = []
    for i in range(2, 41):
        chunks = data[:i * NR_PAIRS * 2].reshape(NR_PAIRS, 2, i)
        scores.append((
            sum(popcount8(p[0] ^ p[1]) / i for p in chunks) / NR_PAIRS,
            i))

    scores.sort()

    best_score = 0
    for _, keysize in scores[:3]:
        cur_score = 0
        cur_key = []
        for i in range(keysize):
            s, _, k = crack_single_byte_xor(data[i::keysize], need_key=True)
            cur_score += s
            cur_key.append(k)
        if cur_score > best_score:
            best_score = cur_score
            key = cur_key
    plain = Bytearr(data_br ^ itertools.cycle(key)).to_str()
    return ''.join(map(chr, key)), '{}...{}'.format(plain[:10], plain[-10:])
