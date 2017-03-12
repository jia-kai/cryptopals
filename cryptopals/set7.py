# -*- coding: utf-8 -*-

from .utils import (challenge, assert_eq, as_bytes, capture_stdout,
                    as_np_bytearr, rangech)
from .bytearr import Bytearr
from .algo.block import cbc_encrypt, aes_ecb, pkcs7_pad
from .algo.hash import sha1
from .algo.numt import round_up

import numpy as np

import zlib
import itertools
import collections

@challenge
def ch49():
    # Too many assumptions without a concrete reference system (e.g. need
    # control on the first block, but does the bank system really allow this?)
    # And the whole challenge is still about xor ...
    return 'skipped'

@challenge
def ch50():
    # javascript is irrelevant; we use python for this demo

    class CBC_MAC:
        def __init__(self, key=b'YELLOW SUBMARINE', iv=b'\x00' * 16):
            self._key = key
            self._iv = iv
            self._encr_impl = aes_ecb(key).encryptor().update

        def __call__(self, msg):
            msg = pkcs7_pad(msg)
            return as_bytes(cbc_encrypt(self._iv, msg, self._encr_impl))[-16:]


    cbc_mac = CBC_MAC()
    assert_eq(Bytearr(cbc_mac("alert('MZA who was that?');\n")).to_hex(),
              '296b8d7cb78a243dda4d0a61d33bbdd1')

    code0 = "print('MZA who was that?');"
    mac = cbc_mac(code0)
    code1 = "print('Ayo, the Wu is back!');#"
    code1 += '#' * ((-len(code1)) % 16)
    code1 = code1.encode('utf-8')
    def get_suffix():
        # assume we know the key
        decr = aes_ecb(cbc_mac._key).decryptor().update
        def blk(out, iv):
            return as_bytes(as_np_bytearr(iv) ^ as_np_bytearr(decr(out)))
        pad = pkcs7_pad(b'')
        prev = blk(mac, pad)
        iv = cbc_encrypt(cbc_mac._iv, code1, cbc_mac._encr_impl)[-16:]
        return blk(prev, iv)

    code1 += get_suffix()
    assert_eq(cbc_mac(code1), mac)
    with capture_stdout() as s:
        exec(code1)
    s = s.getvalue()
    assert_eq(s, 'Ayo, the Wu is back!\n')

@challenge
def ch51():
    COOKIE = 'sessionid=TmV2ZXIgcmV2ZWFsIHRoZSBXdS1UYW5nIFNlY3JldCE='
    def compress_length(payload):
        data = f"""POST / HTTP/1.1
Host: hapless.com
Cookie: {COOKIE}
Content-Length: {len(payload)}
{payload}"""
        return len(zlib.compress(data.encode('utf-8')))

    recon = 'sessionid='
    charset = list(itertools.chain(rangech('a', 'z'), rangech('A', 'Z'),
                                   rangech('0', '9'), '+/='))
    assert len(charset) == 65
    max_len = compress_length(recon) + 1
    while True:
        prefix_cand = ['']
        # loop until find a unique suffix that results in minimal length
        while True:
            min_len = float('inf')
            min_len_data = None
            for prefix in prefix_cand:
                for guess in charset:
                    guess = prefix + guess
                    cur = compress_length(recon + guess)
                    if cur < min_len:
                        min_len = cur
                        min_len_data = [guess]
                    elif cur == min_len:
                        min_len_data.append(guess)
            if len(min_len_data) == 1:
                break
            prefix_cand = min_len_data
        if min_len > max_len:
            break
        cur, = min_len_data
        recon += cur
        if cur.endswith('='):
            charset = '='

    assert_eq(recon, COOKIE)
    return recon

@challenge
def ch52():
    # the key idea, is to brute-force an initial collision; the 2^(b/2)
    # complexity is based on birthday's paradox
    return 'skipped'

@challenge
def ch53():
    # the bottleneck is to find a bridge block, which requires brute-force; it
    # only reduces complexity from O(2^b) to O(2^b/n), where n is the message
    # length which should be infinitesimal compared to 2^b in practice.
    #
    # The essence for construction expandable messages is to utilize something
    # like binary encoding: construct multiple segments, each with two
    # collision messages of length (1, 2^k+1)
    return 'skipped'

@challenge
def ch54():
    # size in bytes
    HASH_STATE_SIZE = 2
    HASH_BLOCK_SIZE = 4

    randint32 = lambda *size: np.random.randint(
        2**32, size=size).astype(np.uint32)
    randint8 = lambda *size: np.random.randint(
        256, size=size).astype(np.uint8)

    class StateDict:
        assert HASH_STATE_SIZE == 2
        __impl = None

        def __init__(self, *items):
            self.__impl = {k[0] * 256 + k[1]: v for k, v in items}

        def __getitem__(self, k):
            return self.__impl[k[0] * 256 + k[1]]

        def __setitem__(self, k, v):
            self.__impl[k[0] * 256 + k[1]] = v

        def __delitem__(self, k):
            del self.__impl[k[0] * 256 + k[1]]

        def get(self, k):
            return self.__impl.get(k[0] * 256 + k[1])

    TreeLevel = collections.namedtuple(
        'TreeLevel', [
            'prev_s2i', # StateDict that maps from previous states to index
            'blocks',   # messsage blocks for state transition, shape(N, b)
            'next',     # collided states, shape (N//2, b)
        ])

    nr_compress_compute = 0
    def batched_compress(
            block, state, *,
            init_block=randint32(16, 1), init_state=randint32(5, 1)):
        nonlocal nr_compress_compute
        N = block.shape[1]
        nr_compress_compute += N
        if N == 1:
            init_block = init_block.copy()
            init_state = init_state.copy()
        else:
            assert N > 1
            rep = lambda x: np.ascontiguousarray(np.repeat(x, N, axis=1))
            init_block = rep(init_block)
            init_state = rep(init_state)
        init_block[:HASH_BLOCK_SIZE] = block
        init_state[:HASH_STATE_SIZE] = state

        s1 = sha1.batched_compress(init_block, init_state)
        return s1[:HASH_STATE_SIZE].astype(np.uint8)

    def build_tree_level(states):
        """make a :class:`TreeLevel` from init states; complexity is
        O(len(states) * 2^(b/2))"""
        N = states.shape[1]
        assert N % 2 == 0
        ret = TreeLevel(StateDict(),
                        np.empty((N, HASH_BLOCK_SIZE), dtype=np.uint8),
                        np.empty((N//2, HASH_STATE_SIZE), dtype=np.uint8))

        def find_and_record_collision(scomp, N):
            nonlocal nr_finished, states
            assert states.shape[0] == HASH_STATE_SIZE, (
                nr_finished, states.shape)
            assert N + nr_finished == ret.blocks.shape[0], (
                nr_finished, N, ret.blocks.shape)

            scomp2idx = StateDict()
            removed_pos = set()
            nr_remove = 0
            for idx, val in enumerate(scomp.T):
                old_idx = scomp2idx.get(val)
                idx_s = idx % N

                if idx_s in removed_pos:
                    continue

                if old_idx is None:
                    scomp2idx[val] = idx
                    continue

                old_idx_s = old_idx % N
                if old_idx_s == idx_s or old_idx_s in removed_pos:
                    continue

                ret.prev_s2i[states[:, old_idx_s]] = nr_finished
                ret.blocks[nr_finished] = try_block[:, old_idx]
                ret.prev_s2i[states[:, idx_s]] = nr_finished + 1
                ret.blocks[nr_finished+1] = try_block[:, idx]
                ret.next[nr_finished//2] = val
                nr_finished += 2
                removed_pos.add(old_idx_s)
                removed_pos.add(idx_s)
                nr_remove += 1
                del scomp2idx[val]

                if len(removed_pos) == N:
                    break

            assert len(removed_pos) == nr_remove * 2
            if not removed_pos:
                return False

            write_pos = read_pos = 0
            while read_pos < N:
                if read_pos not in removed_pos:
                    states[:, write_pos] = states[:, read_pos]
                    write_pos += 1
                read_pos += 1
            assert write_pos + len(removed_pos) == N
            states = states[:, :write_pos]
            return True

        N0 = N
        nr_finished = 0
        while nr_finished < N0:
            N = states.shape[1]
            rep = max(2**(HASH_STATE_SIZE * 8 // 2) // N, 1)
            try_block = np.empty((HASH_BLOCK_SIZE, N * rep), dtype=np.uint8)
            states_rep = np.tile(states, [1, rep])
            while not find_and_record_collision(
                    batched_compress(try_block, states_rep),
                    N):
                i = np.random.randint(HASH_BLOCK_SIZE)
                try_block[i] = np.random.randint(256, size=N * rep)

        return ret


    def build_tree(nr_level):
        """generate a list of :class:`TreeLevel` representing the tree

        :return: tree, final_hash_state
        """
        ret = []
        prev = randint8(HASH_STATE_SIZE, 2**nr_level)
        for i in range(nr_level):
            sub = build_tree_level(prev)
            ret.append(sub)
            prev = np.ascontiguousarray(sub.next.T)
        assert prev.shape[1] == 1, (nr_level, prev.shape)
        return ret, prev.reshape(HASH_STATE_SIZE)

    def compute_hash(message, *, add_padding=True,
                     init_state=randint8(HASH_STATE_SIZE)):
        if add_padding:
            message = sha1.pad(message)
        message = as_np_bytearr(message).reshape(-1, HASH_BLOCK_SIZE)
        state = init_state[:, np.newaxis]
        for i in message:
            state = batched_compress(i[:, np.newaxis], state)
        return state.reshape(HASH_STATE_SIZE)

    NR_LEVEL = HASH_STATE_SIZE * 8 // 2
    MESSAGE = 'This is a lie.'
    tree, endpoint_state = build_tree(NR_LEVEL)

    padding = sha1.pad(b'', round_up(len(MESSAGE), HASH_BLOCK_SIZE) +
                       HASH_BLOCK_SIZE * (1 + NR_LEVEL))
    forged_hash = compute_hash(padding, add_padding=False,
                               init_state=endpoint_state)

    forged_message = MESSAGE.encode('utf-8')
    forged_message += b'\x00' * (round_up(len(MESSAGE), HASH_BLOCK_SIZE) -
                                 len(MESSAGE))
    state0 = compute_hash(forged_message, add_padding=False)[:, np.newaxis]
    while True:
        try_block = randint8(HASH_BLOCK_SIZE, 1)
        state1 = batched_compress(try_block, state0).reshape(HASH_STATE_SIZE)
        idx = tree[0].prev_s2i.get(state1)
        if idx is not None:
            forged_message += as_bytes(try_block.reshape(HASH_BLOCK_SIZE))
            for i in range(NR_LEVEL):
                idx = tree[i].prev_s2i.get(state1)
                forged_message += as_bytes(tree[i].blocks[idx])
                state1 = tree[i].next[idx // 2]
            assert np.alltrue(state1 == endpoint_state)
            break

    assert_eq(compute_hash(forged_message), forged_hash)
    return forged_message, nr_compress_compute
