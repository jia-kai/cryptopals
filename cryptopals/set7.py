# -*- coding: utf-8 -*-

from .utils import (challenge, assert_eq, as_bytes, capture_stdout,
                    as_np_bytearr, rangech)
from .bytearr import Bytearr
from .algo.block import cbc_encrypt, aes_ecb, pkcs7_pad
from .algo.hash import sha1, md4
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

def ch55_impl():
    lrot = md4._left_rotate

    def rrot(n, b):
        return (n >> np.uint32(b)) | (n << np.uint32(32 - b))

    DEFAULT_INIT_STATE = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476]

    IDX = [
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
        [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]],
        [[0, 8, 4, 12], [2, 10, 6, 14], [1, 9, 5, 13], [3, 11, 7, 15]]
    ]

    SFT = [
        [3, 7, 11, 19],
        [3, 5, 9, 13],
        [3, 9, 11, 15]
    ]

    def gen_c5_req(template, first_empty=True):
        return ', '.join(
            map(template.format, [i - 9 for i in [26, 27, 29, 32]]))

    ROUND1_REQUIREMENTS = [
        'a1,7 = b0,7',
        'd1,7 = 0, d1,8 = a1,8, d1,11 = a1,11',
        'c1,7 = 1, c1,8 = 1, c1,11 = 0, c1,26 = d1,26',
        'b1,7 = 1, b1,8 = 0, b1,11 = 0, b1,26 = 0',
        'a2,8 = 1, a2,11 = 1, a2,26 = 0, a2,14 = b1,14',
        ('d2,14 = 0, d2,19 = a2,19, d2,20 = a2,20, d2,21 = a2,21, '
         'd2,22 = a2,22, d2,26 = 1'),
        ('c2,13 = d2,13, c2,14 = 0, c2,15 = d2,15, c2,19 = 0, '
         'c2,20 = 0, c2,21 = 1, c2,22 = 0'),
        ('b2,13 = 1, b2,14 = 1, b2,15 = 0, b2,17 = c2,17, b2,19 = 0, '
         'b2,20 = 0, b2,21 = 0, b2,22 = 0'),
        ('a3,13 = 1, a3,14 = 1, a3,15 = 1, a3,17 = 0, a3,19 = 0, '
         'a3,20 = 0, a3,21 = 0, a3,23 = b2,23, a3,22 = 1, a3,26 = b2,26'),
        ('d3,13 = 1, d3,14 = 1, d3,15 = 1, d3,17 = 0, d3,20 = 0, '
         'd3,21 = 1, d3,22 = 1, d3,23 = 0, d3,26 = 1, d3,30 = a3,30'),
        ('c3,17 = 1, c3,20 = 0, c3,21 = 0, c3,22 = 0, c3,23 = 0, '
         'c3,26 = 0, c3,30 = 1, c3,32 = d3,32'),
        ('b3,20 = 0, b3,21 = 1, b3,22 = 1, b3,23 = c3,23, b3,26 = 1, '
         'b3,30 = 0, b3,32 = 0'),
        ('a4,23 = 0, a4,26 = 0, a4,27 = b3,27, a4,29 = b3,29, '
         'a4,30 = 1, a4,32 = 0'),
        'd4,23 = 0, d4,26 = 0, d4,27 = 1, d4,29 = 1, d4,30 = 0, d4,32 = 1',
        'c4,19 = d4,19, c4,23 = 1, c4,26 = 1, c4,27 = 0, c4,29 = 0, c4,30 = 0',
        'b4,19 = 0, b4,26 = c4,26, b4,27 = 1, b4,29 = 1, b4,30 = 0'
    ]
    assert len(ROUND1_REQUIREMENTS) == 16
    ROUND1_REQUIREMENTS_C5 = [
        'b1,20 = 0',
        gen_c5_req('d2,{} = 0'),
        gen_c5_req('a2,{0} = b1,{0}'),
        gen_c5_req('c2,{} = 0'),
        gen_c5_req('b2,{} = 0'),
    ]
    ROUND2_REQUIREMENTS = [
        'a5,19 = c4,19, a5,26 = 1, a5,27 = 0, a5,29 = 1, a5,32 = 1',
        ('d5,19 = a5,19, d5,26 = b4,26, d5,27 = b4,27, '
         'd5,29 = b4,29, d5,32 = b4,32'),
        ('c5,26 = d5,26, c5,27 = d5,27, c5,29 = d5,29, '
         'c5,30 = d5,30, c5,32 = d5,32'),
        'b5,29 = c5,29, b5,30 = 1, b5,32 = 0',
        'a6,29 = 1, a6,32 = 1',
        'd6,29 = b5,29',
        'c6,29 = d6,29'     # c6,30 = d6,30 + 1, c6,32 = d6,32 + 1 omitted
    ]
    ROUND3_REQUIREMENTS = [
        'b9,32 = 1',
        'a10,32 = 1'
    ]

    def F(x, y, z):
        return x & y | ~x & z
    def G(x, y, z):
        return x & y | x & z | y & z
    def H(x, y, z):
        return x ^ y ^ z

    def md4_collide(m, init_state=DEFAULT_INIT_STATE, *, compute_only=False):
        """
        :param m: 512-bit message block
        :return: message pairs (m0, m1) based on m that are likely to collide
        """
        assert (isinstance(m, np.ndarray) and
                m.dtype == np.uint32 and
                m.shape == (16, ))
        phi = [
            lambda a, b, c, d, k, s: (
                lrot(a + F(b, c, d) + m[k], s)),
            lambda a, b, c, d, k, s: (
                lrot(a + G(b, c, d) + m[k] + np.uint32(0x5a827999), s)),
            lambda a, b, c, d, k, s: (
                lrot(a + H(b, c, d) + m[k] + np.uint32(0x6ed9eba1), s))
        ]

        phi_inv = [
            lambda y, a, b, c, d, s: (
                rrot(y, s) - a - F(b, c, d))
        ]
        """solve m such that phi[i](a, b, c, d, m, s) == y"""

        def inv_phi0_msg(y, a, b, c, d, s):
            return rrot(y, s)

        # compute all internal states
        a, b, c, d = ([np.uint32(i)] for i in init_state)
        def compute_internal_states(combine=False):
            for i in a, b, c, d:
                del i[1:]
            for rnd in range(3):
                p = phi[rnd]
                sft = SFT[rnd]
                for i in range(4):
                    idx = IDX[rnd][i]
                    a.append(p(a[-1], b[-1], c[-1], d[-1], idx[0], sft[0]))
                    d.append(p(d[-1], a[-1], b[-1], c[-1], idx[1], sft[1]))
                    c.append(p(c[-1], d[-1], a[-1], b[-1], idx[2], sft[2]))
                    b.append(p(b[-1], c[-1], d[-1], a[-1], idx[3], sft[3]))
            if combine:
                mat = [[a[i], d[i], c[i], b[i]] for i in range(1, 12)]
                arr = np.array(mat).flatten()
                return arr[:-3]

        get_final_state = lambda: np.array([a[-1], b[-1], c[-1], d[-1]])

        if compute_only:
            compute_internal_states()
            ret = get_final_state()
            ret += a[0], b[0], c[0], d[0]
            return ret.tobytes()

        class VarBitRef:
            """reference to a bit in a var in the format in the paper, like
            ``'a5,19'``"""

            _var_map = dict(a=a, b=b, c=c, d=d)
            _vlist = None
            _vidx = None
            _bit = None
            _spec = None
            _varspec = None

            def __init__(self, spec):
                self._spec = spec
                var, bit = spec.split(',')
                bit = int(bit) - 1
                assert 0 <= bit < 32
                self._vlist = self._var_map[var[0]]
                self._vidx = int(var[1:])
                self._varspec = '{}{}'.format(var[0], self._vidx)
                self._bit = np.uint32(bit)

            @property
            def val32(self):
                return self._vlist[self._vidx]

            @val32.setter
            def val32(self, val):
                self._vlist[self._vidx] = np.uint32(val)

            @property
            def bitus(self):
                """unshifted bit value"""
                return self.val32 & self.shifted()

            @property
            def bitsf(self):
                """bit shifted to the least-significant position"""
                return np.uint32((self.val32 >> self._bit) & 1)

            def shifted(self, init=1):
                """get ``init << shift``"""
                return np.uint32(init) << self._bit

            @property
            def shift(self):
                """bit shift value"""
                return int(self._bit)

            @property
            def spec(self):
                """original text spec"""
                return self._spec

            def __repr__(self):
                return 'VarBitRef({})'.format(self._spec)

        class ConstBit:
            _val = None

            def __init__(self, val):
                val = int(val)
                assert val in (0, 1)
                self._val = val

            @property
            def bitsf(self):
                """see :meth:`~.VarBitRef.bitsf`"""
                return self._val

            @property
            def spec(self):
                return self._val

            def __repr__(self):
                return 'ConstBit({})'.format(self._val)

        def parse_var_spec(spec):
            """parse a var spec to :class:`VarBitRef` or :class:`ConstBit`"""
            if ',' in spec:
                return VarBitRef(spec)
            return ConstBit(spec)

        def refresh_m_from_internal():
            rd = 0  # only use round-1 states
            inv = phi_inv[rd]
            sft = SFT[rd]
            saved_round_1 = [i[:5] for i in (a, b, c, d)]
            for i in range(4):
                idx = IDX[rd][i]
                m[idx[0]] = inv(a[i+1], a[i], b[i], c[i], d[i], sft[0])
                m[idx[1]] = inv(d[i+1], d[i], a[i+1], b[i], c[i], sft[1])
                m[idx[2]] = inv(c[i+1], c[i], d[i+1], a[i+1], b[i], sft[2])
                m[idx[3]] = inv(b[i+1], b[i], c[i+1], d[i+1], a[i+1], sft[3])
            compute_internal_states()

            new_round_1 = [i[:5] for i in (a, b, c, d)]
            for i, j in zip(saved_round_1, new_round_1):
                assert i == j

        def iter_txt_eqn_group(eqn):
            """iterate over text equation group in the format
            ``a=b, [c=d,...]``

            :return: iterator for (lhs, rhs) pairs for the equations
            """
            for stmt in eqn.split(', '):
                lhs, rhs = stmt.split(' = ')
                lhs = VarBitRef(lhs)
                rhs = parse_var_spec(rhs)
                yield lhs, rhs

        def fix_round1():
            """all internal states in first round are free variables, forming a
            one-to-one map with m"""
            remap = dict(a='0', d='1', c='2', b='3')
            for i in sorted(ROUND1_REQUIREMENTS + ROUND1_REQUIREMENTS_C5,
                            key=lambda x: x[1] + remap[x[0]]):
                for lhs, rhs in iter_txt_eqn_group(i):
                    lhs.val32 = lhs.val32 ^ lhs.bitus ^ lhs.shifted(rhs.bitsf)
            refresh_m_from_internal()

        def fix_round2_iter1():
            """the key observation is that ``req`` and ``a[idx+1]`` share
            message ``m[idx*4]``, so we can satisfy ``req`` by inverting bits
            in ``a[idx+1]``, which causes add/sub to ``m[idx*4]``

            However we can not do the same with b5, since satisfying b5
            requires modifying a4, and to maintain a5 unchanged, we need to
            modify m0, causing a1 to change, resulting in either a[1:5] changed
            (while fixing m[[4,8.12]] unchanged) or m[1:5] changed (while
            fixing all states unchanged)
            """

            round1_used = set()
            for i in ROUND1_REQUIREMENTS:
                for lhs, rhs in iter_txt_eqn_group(i):
                    round1_used.add(lhs.spec)
                    round1_used.add(rhs.spec)

            for idx, req in enumerate(ROUND2_REQUIREMENTS[:3]):
                lhs = None
                for lhs, rhs in iter_txt_eqn_group(req):
                    if lhs.bitsf != rhs.bitsf:
                        tgt_shift_sumv = lhs.shift - SFT[1][idx]
                        tgt_shift = tgt_shift_sumv + SFT[0][0]
                        tgt = VarBitRef('a{},{}'.format(
                            idx + 1, tgt_shift % 32 + 1))
                        if tgt.spec in round1_used:
                            try_fix_c5(lhs, rhs)
                            continue

                        delta = np.uint32(1 << (tgt_shift_sumv % 32))
                        lhs_sumv = rrot(lhs.val32, SFT[1][idx])
                        if tgt.bitus:
                            lhs_sumv -= delta
                        else:
                            lhs_sumv += delta
                        tgt.val32 ^= tgt.shifted()
                        lhs.val32 = lrot(lhs_sumv, SFT[1][idx])
                        assert lhs.bitsf == rhs.bitsf

                expected_val = lhs.val32
                refresh_m_from_internal()
                assert expected_val == lhs.val32, idx

        def try_fix_c5(lhs, rhs):
            """utilize the precise modification given in the paper"""
            assert isinstance(lhs, VarBitRef)
            i = lhs.shift + 1
            # can not process c5,29: d2,20 = a2,20 would be violated
            if not (lhs.spec.startswith('c5') and i in [26, 27, 32]):
                return
            d2 = VarBitRef('d2,{}'.format(i - 9))
            assert d2.bitus == 0
            d2.val32 ^= d2.shifted(1)
            refresh_m_from_internal()
            assert lhs.bitsf == rhs.bitsf, (lhs, rhs, lhs.bitsf)

        def fix_b5():
            """b5 = f2(b4, c5, d5, a5, m[12], 13), whose bits can be flipped by
            modifying m[12]; a4 = f1(a3, b3, c3, d3, m[12], 3), to retain a4
            while modifying m[12], we have to manipulate f(b3, c3, d3). The
            involved bits are m12,{16,17,19} which luckily do not have other
            constraints
            """
            for lhs, rhs in iter_txt_eqn_group(ROUND2_REQUIREMENTS[3]):
                if lhs.bitsf == rhs.bitsf:
                    continue
                i = lhs.shift + 1 - 13
                bv, cv, dv = [VarBitRef('{}3,{}'.format(j, i)) for j in 'bcd']
                assert i in [16, 17, 19]
                if i == 17:
                    # c3,17 and d3,17 are required
                    assert cv.bitsf == 1 and dv.bitsf == 0
                    t = bv
                else:
                    if bv.bitsf:
                        t = cv
                    else:
                        t = dv
                t.val32 ^= t.shifted(1)
                refresh_m_from_internal()
                assert lhs.bitsf == rhs.bitsf, (lhs, lhs.bitsf, rhs)

        def fix_a6():
            """modify m[1] by manipulating d1"""
            for lhs, rhs in iter_txt_eqn_group(ROUND2_REQUIREMENTS[4]):
                if lhs.bitsf == rhs.bitsf:
                    continue
                i = (lhs.shift - 3 + 7) % 32 + 1
                assert i in [1, 4]
                dv = VarBitRef('d1,{}'.format(i))
                dv.val32 ^= dv.shifted(1)
                refresh_m_from_internal()
                #assert lhs.bitsf == rhs.bitsf, (i, lhs, lhs.bitsf, rhs)

        def fix_d6():
            for lhs, rhs in iter_txt_eqn_group(ROUND2_REQUIREMENTS[5]):
                if lhs.bitsf == rhs.bitsf:
                    continue
                i = lhs.shift + 1 - 5
                assert i == 24
                av = VarBitRef('a2,24')
                bv = VarBitRef('b1,24')
                cv = VarBitRef('c1,24')
                if av.bitsf:
                    t = bv
                else:
                    t = cv
                t.val32 ^= t.shifted(1)
                refresh_m_from_internal()
                #assert lhs.bitsf == rhs.bitsf, (lhs, lhs.bitsf, rhs)

        def test(check_unsat):
            compute_internal_states()
            fix_round1()
            fix_round2_iter1()
            fix_b5()
            fix_a6()
            fix_d6()
            refresh_m_from_internal()

            if check_unsat:
                unsat = []
                tot_req = 0
                for i in itertools.chain(ROUND1_REQUIREMENTS,
                                         ROUND2_REQUIREMENTS,
                                         ROUND3_REQUIREMENTS):
                    for lhs, rhs in iter_txt_eqn_group(i):
                        tot_req += 1
                        if lhs.bitsf != rhs.bitsf:
                            unsat.append((lhs, rhs))
                print('unsatisfied: {}/{}'.format(len(unsat), tot_req))

            m0 = m.copy()
            fs0 = get_final_state()

            p2 = lambda x: np.uint32(1 << x)
            m[1] += p2(31)
            m[2] += p2(31) - p2(28)
            m[12] -= p2(16)
            compute_internal_states()

            fs1 = get_final_state()
            if np.all(fs0 == fs1):
                return m0, m

            if check_unsat and not unsat:
                m1 = m.copy()
                def get_stat(mv):
                    m[:] = mv
                    return compute_internal_states(True)
                s0 = get_stat(m0)
                s1 = get_stat(m1)
                assert len(s0) == 41
                for i in range(41):
                    print(i + 1, hex(s1[i] - s0[i]))
                mb = lambda x, y: (m[x] >> np.uint32(y)) & 1
                print(mb(1, 31), mb(2, 31), mb(2, 28), mb(12, 16))


        m = m.copy()
        rng = np.random.RandomState(42)
        for cnt in itertools.count():
            if False:
                test(True)
            else:
                result = test(False)
                if result is not None:
                    return result, cnt + 1
            m[rng.randint(16)] ^= rng.randint(2**32)

    def run_test(inp_bytes):
        assert len(inp_bytes) == 64
        inp_arr = np.fromstring(inp_bytes, dtype=np.uint32)
        assert_eq(md4_collide(inp_arr, compute_only=True),
                  md4(inp_bytes, pad=False))
        (m0, m1), cnt = md4_collide(inp_arr)
        m0b = m0.tobytes()
        m1b = m1.tobytes()
        t0 = md4(m0b)
        t1 = md4(m1b)
        assert t0 == t1
        return m0b, as_bytes(as_np_bytearr(m0b) - as_np_bytearr(m1b)), cnt

    # rng = np.random.RandomState(15)
    # return run_test(rng.bytes(64))
    return run_test(
        b'Hello, this is a message to be collided; I hope my program '
        b'works')

@challenge
def ch55():
    warnings_filters = np.warnings.filters[:]
    np.warnings.simplefilter("ignore", RuntimeWarning)
    try:
        return ch55_impl()
    finally:
        np.warnings.filters[:] = warnings_filters
