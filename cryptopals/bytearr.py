# -*- coding: utf-8 -*-

import numpy as np
import binascii
import base64
import itertools
import collections

class Bytearr:
    """bytearray with binary arithmetics

    actuallly wrapper of :class:`numpy.ndarray`
    """

    _data = None

    def __init__(self, data, *, allow_borrow=False):
        if isinstance(data, Bytearr):
            data = data._data

        if isinstance(data, np.ndarray):
            assert data.ndim == 1
            if not allow_borrow:
                self._data = np.array(data, dtype=np.uint8)
                return
        elif isinstance(data, str):
            data = [ord(i) for i in data]
        else:
            if not isinstance(data, (list, tuple)):
                assert isinstance(data, collections.Iterable)
                data = [np.uint8(i) for i in data]
        self._data = np.ascontiguousarray(data, dtype=np.uint8)

    def __eq__(self, rhs):
        if not isinstance(rhs, Bytearr):
            rhs = Bytearr(rhs)
        return np.all(self._data == rhs._data)

    def __getitem__(self, idx):
        return Bytearr(self._data.__getitem__(idx), allow_borrow=True)

    def __xor__(self, rhs):
        """xor with another Bytearr or a single byte or an iterator of byte

        :type rhs: int, :class:`Bytearr` or iterator
        :return: iterator of byte
        """

        if not isinstance(rhs, collections.Iterable):
            assert isinstance(rhs, (int, np.uint8))
            rhs = itertools.repeat(np.uint8(rhs))

        return map(lambda x: x[0] ^ x[1], zip(self._data, rhs))

    def __ixor__(self, rhs):
        self._data ^= Bytearr(rhs, allow_borrow=True).np_data

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    @property
    def np_data(self):
        """data as numpy array"""
        return self._data

    @classmethod
    def from_hex(cls, data):
        assert isinstance(data, str)
        data = data.strip()
        return cls(np.fromstring(binascii.unhexlify(data), dtype=np.uint8))

    @classmethod
    def from_base64(cls, data):
        return cls(np.fromstring(base64.b64decode(data), dtype=np.uint8))

    def to_base64(self):
        return base64.b64encode(self._data.tobytes()).decode('utf-8')

    def to_bytes(self):
        return self._data.tobytes()

    def to_str(self):
        """interpret as utf-8 encoded str"""
        return self._data.tobytes().decode('utf-8')
