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

    def __init__(self, data):
        if isinstance(data, str):
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

    def __xor__(self, rhs):
        """xor with another Bytearr or a single byte or an iterator of byte

        :type rhs: int, :class:`Bytearr` or iterator
        :return: iterator of byte
        """

        if not isinstance(rhs, collections.Iterable):
            assert isinstance(rhs, (int, np.uint8))
            rhs = itertools.repeat(np.uint8(rhs))

        return map(lambda x: x[0] ^ x[1], zip(self._data, rhs))

    def __iter__(self):
        return iter(self._data)

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

    def to_str(self):
        """interpret as utf-8 encoded str"""
        return self._data.tobytes().decode('utf-8')
