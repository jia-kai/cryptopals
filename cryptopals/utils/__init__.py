# -*- coding: utf-8 -*-

import pkgutil
import importlib
import os
import functools

import numpy as np

import pyximport
px = pyximport.install(setup_args={'include_dirs': os.path.dirname(__file__)})
from ._popcount import popcount8
pyximport.uninstall(*px)
del pyximport, px

SRC_ROOT = os.path.join(os.path.dirname(__file__), os.path.pardir)

_all_challenges = []

_cur_challange = None
"""current challenge that is being executed"""

def challenge(func):
    """decorator to mark a function as a challange entrypoint"""
    @functools.wraps(func)
    def new_func():
        global _cur_challange
        assert _cur_challange is None
        _cur_challange = func
        try:
            return func()
        finally:
            assert _cur_challange is func
            _cur_challange = None
    _all_challenges.append(new_func)
    return new_func

def discover_challenges():
    for loader, module_name, is_pkg in pkgutil.walk_packages(
            [SRC_ROOT], __name__[:__name__.rfind('.')+1]):
        if not is_pkg:
            importlib.import_module(module_name)
    _all_challenges.sort(key=lambda x: x.__name__)
    return _all_challenges

def assert_eq(a, b):
    assert a == b, 'assert_eq failed: a={!r} b={!r}'.format(a, b)

def open_resource(ext='.txt', mode='r'):
    """open resource file associated with current challenge"""
    assert _cur_challange is not None, (
        'open_resource can only be called from a challenge')
    chnum = int(_cur_challange.__name__[2:])
    return open(os.path.join(SRC_ROOT, os.path.pardir,
                             'res', str(chnum) + ext),
                mode)

def summarize_str(s):
    """summarize a string for display"""
    if isinstance(s, bytes):
        s = s.decode('utf-8')
    assert isinstance(s, str)
    return '{}...{}'.format(s[:10], s[-10:])

def as_bytes(val):
    """convert some value to bytes"""
    from ..bytearr import Bytearr
    if isinstance(val, str):
        val = val.encode('utf-8')
    elif isinstance(val, np.ndarray):
        assert val.dtype == np.uint8
        val = val.tostring()
    elif isinstance(val, Bytearr):
        val = val.to_bytes()
    elif isinstance(val, (list, tuple)):
        if not val:
            return bytes()
        if isinstance(val[0], str):
            val = ''.join(val).encode('utf-8')
        else:
            assert isinstance(val[0], int)
            val = bytes(val)
    assert isinstance(val, bytes)
    return val

def as_np_bytearr(val):
    from ..bytearr import Bytearr
    return Bytearr(val, allow_borrow=True).np_data

class CipherError(RuntimeError):
    """exception class for cipher algorithms"""
