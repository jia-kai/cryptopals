# -*- coding: utf-8 -*-

import pkgutil
import importlib
import os
import sys
import io
import functools
import contextlib

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

def assert_eq(a, b, msg=None):
    check = lambda a, b: a == b
    if isinstance(a, np.ndarray):
        assert (isinstance(b, np.ndarray) and
                a.dtype == b.dtype and a.shape == b.shape), (a, b)

        check = lambda a, b: np.alltrue(a == b)

    if msg is not None:
        msg = '; {}'.format(msg)
    else:
        msg = ''
    assert check(a, b), 'assert_eq failed: a={!r} b={!r}{}'.format(a, b, msg)

def open_resource(ext='.txt', mode='r'):
    """open resource file associated with current challenge"""
    assert _cur_challange is not None, (
        'open_resource can only be called from a challenge')
    chnum = int(_cur_challange.__name__[2:])
    return open(os.path.join(SRC_ROOT, os.path.pardir,
                             'res', str(chnum) + ext),
                mode)

def open_output(ext='.out', mode='w'):
    """open resource file associated with current challenge"""
    assert _cur_challange is not None, (
        'open_output can only be called from a challenge')
    chnum = int(_cur_challange.__name__[2:])
    odir = os.path.join(SRC_ROOT, os.path.pardir, 'output')
    if not os.path.isdir(odir):
        os.makedirs(odir)
    return open(os.path.join(odir, str(chnum) + ext), mode)

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
        assert val.dtype == np.uint8, val.dtype
        val = val.tostring()
    elif isinstance(val, Bytearr):
        val = val.to_bytes()
    elif isinstance(val, (list, tuple)):
        if not val:
            return bytes()
        if isinstance(val[0], str):
            val = ''.join(val).encode('utf-8')
        else:
            if isinstance(val[0], np.uint8):
                val = list(map(int, val))
            assert isinstance(val[0], int)
            val = bytes(val)
    assert isinstance(val, bytes)
    return val

def as_np_bytearr(val):
    from ..bytearr import Bytearr
    return Bytearr(val, allow_borrow=True).np_data

@contextlib.contextmanager
def capture_stdout():
    """capture stdout of statements in this context"""
    old = sys.stdout
    new = io.StringIO()
    sys.stdout = new
    try:
        yield new
    finally:
        sys.stdout = old

class CipherError(RuntimeError):
    """exception class for cipher algorithms"""

def rangech(begin, end, incl_end=True):
    """range for chars"""
    for i in range(ord(begin), ord(end) + int(bool(incl_end))):
        yield chr(i)
