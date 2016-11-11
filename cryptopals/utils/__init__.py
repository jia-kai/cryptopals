# -*- coding: utf-8 -*-

import pkgutil
import importlib
import os
import functools

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
