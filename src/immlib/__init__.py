# -*- coding: utf-8 -*-
################################################################################
# immlib/__init__.py

'''`immlib` is a library of tools for manipulating immutable scientific data.

The `immlib` library is designed to enable immutable data structures and lazy
computation in a scientific context, and it works primarily via a collection of
utility functions and through the use of decorators, which are generally applied
to classes and their members to declare how an immutable data-structure's
members are related.  Taken together, these utilities form a DSL-like system for
declaring workflows and immutable data-structures with full inheritance support.
'''


# Imports ######################################################################

from .doc      import *
from .util     import *
from .pathlib  import *
from .iolib    import *
from .workflow import *
# We want the version object from the ._version namespace.
from ._version import version
# Import the Global UnitRegistry object to the global immlib scope. This is the
# value that gets updated when one runs `immlib.default_ureg()`, and this is the
# UnitRegistry that is used as the default registry for all `immlib` functions.
from .util._quantity import _initial_global_ureg as units
"""UnitRegistry: the registry for units tracked by immlib.

`immlib.units` is a global `pint`-module unit registry that can be used as a
single global place for tracking units. Immlib functions that interact with
units generally take an argument `ureg` that can be used to modify this
registry.  Additionally, the default registry (this object, `immlib.units`) can
be temporarily changed in a local block using `with immlib.default_ureg(ureg):
...`.
"""


# Modules/Reloading ############################################################

submodules = (
    'immlib.doc._core',
    'immlib.doc',
    'immlib.util._core',
    'immlib.util._numeric',
    'immlib.util._quantity',
    'immlib.util',
    'immlib.pathlib._osf',
    'immlib.pathlib._cache',
    'immlib.pathlib._core',
    'immlib.pathlib',
    'immlib.iolib._core',
    'immlib.iolib',
    'immlib.workflow._core',
    'immlib.workflow._plantype',
    'immlib.workflow',
    'immlib._version')
"""tuple: a list of all immlib subpackage names in load-order.

`immlib.submodules` is a tuple of strings, each of which is the name of one of
the sub-submodules in `immlib`. The modules are listed in load-order and all
`immlib` submodules are included.
"""
def reload_immlib():
    """Reload and return the entire `immlib` package.

    `immlib.reload_immlib()` reloads every submodule in the `immlib` package
    then reloads `immlib` itself, and returns the reloaded package.

    This function exists primarily for debugging purposes; its use is not
    generally needed or advised.
    """
    import sys, importlib
    for mod in submodules:
        importlib.reload(sys.modules[mod])
    return importlib.reload(sys.modules['immlib'])


# Package Meta-Code ############################################################

__version__ = version.string
__all__ = tuple(
    [k for k in locals()
     if k[0] != '_'
     if k != 'reload_immlib'
     if k != 'submodules'
     if ('immlib.' + k) not in submodules])
