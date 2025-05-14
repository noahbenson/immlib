# -*- coding: utf-8 -*-
###############################################################################
# immlib/__init__.py

'''``immlib`` is a library of tools for manipulating immutable scientific data.

The ``immlib`` library is designed to enable immutable data structures and lazy
computation in a scientific context, and it works primarily via a collection of
utility functions and through the use of decorators, which are generally
applied to classes and their members to declare how an immutable
data-structure's members are related.  Taken together, these utilities form a
DSL-like system for declaring workflows and immutable data-structures with full
inheritance support.

Attributes
----------
units : pint.UnitRegistry
    The registry for units tracked by ``immlib``. The ``immlib.units`` object
    is a global ``pint``-module unit registry that can be used as a single
    global place for tracking units. Immlib functions that interact with units
    generally take an argument ``ureg`` that can be used to modify this
    registry.  Additionally, the default registry (this object,
    ``immlib.units``) can be temporarily changed in a local block using ``with
    immlib.default_ureg(ureg): ...``.
version : immlib.Version
    A representation of the ``immlib`` version. The version string may be
    obtained via ``immlib.version.string``; major, minor, and micro numbers
    (when present) may be obtained via ``immlib.version.major``,
    ``immlib.version.minor``, and ``immlib.version.micro`` (when not provided
    the are set to ``None``), and a stage tag (a string), if given, can be
    obtained via ``immlib.version.stage``.
submodules : tuple of str
    A tuple of strings, each of which is the name of one of the submodules in
    ``immlib``. The modules are listed in load-order and all ``immlib``
    submodules, including private submodules, are included.
docproc: docrep.DocstringProcessor object
    This object is used to process all of the doc-strings in the ``immlib``
    library; it should be used only with the ``immlib.docwrap`` decorator,
    which can safely be applied anywhere in a sequence of decorators and which
    correctly applies the ``wraps`` decorator to its argument. Function
    documentation is always processed using the ``sections=('Parameters',
    'Returns', 'Raises', 'Examples', 'Inputs', 'Outputs')`` parameter and the
    ``with_indent(4)`` decorator. The base-name for the function ``f`` is
    ``f.__module__ + '.' + f.__name__``.
'''


# Imports #####################################################################

# We always load _init first.
from ._init    import reclaim
# Then the core library.
from .doc      import *
from .util     import *
from .pathlib  import *
from .iolib    import *
from .workflow import *
from .types    import *
# Import the Global UnitRegistry object to the global immlib scope. This is the
# value that gets updated when one runs `immlib.default_ureg()`, and this is
# the UnitRegistry that is used as the default registry for all ``immlib``
# functions.
from .util._quantity import _initial_global_ureg as units
# Do the same for the global DocstringProcessor (from the docrep library) from
# the doc subpackage.
from .doc._core import _initial_global_docproc as docproc
# We want the version object from the ._version namespace; this is always last.
from ._version import (version, Version)


# Modules/Reloading ###########################################################

submodules = (
    'immlib._init',
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
    'immlib.types._core',
    'immlib.types',
    'immlib._version')
def reload_immlib():
    """Reload and return the entire ``immlib`` package.

    ``immlib.reload_immlib()`` reloads every submodule in the ``immlib``
    package then reloads ``immlib`` itself, and returns the reloaded package.

    .. Warning:: This function exists primarily for debugging purposes; its use
        is not generally needed or advised by users of the library.

    Returns
    -------
    module
        The newly reloaded ``immlib`` module.

    Examples
    --------
    >>> import immlib as il
    >>> il.units = None  # This will break parts of the library.
    >>> il = il.reload_immlib()  # But this resets it.
    >>> il.units is not None
    True
    """
    import sys, importlib
    for mod in submodules:
        importlib.reload(sys.modules[mod])
    return importlib.reload(sys.modules[__name__])


# Package Meta-Code ###########################################################

__version__ = version.string
__all__ = tuple(
    [k for k in locals()
     if k[0] != '_'
     if k != 'submodules'
     if k != 'version'
     if ('immlib.' + k) not in submodules])
# We want to mark our functions as being from the immlib module.
reclaim(__name__, __all__, del_reclaim=True)
