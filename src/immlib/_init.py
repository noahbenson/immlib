# -*- coding: utf-8 -*-
###############################################################################
# immlib/_init.py

"""Utilities and global definitions needed by theoretically the entire rest of
the `immlib` library during the import process.

The `immlib._init` module is the first submodule of the `immlib` library that
is loaded during the import process and is the firt that is reloaded when the
`immlib.reload_immlib()` library is called. Accordingly, functions and
variables that are needed by other modules can be put here. These functions and
variables, if they are intended to be public, should be reclaimed by another
module (see the `immlib._init.reclaim` / `immlib.util.reclaim` function).
"""


# Dependencies ################################################################

import sys
from types import ModuleType


# Utility Functions ###########################################################

def reclaim(modname, attributes=Ellipsis, *,
            del_reclaim=False,
            skip_externs=True,
            skip_private=True,
            skip_modules=True):
    """Sets the ``__module__`` attributes of objects in the module whose name
    is `modname` to be `modname`.

    ``reclaim(__name__, __all__)``, when run at the end of an ``__init__.py``
    file, will set the module names of all objects in that module's ``__all__``
    list to be ``__name__``, effectively hiding any submodule ownership of
    those objects.

    ``reclaim(__name__)`` attempts to reclaim all attributes in the module, not
    just those in the ``__all__`` list.

    Parameters
    ----------
    modname : str
        The name of the module to reclaim. Typically, ``reclaim`` is called
        from an ``__init__.py`` file using the syntax ``reclaim(__name__)`` or
        ``reclaim(__name__, __all__)``.
    attributes : Ellipsis | iterable of str, optional
        The attributes to reclaim. If this is explicitly given then it must be
        an iterable of attribute names (strings). In this case, no additional
        checks implied by the subsequent ``skip_`` and ``only_`` parameters are
        performed. If `attributes` is not provided or is given the value of
        ``Ellipsis`` (the default), then all attributes of the module are
        considered, and the filters specified by the subsequent options are
        applied.
    del_reclaim : bool, optional
        If the optional argument `del_reclaim` is set to ``True``, then the
        ``reclaim`` attribute is deleted from the module as well. This can be
        used to automatically sanitize a module.
    skip_externs : bool, optional
        The `skip_externs` option (default: ``True``) can be set to ``False``
        to instruct the ``reclaim`` function to include attributes that are not
        from a submodule of the module named by `modname`.

        If this option is set to ``True``, then attributes of the module on
        which ``reclaim`` is run are only reclaimed if they belong to a
        submodule of of that module.

        .. Warning:: If `skip_externs` is set to ``False``, attributes like
            ``np`` from the import statement ``import numpy as np`` at the top
            of an ``__init__.py`` file can result in the ``numpy`` module being
            reclaimed (if ``skip_modules`` is also ``False``). Similarly, if
            the line ``from immlib.util import reclaim`` appears in an
            ``__init__.py`` file followed by ``reclaim(__name__,
            skip_externs=False)`` then the ``reclaim`` function will be
            reclaimed by the module.
    skip_private : boolean, optional
        The `skip_private` option may be set to ``False`` to instruct the
        function to claim ownership over attributes whose names begin with the
        underscore (``_``) charcter. If this is not provided or is ``True``,
        then such attributes are skipped.
    skip_modules : boolean, optional
        The `skip_modules` option (default: ``True``) may be set to false to
        instruct the ``reclaim`` function to attempt to reclaim modules as well
        as functions and attributes.

    Returns
    -------
    str
        The module named by the given `modname` argument.
    """
    mod = sys.modules[modname]
    for k in dir(mod):
        if skip_private and k[0] == '_':
            continue
        obj = getattr(mod, k)
        if skip_modules and isinstance(obj, ModuleType):
            continue
        if del_reclaim and obj is reclaim:
            continue
        objmodname = getattr(obj, '__module__', modname)
        if objmodname == modname:
            continue
        if skip_externs and not objmodname.startswith(modname):
            continue
        obj.__module__ = modname
    if del_reclaim and hasattr(mod, 'reclaim'):
        delattr(mod, 'reclaim')
    return mod
