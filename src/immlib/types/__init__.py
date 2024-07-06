# -*- coding: utf-8 -*-
################################################################################
# pimms/types/__init__.py

"""The pimms subpackage containing various utility types.

The utility types included in pimms are:
 * `MetaObject` is a `planobject` type that implements metadata via the
   value `metadata` (a lazy dictionary) and the `withmeta` and `dropmeta`
   methods.
 * `PropTable` is a lazy table similar to a pandas `DataFrame` but built around
   the abilities to lazily load properties (columns) and to annotating multiple
   values for each property along arbitrary dimensions such as time or depth.
"""

from ._core import (
    MetaObject,
    larray,
    ArrayIndex,
    LazyFrame)

__all__ = (
    "MetaObject",
    "larray",
    "ArrayIndex",
    "LazyFrame")

# Mark all the imported functions as belonging to this module instead of the
# hidden submodules:
from sys import modules
thismod = modules[__name__]
for k in dir():
    if k[0] == '_':
        continue
    obj = getattr(thismod, k)
    if getattr(obj, '__module__', __name__) == __name__:
        continue
    obj.__module__ = __name__
del obj, thismod, modules, k

    
