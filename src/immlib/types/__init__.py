# -*- coding: utf-8 -*-
###############################################################################
# pimms/types/__init__.py

"""The pimms subpackage containing various utility types.

The utility types included in pimms are:
 * `MetaObject` is a `planobject` type that implements metadata via the
   value `metadata` (a lazy dictionary) and the `withmeta` and `dropmeta`
   methods.
 * `ArrayInde` is a `planobject` type that implements 
"""

from ._core import (
    MetaObject,
    ArrayIndex,
    ImmutableType,
    Immutable)

__all__ = (
    "MetaObject",
    "ArrayIndex",
    "ImmutableType",
    "Immutable")

# Mark all the imported functions as belonging to this module instead of the
# hidden submodules:
from .._init import reclaim
reclaim(__name__, del_reclaim=True)
