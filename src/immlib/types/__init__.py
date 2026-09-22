# -*- coding: utf-8 -*-
###############################################################################
# immlib/types/__init__.py

"""The immlib subpackage containing various utility types.

The utility types included in immlib are:
 * `MetaObject` is a `planobject` type that implements metadata via the
   value `metadata` (a lazy dictionary) and the `withmeta` and `dropmeta`
   methods.
 * `ArrayIndex` is a `planobject` type that indexes the elements of an array
   for easy searching.
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
