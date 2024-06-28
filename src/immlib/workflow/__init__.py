# -*- coding: utf-8 -*-
################################################################################
# immlib/workflow/__init__.py

"""Tools for organizing simple directed acyclic graph workflows."""


from ._core import (
    to_pathcache,
    to_lrucache,
    calc,
    is_calc,
    plan,
    is_plan,
    plandict,
    is_plandict)

from ._plantype import (
    plantype,
    planobject,
    is_plantype,
    is_planobject)

__all__ = (
    #"to_pathcache",
    #"to_lrucache",
    "calc",
    "is_calc",
    "plan",
    "is_plan",
    "plandict",
    "is_plandict",
    "plantype",
    "planobject",
    "is_plantype",
    "is_planobject")

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
