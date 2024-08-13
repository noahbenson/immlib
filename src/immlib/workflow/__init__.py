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
from .._init import reclaim
reclaim(__name__, __all__)
