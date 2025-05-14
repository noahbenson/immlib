# -*- coding: utf-8 -*-
################################################################################
# immlib/workflow/__init__.py

"""Tools for organizing simple directed acyclic graph workflows."""


from ._core import (
    to_pathcache,
    to_lrucache,
    calc,
    is_calc,
    is_calcfn,
    to_calc,
    plan,
    is_plan,
    plandict,
    is_plandict,
    is_tplandict)

from ._plantype import (
    plantype,
    planobject,
    is_plantype,
    is_planobject)

__all__ = (
    #"to_pathcache",
    #"to_lrucache",
    "calc",
    # We don't export is_calc because its presence in the library outside of
    # this subpackage is likely to lead to people using it when the function
    # really want is is_calcfn.
    "is_calcfn",
    "plan",
    "is_plan",
    "plandict",
    "is_plandict",
    "is_tplandict",
    "plantype",
    "planobject",
    "is_plantype",
    "is_planobject")

# Mark all the imported functions as belonging to this module instead of the
# hidden submodules:
from .._init import reclaim
reclaim(__name__, del_reclaim=True)
