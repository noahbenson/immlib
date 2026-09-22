# -*- coding: utf-8 -*-
################################################################################
# immlib/workflow/__init__.py

"""Tools for organizing simple directed acyclic graph workflows."""


from ._core import (
    CALC_DOC_SECTIONS,
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
    PlanError,
    save_ready,
    is_tplandict)

from ._plantype import (
    plantype,
    planobject,
    is_plantype,
    is_planobject)

__all__ = (
    "CALC_DOC_SECTIONS",
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
    "PlanError",
    "save_ready",
    "is_tplandict",
    "plantype",
    "planobject",
    "is_plantype",
    "is_planobject")
