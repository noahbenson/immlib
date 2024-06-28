# -*- coding: utf-8 -*-
################################################################################
# immlib/util/__init__.py

"""Utilites managed by immlib.

The `immlib.util` module contains numerous utility functions that are intended
to be useful when writing APIs for scientific libraries. These include functions
that test for particular types or common object features, functions for coercing
types into other types, and functions for querying numpy arrays and pytorch
tensors.
"""

from ._core import (
    is_str,
    strnorm,
    strcmp,
    streq,
    strstarts,
    strends,
    strissym,
    striskey,
    strisvar,
    is_acallable,
    is_lambda,
    is_asized,
    is_acontainer,
    is_aiterable,
    is_aiterator,
    is_areversible,
    is_acoll,
    is_aseq,
    is_amseq,
    is_apseq,
    is_abytes,
    is_bytes,
    is_aset,
    is_amset,
    is_apset,
    is_amap,
    is_ammap,
    is_apmap,
    is_ahashable,
    is_tuple,
    is_list,
    is_plist,
    is_llist,
    is_set,
    is_frozenset,
    is_pset,
    is_dict,
    is_odict,
    is_ddict,
    is_pdict,
    is_ldict,
    get,
    nestget,
    hashsafe,
    can_hash,
    itersafe,
    can_iter,
    is_pcoll,
    is_tcoll,
    is_mcoll,
    to_pcoll,
    to_tcoll,
    to_mcoll,
    to_frozenarray,
    frozenarray,
    lazykeymap,
    lazyvalmap,
    lazyitemmap,
    keymap,
    valmap,
    itemmap,
    dictmap,
    pdictmap,
    ldictmap,
    merge,
    rmerge,
    assoc,
    dissoc,
    lambdadict,
    argfilter,
    unitregistry)

from ._numeric import (
    is_numberdata,
    is_booldata,
    is_intdata,
    is_realdata,
    is_complexdata,
    is_numpydtype,
    like_numpydtype,
    to_numpydtype,
    is_array,
    to_array,
    is_torchdtype,
    like_torchdtype,
    to_torchdtype,
    is_tensor,
    to_tensor,
    is_numeric,
    to_numeric,
    is_sparse,
    to_sparse,
    is_dense,
    to_dense,
    is_number,
    is_bool,
    is_integer,
    is_real,
    is_complex,
    is_number,
    like_number,
    to_number)

from ._quantity import (
    is_ureg,
    is_unit,
    like_unit,
    is_quant,
    default_ureg,
    alike_units,
    unit,
    quant,
    mag,
    promote)

from ._url import (
    is_url,
    can_download_url,
    url_download)

__all__ = (
    #"is_numpydtype",
    #"like_numpydtype",
    #"to_numpydtype",
    "is_array",
    "to_array",
    #"is_torchdtype",
    #"like_torchdtype",
    #"to_torchdtype",
    "is_tensor",
    "to_tensor",
    "is_numeric",
    "to_numeric",
    "is_sparse",
    "to_sparse",
    "is_dense",
    "to_dense",
    "is_number",
    "like_number",
    "to_number",
    "is_str",
    "strnorm",
    "strcmp",
    "streq",
    "strstarts",
    "strends",
    "strissym",
    "striskey",
    "strisvar",
    "is_ureg",
    "is_unit",
    "is_quant",
    "is_lambda",
    "is_asized",
    "is_acontainer",
    "is_aiterable",
    "is_aiterator",
    "is_areversible",
    "is_acoll",
    "is_aseq",
    "is_amseq",
    "is_apseq",
    "is_abytes",
    "is_bytes",
    "is_aset",
    "is_amset",
    "is_apset",
    "is_pset",
    "is_amap",
    "is_ammap",
    "is_apmap",
    "is_ahashable",
    "is_tuple",
    "is_list",
    "is_plist",
    "is_llist",
    "is_set",
    "is_frozenset",
    "is_dict",
    "is_odict",
    "is_ddict",
    "is_pdict",
    "is_ldict",
    "get",
    "nestget",
    "hashsafe",
    "is_ahashable",
    "can_hash",
    "itersafe",
    "can_iter",
    "to_frozenarray",
    "frozenarray",
    "is_number",
    "is_bool",
    "is_integer",
    "is_real",
    "is_complex",
    "is_numberdata",
    "is_booldata",
    "is_intdata",
    "is_realdata",
    "is_complexdata",
    "default_ureg",
    "like_unit",
    "alike_units",
    "unit",
    "quant",
    "mag",
    "promote",
    "lazykeymap",
    "lazyvalmap",
    "lazyitemmap",
    "keymap",
    "valmap",
    "itemmap",
    "dictmap",
    "pdictmap",
    "ldictmap",
    "merge",
    "rmerge",
    "assoc",
    "dissoc",
    "lambdadict",
    "is_url",
    "url_download")

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
