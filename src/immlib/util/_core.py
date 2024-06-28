# -*- coding: utf-8 -*-
################################################################################
# immlib/util/_core.py


# Dependencies #################################################################

from inspect import (signature, getfullargspec)
from functools import (wraps, partial)

import pint
import numpy as np
import scipy.sparse as sps

from ..doc import docwrap


# Strings ######################################################################

@docwrap
def is_str(obj):
    """Returns `True` if an object is a string and `False` otherwise.

    `is_str(obj)` returns `True` if the given object `obj` is an instance
    of the `str` type and `False` otherwise.

    Parameters
    ----------
    obj : object
        The object whose quality as a string object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is a string, otherwise `False`.
    """
    return isinstance(obj, str)
from unicodedata import normalize as unicodedata_normalize
@docwrap
def strnorm(s, case=False, unicode=True):
    """Normalizes a string using the `unicodedata` package.

    `strnorm(s)` returns a version of `s` that has been unicode-normalized using
    the `unicodedata.normalize(s)` function. Case-normalization can also be
    requested via the `case` parameter.

    Parameters
    ----------
    s : object
        The string to be normalized.
    case : boolean, optional
        Whether to perform case-normalization (`case=True`) or not
        (`case=False`, the default). If two strings are case-normalized, then an
        equality comparison will reveal whether the original (unnormalized
        strings) were equal up to the case of the characters. Case normalization
        is performed using the `s.casefold()` method.
    unicode : boolean, optional
        Whether to perform unicode normalization via the `unicodedata.normalize`
        function. The default behavior (`unicode=True`) is to perform
        normalization, but this can be disabled with `unicode=False`.

    Returns
    -------
    str
       A normalized version of `s`.
    """
    if unicode is True:
        unicode = 'NFD'
    if unicode:
        s = unicodedata_normalize(unicode, s)
        if case:
            s = s.casefold()
            s = unicodedata_normalize(unicode, s)
    elif case:
        s = s.casefold()
    return s
def _strbinop_prep(a, b, case=True, unicode=None, strip=False):
    if not is_str(a) or not is_str(b): return None
    # We do case normalization when case comparison is *not* requested
    casenorm = not bool(case)
    # When unicode is None, we do its normalization only when case normalization
    # is being done.
    if unicode is None: unicode = casenorm
    # We now perform normalization if it is required.
    if unicode or casenorm:
        a = strnorm(a, case=casenorm, unicode=unicode)
        b = strnorm(b, case=casenorm, unicode=unicode)
    # If we requested stripping, do that now.
    if strip is True:
        a = a.strip()
        b = b.strip()
    elif strip is not False:
        a = a.strip(strip)
        b = b.strip(strip)
    return (a,b)
@docwrap
def strcmp(a, b, case=True, unicode=None, strip=False, split=False):
    """Determines if the given objects are equal strings and compares them.

    `strcmp(a, b)` returns `None` if either `a` or `b` is not a string;
    otherwise, it returns `-1`, `0`, or `1` if `a` is less than, equal to, or
    greater than `b`, respectively, subject to the constraints of the
    parameters.

    Parameters
    ----------
    a : object
        The first argument.
    b : object
        The second argument.
    case : boolean, optional
        Whether to perform case-sensitive (`case=True`, the default) or
        case-insensitive (`case=False`) string comparison.
    unicode : boolean or None, optional
        Whether to run unicode normalization on `a` and `b` prior to
        comparison. By default, this is `None`, which is interpreted as a `True`
        value when `case` is `False` and a `False` value when `case` is `True`
        (i.e., unicode normalization is performed when case-insensitive
        comparison is being performed but not when standard string comparison is
        being performed). Unicode normalization is always performed both before
        and after casefolding. Unicode normalization is performed using the
        `unicodedata` package's `normalize(unicode, string)` function. If this
        argument is a string, it is instead passed to the `normalize` function
        as the first argument.
    strip : boolean, optional
        If set to `True`, then `a.strip()` and `b.strip()` are used in place of
        `a` and `b`. If set to `False` (the default), then no stripping is
        performed. If a non-boolean value is given, then it is passed as an
        argument to the `strip()` method.
    split : boolean, optional
        If set to `True`, then `a.split()` and `b.split()` are used in place of
        `a` and `b`, where the lists that result from `a.split()` and
        `b.split()` are compared for their sorted ordering of strings. If set to
        `False` (the default), then no splitting is performed. If a non-boolean
        value is given, then it is passed as an argument to the `split()`
        method.

    Returns
    -------
    boolean or None
        `None` if either `a` is not a string or `b` is not a string; otherwise,
        `-1` if `a` is lexicographically less than `b`, `0` if `a == b`, and `1`
        if `a` is lexicographically greater than `b`, subject to the constraints
        of the optional parameters.
    """
    prep = _strbinop_prep(a, b, case=case, unicode=unicode, strip=strip)
    if prep is None:
        return None
    (a, b) = prep
    # If straightforward non-split comparison is requested, we can just return
    # the comparison
    if split is False:
        return (-1 if a < b else 1 if a > b else 0)
    # Otherwise, we split then do a list comparison.
    if split is True:
        a = a.split()
        b = b.split()
    else:
        a = a.split(split)
        b = b.split(split)
    # We do this efficiently, by first comparing elements in order, then
    # lenghts.
    for (aa,bb) in zip(a,b):
        if aa < bb:
            return -1
        elif aa > bb:
            return 1
    na = len(a)
    nb = len(b)
    if na < nb:
        return -1
    elif na > nb:
        return 1
    return 0
@docwrap
def streq(a, b, case=True, unicode=None, strip=False, split=False):
    """Determines if the given objects are equal strings or not.

    `streq(a, b)` returns `True` if `a` and `b` are both strings and are equal
    to each other, subject to the constraints of the parameters.

    Parameters
    ----------
    %(immlib.util._core.strcmp.parameters)s

    Returns
    -------
    boolean or None
        `True` if `a` and `b` are both strings and if `a` is equal to `b`,
        subject to the constraints of the optional parameters. If either `a` or
        `b` is not a string, then `None` is returned.
    """
    cmpval = strcmp(a, b, case=case, unicode=unicode, strip=strip, split=split)
    return None if cmpval is None else (cmpval == 0)
@docwrap
def strends(a, b, case=True, unicode=None, strip=False):
    """Determines if the string `a` ends with the string `b` or not.

    `strends(a, b)` returns `True` if `a` and `b` are both strings and if `a`
    ends with `b`, subject to the constraints of the parameters.

    Parameters
    ----------
    %(immlib.util._core.strcmp.parameters.case)s
    %(immlib.util._core.strcmp.parameters.unicode)s
    %(immlib.util._core.strcmp.parameters.strip)s

    Returns
    -------
    boolean or None
        `True` if `a` and `b` are both strings and if `a` ends with `b`,
        subject to the constraints of the optional parameters. If either `a` or
        `b` is not a string, then `None` is returned.
    """
    prep = _strbinop_prep(a, b, case=case, unicode=unicode, strip=strip)
    if prep is None: return None
    else: (a, b) = prep
    # Check the ending
    return a.endswith(b)
@docwrap
def strstarts(a, b, case=True, unicode=None, strip=False):
    """Determines if the string `a` starts with the string `b` or not.

    `strstarts(a, b)` returns `True` if `a` and `b` are both strings
    and if `a` starts with `b`, subject to the constraints of the
    parameters.

    Parameters
    ----------
    %(immlib.util._core.strcmp.parameters.case)s
    %(immlib.util._core.strcmp.parameters.unicode)s
    %(immlib.util._core.strcmp.parameters.strip)s

    Returns
    -------
    boolean or None
        `True` if `a` and `b` are both strings and if `a` starts with `b`,
        subject to the constraints of the optional parameters. If either `a` or
        `b` is not a string, then `None` is returned.
    """
    prep = _strbinop_prep(a, b, case=case, unicode=unicode, strip=strip)
    if prep is None: return None
    else: (a, b) = prep
    # Check the beginning.
    return a.startswith(b)
@docwrap
def strissym(s):
    """Determines if the given string is a valid symbol (identifier).

    `strissym(s)` returns `True` if `s` is both a string and a valid identifier.
    Otherwise, it returns `False` if `s` is a string and `None` if not.

    See also: `striskey`, `strisvar`
    """
    return s.isidentifier() if is_str(s) else None
from keyword import iskeyword
@docwrap
def striskey(s):
    """Determines if the given string is a valid keyword.

    `strissym(s)` returns `True` if `s` is both a string and a valid keyword
    (such as `'if'` or `'while'`). Otherwise, it returns `False` if `s` is a
    string and `None` if not.

    See also: `strissym`, `strisvar`
    """
    return iskeyword(s) if is_str(s) else None
@docwrap
def strisvar(s):
    """Determines if the given string is a valid variable name.

    `strissym(s)` returns `True` if `s` is both a string and a valid name (i.e.,
    a symbol but not a keyword). Otherwise, it returns `False` if `s` is a
    string and `None` if not.

    See also: `strissym`, `striskey`
    """
    return (
        None if not is_str(s) else
        False if iskeyword(s) else
        s.isidentifier())


# Builtin Python Abstract Types ################################################

from collections.abc import Callable
@docwrap
def is_acallable(obj):
    """Returns `True` if an object is a callable object like a function.

    `is_acallable(obj)` returns `True` if the given object `obj` is an instance
    of the abstract callable type, `collections.abc.Callable`. Note that in
    general, it is more common and preferable to use the builtin `callable`
    method; however, `is_callable` is included in `immlib` for completeness.

    Parameters
    ----------
    obj : object
        The object whose quality as an `Callable` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `Callable`, otherwise `False`.
    """
    return isinstance(obj, Callable)
from types import LambdaType
@docwrap
def is_lambda(obj):
    """Returns `True` if an object is a lambda function, otherwise `False`.

    `is_lambda(obj)` returns `True` if the given object `obj` is an instance
    of the `types.LambdaType` type.

    Parameters
    ----------
    obj : object
        The object whose quality as a `LambdaType` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `LambdaType`, otherwise `False`.
    """
    return isinstance(obj, LambdaType)
from collections.abc import Sized
@docwrap
def is_asized(obj):
    """Returns `True` if an object implements `len()`, otherwise `False`.

    `is_asized(obj)` returns `True` if the given object `obj` is an instance of
    the abstract `collections.abc.Sized` type.

    Parameters
    ----------
    obj : object
        The object whose quality as a `Sized` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `Sized`, otherwise `False`.
    """
    return isinstance(obj, Sized)
from collections.abc import Container
@docwrap
def is_acontainer(obj):
    """Returns `True` if an object implements `__contains__`, otherwise `False`.

    `is_acontainer(obj)` returns `True` if the given object `obj` is an instance
    of the abstract `collections.abc.Container` type.

    Parameters
    ----------
    obj : object
        The object whose quality as a `Container` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `Container`, otherwise `False`.
    """
    return isinstance(obj, Container)
from collections.abc import Iterable
@docwrap
def is_aiterable(obj):
    """Returns `True` if an object implements `__iter__`, otherwise `False`.

    `is_aiterable(obj)` returns `True` if the given object `obj` is an instance
    of the abstract `collections.abc.Iterable` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `Iterable` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `Iterable`, otherwise `False`.
    """
    return isinstance(obj, Iterable)
from collections.abc import Iterator
@docwrap
def is_aiterator(obj):
    """Returns `True` if an object is an instance of `collections.abc.Iterator`.

    `is_aiterable(obj)` returns `True` if the given object `obj` is an instance
    of the abstract `collections.abc.Iterator` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `Iterator` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `Iterator`, otherwise `False`.
    """
    return isinstance(obj, Iterator)
from collections.abc import Reversible
@docwrap
def is_areversible(obj):
    """Returns `True` if an object is an instance of `Reversible`.

    `is_areversible(obj)` returns `True` if the given object `obj` is an
    instance of the abstract `collections.abc.Reversible` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `Reversible` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `Reversible`, otherwise `False`.
    """
    return isinstance(obj, Reversible)
from collections.abc import Collection
@docwrap
def is_acoll(obj):
    """Returns `True` if an object is a collection (a sized iterable container).

    `is_acoll(obj)` returns `True` if the given object `obj` is an instance
    of the abstract `collections.abc.Collection` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `Collection` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `Collection`, otherwise `False`.
    """
    return isinstance(obj, Collection)
from collections.abc import Sequence
@docwrap
def is_aseq(obj):
    """Returns `True` if an object is a sequence, otherwise `False`.

    `is_aseq(obj)` returns `True` if the given object `obj` is an instance of
    the abstract `collections.abc.Sequence` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `Sequence` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `Sequence`, otherwise `False`.
    """
    return isinstance(obj, Sequence)
from collections.abc import MutableSequence
@docwrap
def is_amseq(obj):
    """Returns `True` if an object is a mutable sequence, otherwise `False`.

    `is_amseq(obj)` returns `True` if the given object `obj` is an instance of
    the abstract `collections.abc.MutableSequence` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `MutableSequence` object is to be
        assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `MutableSequence`, otherwise `False`.
    """
    return isinstance(obj, MutableSequence)
from pcollections.abc import PersistentSequence
@docwrap
def is_apseq(obj):
    """Returns `True` if an object is a persistent sequence, otherwise `False`.

    `is_apseq(obj)` returns `True` if the given object `obj` is an instance of
    the abstract `pcollections.abc.PersistentSequence` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `PersistentSequence` object is to be
        assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `PersistentSequence`, otherwise
        `False`.
    """
    return isinstance(obj, PersistentSequence)
from collections.abc import ByteString
@docwrap
def is_abytes(obj):
    """Returns `True` if an object is a byte-string, otherwise `False`.

    `is_abytes(obj)` returns `True` if the given object `obj` is an instance of
    the abstract `collections.abc.ByteString` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `ByteString` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `ByteString`, otherwise `False`.
    """
    return isinstance(obj, ByteString)
@docwrap
def is_bytes(obj):
    """Returns `True` if an object is a `bytes` object, otherwise `False`.

    `is_bytes(obj)` returns `True` if the given object `obj` is an instance of
    the `bytes` type and returns `False` otherwise.

    Parameters
    ----------
    obj : object
        The object whose quality as an `bytes` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `bytes`, otherwise `False`.
    """
    return isinstance(obj, bytes)
from collections.abc import Set
@docwrap
def is_aset(obj):
    """Returns `True` if an object is a set type, otherwise `False`.

    `is_aset(obj)` returns `True` if the given object `obj` is an instance of
    the abstract `collections.abc.Set` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `Set` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `Set`, otherwise `False`.
    """
    return isinstance(obj, Set)
from collections.abc import MutableSet
@docwrap
def is_amset(obj):
    """Returns `True` if an object is a mutable set, otherwise `False`.

    `is_amset(obj)` returns `True` if the given object `obj` is an instance of
    the abstract `collections.abc.MutableSet` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `MutableSet` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `MutableSet`, otherwise `False`.
    """
    return isinstance(obj, MutableSet)
from pcollections.abc import PersistentSet
@docwrap
def is_apset(obj):
    """Returns `True` if an object is a persistent set, otherwise `False`.

    `is_apset(obj)` returns `True` if the given object `obj` is an instance of
    the abstract `pcollections.abc.PersistentSet` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `PersistentSet` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `PersistentSet`, otherwise `False`.
    """
    return isinstance(obj, PersistentSet)
from collections.abc import Mapping
@docwrap
def is_amap(obj):
    """Returns `True` if an object is an abstract mapping, otherwise `False`.

    `is_amap(obj)` returns `True` if the given object `obj` is an instance of
    the abstract `collections.abc.Mapping` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `Mapping` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `Mapping`, otherwise `False`.
    """
    return isinstance(obj, Mapping)
from collections.abc import MutableMapping
@docwrap
def is_ammap(obj):
    """Returns `True` if an object is a mutable mapping, otherwise `False`.

    `is_ammap(obj)` returns `True` if the given object `obj` is an instance of
    the abstract `collections.abc.MutableMapping` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `MutableMapping` object is to be
        assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `MutableMapping`, otherwise `False`.
    """
    return isinstance(obj, MutableMapping)
from pcollections.abc import PersistentMapping
@docwrap
def is_apmap(obj):
    """Returns `True` if an object is a persistent mapping, otherwise `False`.

    `is_apmap(obj)` returns `True` if the given object `obj` is an instance of
    the abstract `pcollections.abc.PersistentMapping` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `PersistentMapping` object is to be
        assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `PersistentMapping`, otherwise
        `False`.
    """
    return isinstance(obj, PersistentMapping)
from collections.abc import Hashable
@docwrap
def is_ahashable(obj):
    """Returns `True` if an object is a hashable object, otherwise `False`.

    `is_ahashable(obj)` returns `True` if the given object `obj` is an instance
    of the abstract `collections.abc.Hashable` type. This differs from the
    `can_hash` function, which checks whehter calling `hash` on an object raises
    an exception.

    See also: `can_hash`

    Parameters
    ----------
    obj : object
        The object whose quality as an `Hashable` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `Hashable`, otherwise `False`.
    """
    return isinstance(obj, Hashable)


# Builtin Python Concrete Types ################################################

@docwrap
def is_list(obj):
    """Returns `True` if an object is a `list` object.

    `is_list(obj)` returns `True` if the given object `obj` is an instance of
    the `list` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `list` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `list`, otherwise `False`.
    """
    return isinstance(obj, list)
@docwrap
def is_tuple(obj):
    """Returns `True` if an object is a `tuple` object.

    `is_tuple(obj)` returns `True` if the given object `obj` is an instance of
    the `tuple` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `tuple` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `tuple`, otherwise `False`.
    """
    return isinstance(obj, tuple)
from pcollections import plist
@docwrap
def is_plist(obj):
    """Returns `True` if an object is a persistent list object.

    `is_plist(obj)` returns `True` if the given object `obj` is an instance of
    the `pcollections.plist` type and `False` otherwise.

    Parameters
    ----------
    obj : object
        The object whose quality as a `plist` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `plist`, otherwise `False`.
    """
    return isinstance(obj, plist)
from pcollections import tlist
@docwrap
def is_tlist(obj):
    """Returns `True` if an object is a transient list object.

    `is_plist(obj)` returns `True` if the given object `obj` is an instance of
    the `pcollections.tlist` type and `False` otherwise.

    Parameters
    ----------
    obj : object
        The object whose quality as a `tlist` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `tlist`, otherwise `False`.
    """
    return isinstance(obj, tlist)
from pcollections import llist
@docwrap
def is_llist(obj):
    """Returns `True` if an object is a persistent lazy list object.

    `is_llist(obj)` returns `True` if the given object `obj` is an instance of
    the `pcollections.llist` type and `False` otherwise.

    Parameters
    ----------
    obj : object
        The object whose quality as a `llist` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `llist`, otherwise `False`.
    """
    return isinstance(obj, llist)
@docwrap
def is_set(obj):
    """Returns `True` if an object is a `set` object.

    `is_set(obj)` returns `True` if the given object `obj` is an instance of
    the `set` type. Note that this is not the same as `is_aset` which determines
    whether the object is of the `collections.abc.Set` abstract type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `set` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `set`, otherwise `False`.
    """
    return isinstance(obj, set)
@docwrap
def is_frozenset(obj):
    """Returns `True` if an object is a `frozenset` object.

    `is_frozenset(obj)` returns `True` if the given object `obj` is an instance
    of the `frozenset` type.

    `is_fset(obj)` is an alias for `is_frozenset(obj)`.

    Parameters
    ----------
    obj : object
        The object whose quality as an `frozenset` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `frozenset`, otherwise `False`.
    """
    return isinstance(obj, frozenset)
from pcollections import pset
@docwrap
def is_pset(obj):
    """Returns `True` if an object is a persistent set object.

    `is_pset(obj)` returns `True` if the given object `obj` is an instance of
    the `pcollections.pset` type and `False` otherwise.

    Parameters
    ----------
    obj : object
        The object whose quality as a `pset` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `pset`, otherwise `False`.
    """
    return isinstance(obj, pset)
from pcollections import tset
@docwrap
def is_tset(obj):
    """Returns `True` if an object is a transient set object.

    `is_tset(obj)` returns `True` if the given object `obj` is an instance of
    the `pcollections.tset` type and `False` otherwise.

    Parameters
    ----------
    obj : object
        The object whose quality as a `tset` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `tset`, otherwise `False`.
    """
    return isinstance(obj, tset)
@docwrap
def is_dict(obj):
    """Returns `True` if an object is a `dict` object.

    `is_dict(obj)` returns `True` if the given object `obj` is an instance of
    the `dict` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `dict` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `dict`, otherwise `False`.
    """
    return isinstance(obj, dict)
from collections import OrderedDict
@docwrap
def is_odict(obj):
    """Returns `True` if an object is an `OrderedDict` object.

    `is_odict(obj)` returns `True` if the given object `obj` is an instance of
    the `collections.OrderedDict` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `OrderedDict` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `OrderedDict`, otherwise `False`.
    """
    return isinstance(obj, OrderedDict)
from collections import defaultdict
@docwrap
def is_ddict(obj):
    """Returns `True` if an object is a `defaultdict` object.

    `is_ddict(obj)` returns `True` if the given object `obj` is an instance of
    the `collections.defaultdict` type.

    Parameters
    ----------
    obj : object
        The object whose quality as a `defaultdict` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `defaultdict`, otherwise `False`.
    """
    return isinstance(obj, defaultdict)
from pcollections import pdict
@docwrap
def is_pdict(obj):
    """Returns `True` if an object is a persistent dictionary object.

    `is_pdict(obj)` returns `True` if the given object `obj` is an instance of
    the `pcollections.pdict` type and `False` otherwise.

    Parameters
    ----------
    obj : object
        The object whose quality as a `pdict` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `pdict`, otherwise `False`.
    """
    return isinstance(obj, pdict)
from pcollections import tdict
@docwrap
def is_tdict(obj):
    """Returns `True` if an object is a transient dictionary object.

    `is_tdict(obj)` returns `True` if the given object `obj` is an instance of
    the `pcollections.tdict` type and `False` otherwise.

    Parameters
    ----------
    obj : object
        The object whose quality as a `tdict` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `tdict`, otherwise `False`.
    """
    return isinstance(obj, tdict)
from pcollections import ldict
@docwrap
def is_ldict(obj):
    """Returns `True` if an object is a persistent lazy dictionary object.

    `is_ldict(obj)` returns `True` if the given object `obj` is an instance of
    the `pcollections.ldict` type and `False` otherwise.

    Parameters
    ----------
    obj : object
        The object whose quality as a `ldict` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `ldict`, otherwise `False`.
    """
    return isinstance(obj, ldict)
@docwrap
def hashsafe(obj):
    """Returns `hash(obj)` if `obj` is hashable, otherwise returns `None`.

    See also: `can_hash`, `is_ahashable` (note that a fairly reliable test of
    immutability versus mutability for a Python object is whether it is
    hashable).
        
    Parameters
    ----------
    obj : object
        The object to be hashed.

    Returns
    -------
    int
        If the object is hashable, returns the hashcode.
    None
        If the object is not hashable, returns `None`.

    """
    try:
        return hash(obj)
    except TypeError:
        return None
@docwrap
def can_hash(obj):
    """Returns `True` if `obj` is safe to hash and `False` otherwise.

    `can_hash(obj)` is equivalent to `hashsafe(obj) is not None`. This differs
    from `is_ahashable(obj)` in that `is_ahashable` only checks whether `obj` is
    an instance of `Hashable`.

    Note that in Python, all builtin immutable types are hashable while all
    builtin mutable types are not.

    See also: `is_ahashable`, `hashsafe`
    """
    return hashsafe(obj) is not None
@docwrap
def itersafe(obj):
    """Returns an iterator or None if the given object is not iterable.

    `itersafe(obj)` is equivalent to `iter(obj)` with the exception that, if
    `obj` is not iterable, it returns `None` instead of raising an exception.

    Parameters
    ----------
    obj : object
        The object to be iterated.

    Returns
    -------
    iterator or None
        If `obj` is iterable, returns `iter(obj)`; otherwise, returns `None`.
    """
    try:
        return iter(obj)
    except TypeError:
        return None
@docwrap
def can_iter(obj):
    """Returns `True` if `obj` is safe to iterate and `False` otherwise.

    `can_iter(obj)` is equivalent to `itersafe(obj) is not None`. This differs
    from `is_aiterable(obj)` in that `is_aiterable` only checks whether `obj` is
    an instance of `Iterable`.

    See also: `is_aiterable`, `itersafe`
    """
    return itersafe(obj) is not None
@docwrap
def is_pcoll(obj):
    """Detects if an object is a `plist`, `pset`, `pdict`, `llist` or `ldict`.

    `is_pcoll(obj)` returns `True` if the given object `obj` is an instance of
    the persistent collection types `plist`, `pset`, `pdict`, `llist`, or
    `ldict`. Otherwise, `False` is returned.

    Note that this function tests against a specific set of concrete types;
    instances of objects whose types are overloaded versions of these types
    will be treated as instances of the base types; however other immutable
    types not defined in `immlib` or `pcollections` will not be recognized by
    this function.

    Parameters
    ----------
    obj : object
        The object whose quality as a persistent collection is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is a persistent collection and `False` otherwise.
    """
    return isinstance(obj, is_pcoll.types)
is_pcoll.types = (plist, pset, pdict, llist, ldict)
@docwrap
def is_tcoll(obj):
    """Returns `True` if an object is a transient `tlist`, `tset`, or `tdict`.

    `is_tcoll(obj)` returns `True` if the given object `obj` is an instance of
    the `tdict`, `tset`, or `tlist` types, all of which are transient
    collections. Otherwise, `False` is returned.

    Parameters
    ----------
    obj : object
        The object whose quality as a transient collection is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is a `tlist`, `tset`, or `tdict` and `False` otherwise.
    """
    return isinstance(obj, is_tcoll.types)
is_tcoll.types = (tlist, tset, tdict)
@docwrap
def is_mcoll(obj):
    """Returns `True` if an object is a mutable `list`, `set`, or `dict`.

    `is_mcoll(obj)` returns `True` if the given object `obj` is an instance of
    the `dict`, `set`, or `list` types, all of which are mutable
    collections. Otherwise, `False` is returned.

    Parameters
    ----------
    obj : object
        The object whose quality as a mutable collection is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is a `list`, `set`, or `dict` and `False` otherwise.
    """
    return isinstance(obj, is_mcoll.types)
is_mcoll.types = (list, set, dict)
def to_pcoll(obj):
    """Returns a persistent copy of `obj`.

    `to_pcoll(obj)` returns `obj` itself if `obj` is a persistent collection;
    otherwise, it returns a persistent copy of `obj`. If `obj` is not a
    collection that can be converted into a persistent collection, then an error
    is raised.

    Parameters
    ----------
    obj : collection
        An object that is to be converted into a persistent collection.

    Returns
    -------
    object
        A persistent version of `obj`.
    """
    if isinstance(obj, Sequence):
        return plist(obj)
    elif isinstance(obj, Set):
        return pset(obj)
    elif isinstance(obj, Mapping):
        return pdict(obj)
    else:
        raise TypeError(f"argument is not a recognized collection")
def to_tcoll(obj, copy=True):
    """Returns a transient copy of `obj`.

    `to_tcoll(obj)` returns a copy of `obj` as a transient collection. If `obj`
    is not a collection that can be converted into a persistent collection, then
    an error is raised. Note that if `obj` is already a transient collection,
    then a copy is made.

    Note that if `to_tcoll` is given a lazy dict (`ldict`), the resulting
    transient dictionary does not dereference the lazy elements. To prevent
    this, an `ldict` object can be passed as `to_tcoll(obj.to_pdict())`.

    Parameters
    ----------
    obj : collection
        An object that is to be converted into a persistent collection.
    copy : boolean, optional
        If `obj` is already a transient collection, then a copy is made if and
        only if `copy` is `True`; otherwise, `obj` is returned as-is when it is
        already a transient type. The default is `True`.

    Returns
    -------
    object
        A transient version of `obj`. The returned value is always either a
        `tlist`, `tset`, or `tdict` object.
    """
    if not copy and is_tcoll(obj):
        return obj
    if isinstance(obj, (PersistentList, PersistentSet, PersistentMapping)):
        return obj.transient()
    elif isinstance(obj, (TransientList, TransientSet, TransientMapping)):
        # This makes a duplicate but prevents tldicts from becoming tdicts and
        # tllists from becoming tlists.
        return obj.persistent().transient()
    elif isinstance(obj, Sequence):
        return tlist(obj)
    elif isinstance(obj, Set):
        return tset(obj)
    elif isinstance(obj, Mapping):
        return tdict(obj)
    else:
        raise TypeError(f"argument is not a recognized collection")
def to_mcoll(obj):
    """Returns a mutable copy of `obj`.

    `to_mcoll(obj)` returns a mutable copy of the given collection `obj`. If
    `obj` is not a collection that can be converted into a mutable collection,
    then an error is raised.

    Parameters
    ----------
    obj : collection
        An object that is to be converted into a persistent collection.
    copy : boolean, optional
        If `obj` is already a mutable collection, then a copy is made if and
        only if `copy` is `True`; otherwise, `obj` is returned as-is when it is
        already a mutable type. The default is `True`.

    Returns
    -------
    object
        A mutable version of `obj`; the return value's type will always be one
        of `list`, `set`, or `dict`.
    """
    if not copy and is_mcoll(obj):
        return obj
    if isinstance(obj, Sequence):
        return list(obj)
    elif isinstance(obj, Set):
        return set(obj)
    elif isinstance(obj, Mapping):
        return dict(obj)
    else:
        raise TypeError(f"argument is not a collection")
@docwrap
def to_frozenarray(arr, copy=None, subok=False):
    """Returns a copy of the given NumPy array that is read-only.

    If the argument `arr` is already a NumPy `ndarray` with its `'WRITEABLE'`
    flag set to `False`, then `arr` is returned; otherwise, a copy is made, its
    write-flag is set to `False`, and it is returned.

    If `arr` is not a NumPy array, it is first converted into one.

    Parameters
    ----------
    arr : array-like
        The NumPy array to freeze or an object to convert into a frozen NumPy
        array.
    copy : boolean or None, optional
        If `True` or `False`, then forces the array to be copied or not copied;
        if `None`, then a copy is made if the object is writeable and no copy
        is made otherwise.
    subok : boolean, optional
        If True, then sub-classes will be passed-through, otherwise the returned
        array will be forced to be a base-class array (defaults to False).

    Returns
    -------
    numpy.ndarray
        A read-only copy of `arr` (or `arr` itself if it is already read-only).

    Raises
    ------
    TypeError
        If `arr` is not a NumPy array.
    """
    if sps.issparse(arr):
        if not (arr.data.flags['WRITEABLE'] or copy):
            return arr
        arr = arr.copy()
        arr.data.flags['WRITEABLE'] = False
    elif not isinstance(arr, np.ndarray):
        arr = np.array(arr, copy=(copy is None or copy), subok=subok)
        copy = False
    rw = arr.flags['WRITEABLE']
    if copy is None:
        copy = rw
    if copy:
        arr = np.copy(arr, subok=subok)
    if rw:
        arr.setflags(write=False)
    return arr
@docwrap
def frozenarray(obj, *args, **kwargs):
    """Equivalent to `numpy.array` but returns read-only arrays.

    `frozenarray(obj)` is equivalent to `numpy.array(obj)` with the exception
    that the returned object is always a frozen array (i.e., an array with the
    `'WRITEABLE'` flag set to `False`). If `copy=False` is requested, but the
    passed array is writeable, then an error is raised; if you wish to convert
    an array to a read-only array without copying, you must use the
    `to_frozenarray` function.

    See Also
    --------
    `numpy.array`
        Create an array that is not frozen by default.
    `to_frozenarray`
        Convert an argument to a frozen array (allows `copy=False`).
    """
    arr = np.array(obj, *args, **kwargs)
    if arr is obj and arr.flags['WRITEABLE']:
        arr = arr.copy()
    arr.setflags(write=False)
    return arr


# Mapping/Sequence Utilities ###################################################

@docwrap
def get(d, arg, **kw):
    """Returns a value from either a mapping or a sequence.

    The `get` function is essentially a function version of the `get` method
    that works for both `Mapping` and `Sequence` types (e.g., `dict`s, `list`s,
    `tuple`s, and related types that implement these abstract bases).

    `get(data, k)` extracts element `k` from `data` and returns it. If `k` is
    not a valid index of `data` (i.e., `k` is not a key of `data` if `data` is a
    mapping or is not an integer index of `data` if `data` is a sequence), then
    the optional value `default` is returned. If `detault` is not explicitly
    provided, then an error is raised. Note that if a non-integer key is
    provided for a sequence, this is treated as a missing index.

    Parameters
    ----------
    d : object
        The dict-like or list-like object from which an element is being
        extracted.
    k : object
        The key or index into `d` whose value is to be extracted.
    default : object, optional
        The default value to be returned if an item is not found. If this option
        is not provided, then an error is raised should the item not be found.
    """
    if "default" in kw:
        error = False
        default = kw.pop("default")
    else:
        error = True
        default = None
    if len(kw) > 0:
        k = repr(next(iter(kw.keys())))
        raise TypeError(f"get() got an unexpected keyword argument {k}")
    if isinstance(d, (Mapping, Sequence)):
        try:
            return d[k]
        except IndexError:
            pass
        except TypeError:
            pass
        except KeyError:
            pass
    else:
        raise TypeError(f"cannot get item from type {type(d)}")
    # If we reach this point, we failed to find the key.
    if error:
        raise KeyError(k)
    else:
        return default
@docwrap
def nestget(d, *args, **kw):
    """Returns a value from a data structure of nested mappings and sequences.

    The `nestget` function is essentially a nested version of the `get` method
    that works for both `Mapping` and `Sequence` types (e.g., `dict`s, `list`s,
    `tuple`s, and related types that implement these abstract bases).

    `nestget(data, k1, k2, k3...)` extracts element `k1` from `data` then
    element `k2` from that value, then element `k3` from that value, etc., until
    there are no more keys; the final value is returned. If any of the values
    are missing, then the optional value `default` is returned if it is provided
    and an error is raised if it is not. Note that the provided keys may be
    integer indices for list-like objects that may be included in the
    nesting. If a string key is given for a list-like container, then this is
    treated as a missing key, not an error.

    Parameters
    ----------
    d : object
        The dict-like or list-like object from which an element is being
        extracted.
    *args
        The list of keys and indices to be extracted.
    default : object, optional
        The default value to be returned if an item is not found. If this option
        is not provided, then an error is raised should the item not be found.

    """
    if "default" in kw:
        error = False
        default = kw.pop("default")
    else:
        error = True
        default = None
    if len(kw) > 0:
        k = repr(next(iter(kw.keys())))
        raise TypeError(f"nestget() got an unexpected keyword argument {k}")
    for k in args:
        if isinstance(d, Mapping):
            if k in d:
                d = d[k]
                continue
        elif isinstance(d, Sequence):
            try:
                d = d[k]
                continue
            except IndexError:
                pass
            except TypeError:
                pass
        else:
            raise TypeError(f"cannot get item from type {type(d)}")
        # If we reach this point, we failed to find the key.
        if error:
            raise KeyError(k)
        else:
            return default
    return d
from pcollections import lazy
def _lazyvalmap_extract(f, d, k, *args, **kw):
    return f(d[k], *args, **kw)
def lazyvalmap(f, d, *args, **kwargs):
    """Returns a dict object whose values are transformed by a function.

    `lazyvalmap(f, d)` returns a dict whose keys are the same as those of the
    given dict object and whose values, for each key `k` are `f(d[k])`. All
    values are created lazily.

    `lazyvalmap(f, d, *args, **kw)` additionally passes the given arguments to
    the function `f`, such that in the resulting map, each key `k` is mapped to
    `f(d[k], *args, **kw)`.

    This function always returns a `ldict` object.
    """
    t = tdict()
    if is_ammap(d):
        # For mutable maps, we do not try to respect laziness; they may change
        # so we cannot rely on them.
        for (k,v) in d.items():
            t[k] = lazy(f, v, *args, **kwargs)
    else:
        # Anything else we assume is immutable, so we respect any possible lazy
        # implementation.
        for k in d.keys():
            t[k] = lazy(_lazyvalmap_extract, f, d, k, *args, **kwargs)
    return ldict(t)
def valmap(f, d, *args, **kwargs):
    """Returns a dict object whose values are transformed by a function.

    `valmap(f, d)` yields a dict whose keys are the same as those of the given
    dict object and whose values, for each key `k` are `f(d[k])`.

    `valmap(f, d, *args, **kw)` additionally passes the given arguments to the
    function `f`, such that in the resulting map, each key `k` is mapped to
    `f(d[k], *args, **kw)`.

    Unlike `lazyvalmap`, this function returns either a `dict`, a `pdict`, or an
    `ldict` depending on the input argument `d`. If `d` is a (lazy) `ldict`,
    then an `ldict` is returned; if `d` is a `pdict`, a `pdict` is returned, and
    otherwise, a `dict` is returnd.
    """
    if is_ldict(d):
        return lazyvalmap(f, d, *args, **kwargs)
    elif is_pdict(d):
        t = tdict()
        for (k,v) in d.items():
            t[k] = v
        return t.persistent()
    else:
        return {k: f(v, *args, **kwargs) for (k,v) in d.items()}
def lazykeymap(f, d, *args, **kwargs):
    """Returns a `ldict` object whose values are a function of a mapping's keys.

    `keymap(f, d)` yields a dict whose keys are the same as those of the given
    dict object and whose values, for each key `k` are `f(k)`.

    `keymap(f, d, *args, **kw)` additionally passes the given arguments to the
    function `f`, such that in the resulting map, each key `k` is mapped to
    `f(k, *args, **kw)`.

    This function always returns an `ldict` whose values have not yet been
    cached.
    """
    t = tdict()
    for k in d.keys():
        t[k] = lazy(f, k, *args, **kwargs)
    return ldict(t)
def keymap(f, d, *args, **kwargs):
    """Returns a dict object whose values are a function of a dict's keys.

    `keymap(f, d)` yields a dict whose keys are the same as those of the given
    dict object and whose values, for each key `k` are `f(k)`.

    `keymap(f, d, *args, **kw)` additionally passes the given arguments to the
    function `f`, such that in the resulting map, each key `k` is mapped to
    `f(k, *args, **kw)`.

    This function returns either a `dict`, a `pdict`, or an `ldict` depending on
    the input argument `d`. If `d` is an `ldict`, then an `ldict` is returned;
    if `d` is a `pdict`, a `pdict` is returned, and otherwise, a `dict` is
    returnd.
    """
    if is_ldict(d):
        return lazykeymap(f, d, *args, **kwargs)
    elif is_pdict(d):
        t = tdict()
        for k in d.keys():
            t[k] = f(k, *args, **kwargs)
        return t.persistent()
    else:
        return {k: f(k, *args, **kwargs) for k in d.keys()}
def _lazyitemmap_extract(f, d, k, *args, **kw):
    return f(k, d[k], *args, **kw)
def lazyitemmap(f, d, *args, **kwargs):
    """Returns an `ldict` object whose values are a function of a dict's items.

    `lazyitemmap(f, d)` yields an `ldict` whose keys are the same as those of
    the given dict object and whose values, for each key `k`, are lazily
    computed as `f(k, d[k])`.

    `itemmap(f, d, *args, **kw)` additionally passes the given arguments to the
    function `f`, such that in the resulting map, each key `k` is mapped to
    `f(k, d[k], *args, **kw)`.
    """
    t = tdict()
    if is_ammap(d):
        # For mutable maps, we do not try to respect laziness; they may change
        # so we cannot rely on them.
        for (k,v) in d.items():
            t[k] = lazy(f, k, v, *args, **kwargs)
    else:
        # Otherwise, we assume that it's an immutable map, and we respect any
        # possible laziness that could be implemented.
        for k in d.keys():
            t[k] = lazy(_lazyitemmap_extract, f, d, k, *args, **kwargs)
    return ldict(t)
def itemmap(f, d, *args, **kwargs):
    """Returns a dict object whose values are a function of a dict's items.

    `itemmap(f, d)` yields a dict whose keys are the same as those of the given
    dict object and whose values, for each key `k` are `f(k, d[k])`.

    `itemmap(f, d, *args, **kw)` additionally passes the given arguments to the
    function `f`, such that in the resulting map, each key `k` is mapped to
    `f(k, d[k], *args, **kw)`.

    Unlike `lazyitemmap`, this function returns either a `dict`, a `pdict`, or
    an `ldict` depending on the input argument `d`. If `d` is an `ldict`, then
    an `ldict` is returned; if `d` is a `pdict`, a `pdict` is returned, and
    otherwise, a `dict` is returnd.
    """
    if is_ldict(d):
        return lazyitemmap(f, d, *args, **kwargs)
    elif is_pdict(d):
        t = tdict()
        for (k,v) in d.items():
            t[k] = f(k, v, *args, **kwargs)
        return t.persistent()
    else:
        return {k: f(k, v, *args, **kwargs) for (k,v) in d.items()}
def dictmap(f, keys, *args, **kw):
    """Returns a dict with the given keys and the values `map(f, keys)`.

    `dictmap(f, keys)` returns a dict object whose keys are the elements of
    `iter(keys)` and whose values are the elements of `map(f, keys)`.

    `dictmap(f, keys, *args, **kw)` returns a dict object whose keys are the
    elements of `iter(keys)` and whose values are the elements of
    `[f(k, *args, **kw) for k in iter(keys)]`.
    """
    return {k: f(k, *args, **kw) for k in keys}
def pdictmap(f, keys, *args, **kw):
    """Returns a `pdict` with the given keys and the values `map(f, keys)`.

    `pdictmap(f, keys)` returns a dict object whose keys are the elements
    of `iter(keys)` and whose values are the elements of `map(f, keys)`.

    `pdictmap(f, keys, *args, **kw)` returns a dict object whose keys are
    the elements of `iter(keys)` and whose values are the elements of
    `[f(k, *args, **kw) for k in iter(keys)]`.
    """
    t = tdict()
    for k in keys:
        t[k] = f(k, *args, **kw)
    return t.persistent()
def ldictmap(f, keys, *args, **kw):
    """Returns a lazy `ldict` with the given keys and the values `map(f, keys)`.

    `lazydictmap(f, keys)` returns a dict object whose keys are the elements of
    `iter(keys)` and whose values are the elements of `map(f, keys)`. All values
    are uncached and lazy.

    `lazydictmap(f, keys, *args, **kw)` returns a dict object whose keys are the
    elements of `iter(keys)` and whose values are the elements of
    `[f(k, *args, **kw) for k in iter(keys)]`.
    """
    t = tdict()
    for k in keys:
        t[k] = lazy(f, k, *args, **kw)
    return t.persistent()
def merge(*args, **kw):
    '''Merges dict-like objects left-to-right. See also `rmerge`.

    `merge(...)` collapses all arguments, which must be python `Mapping` objects
    of some kind, into a single mapping from left-to-right. The mapping that is
    returned depends on the inputs: if any of the input mappings are lazydict
    objects, then a lazydict is returned (and the laziness of arguments is
    respected); otherwise, a frozendict object is retuend.

    Note that optional keyword arguments may be passed; these are considered the
    right-most dictionary argument.
    '''
    if len(args) == 0:
        return pdict(kw)
    # Make the initial dictionary.
    res = args[0]
    lazy = is_ldict(res)
    res = tdict(res)
    for d in args[1:]:
        if is_ldict(d):
            lazy = True
            res.update(d.to_pdict())
        else:
            res.update(d)
    res.update(kw)
    return ldict(res) if lazy else pdict(res)
def rmerge(*args, **kw):
    '''Merges dict-like objects right-to-left. See also `merge`.

    `rmerge(...)` collapses all arguments, which must be python `Mapping`
    objects of some kind, into a single mapping from right-to-left. The mapping
    that is returned depends on the inputs: if any of the input mappings are
    lazydict objects, then a lazydict is returned (and the laziness of arguments
    is respected); otherwise, a frozendict object is retuend.

    Note that optional keyword arguments may be passed; these are considered the
    right-most dictionary argument.
    '''
    if len(args) == 0:
        return pdict(kw)
    # Make the initial dictionary.
    res = tdict(kw)
    lazy = False
    for d in reversed(args):
        if is_ldict(d):
            lazy = True
            res.update(d.to_pdict())
        else:
            res.update(d)
    res.update(kw)
    return ldict(res) if lazy else pdict(res)
def assoc(d, *args, **kw):
    """Returns a copy of the given dictionary with additional key-value pairs.

    `assoc(d, key, val)` returns a copy of the dictionary `d` with the given
    key-value pair associated in the new copy. The return value is always the
    same type as the argument `d`.

    `assoc(d, key1, val1, key2, val2 ...)` associates all the given keys to the
    given values in the returned copy.

    `assoc(d, key1=val1, key2=val2 ...)` uses the keyword arguments as the
    arguments that are to be associated.
    """
    if len(args) % 2 != 0:
        raise ValueError("assoc requires matched key-value arguments")
    ks = args[0::2]
    vs = args[1::2]
    if is_ammap(d):
        # This is a mutable mapping, so we copy it.
        d = d.copy()
        for (k,v) in zip(ks,vs):
            d[k] = v
        for (k,v) in kw.items():
            d[k] = v
    elif is_pdict(d):
        nels = len(ks) + len(kw)
        if nels > 1:
            d = d.transient()
            for (k,v) in zip(ks,vs):
                d[k] = v
            for (k,v) in kw.items():
                d[k] = v
            d = d.persistent()
        else:
            for (k,v) in zip(ks,vs):
                d = d.set(k, v)
            for (k,v) in kw.items():
                d = d.set(k, v)
    else:
        raise TypeError(f"cannot assoc to type {type(d)}")
    return d
def dissoc(d, *args):
    """Returns a copy of the given dictionary with certain keys removed.

    `dissoc(d, key)` returns a copy of the dictionary `d` with the given
    key disssociated in the new copy. The return value is always the
    same type as the argument `d`.

    `dissoc(d, key1, key2 ...)` dissociates all the given keys from their values
    in the returned copy.
    """
    if is_ammap(d):
        # This is a mutable mapping, so we copy it.
        d = d.copy()
        for k in args:
            if k in d:
                del d[k]
        return d
    elif is_pdict(d):
        if len(args) == 1:
            return d.delete(args[0])
        else:
            d = d.transient()
            for k in args:
                del d[k]
            return d.persistent()
    else:
        raise TypeError(f"cannot dissoc from type {type(d)}")
def _lambdadict_call(data, fn):
    spec = inspect.getfullargspec(fn)
    dflts = spec.defaults or fdict()
    args = []
    kwargs = {}
    pos = True
    for k in spec.args:
        if k in data:
            v = undelay(data[k])
            if pos: args.append(v)
            else:   kwargs[k] = v
        else:
            pos = False
            if k in dflts: kwargs[k] = dflts[k]
    for k in spec.kwonlyargs:
        if k in data:
            kwargs[k] = undelay(data[k])
        else:
            if k in dflts: kwargs[k] = dflts[k]
    return fn(*args, **kwargs)
def lambdadict(*args, **kwargs):
    """Builds and returns a `ldict` with lambda functions calculated lazily.

    `lambdadict(args...)` is equivalent to `ldict(args...)` except that any
    lambda function in the values provided by the merged arguments is made into
    a lazy partial function whose inputs come from the lambda-function variable
    names in the same resulting `ldict`.

    WARNING: This function will gladly return an `ldict` that encapsulates an
    infinite loop if you are not careful. For example, the following lambdadict
    will infinite loop when either key is requested:
    `ld = lambdadict(a=lambda b:b, b=lambda a:a)`.
    
    Examples
    --------
    >>> d = lambdadict(a=1, b=2, c=lambda a,b: a + b)
    >>> d.is_cached('c')
    False
    >>> d['c']
    3
    >>> d
    {|'a': 1, 'b': 2, 'c': 3|}
    """
    d = merge(*args, **kwargs)
    finals = d.transient()
    for (k,v) in d.to_pdict().items():
        if isinstance(v, LambdaType):
            finals[k] = lazy(_lambdadict_call, finals, v)
        else:
            finals[k] = v
    return ldict(finals)


# Argument Utilities ###########################################################

class args:
    """An object type that represents a set of function arguments.

    `args(x1, x2 ... k1=v1, k2=v2 ...)` yields an `args` object that represents
    the parameters `x1, x2 ...` and the named parameters `k1=v1, k2=v2 ...`.

    If `a` is an instance of `args` and `f` is a function, then the arguments in
    `a` can be applied to `f` using either of the following methods:
     * `f @ a`
     * `a.passto(f)`
    Note that if `f` is an object that defines the `__matmul__` method, then the
    former syntax will call that method instead of the `__rmatmul__` method of
    the `args` object `a` and thus won't work.
    """
    __slots__ = ('args', 'kwargs')
    def __setattr__(self, k, v):
        raise TypeError("args type is immutable")
    @classmethod
    def _new(cls, args, kwargs):
        self = object.__new__(cls)
        object.__setattr__(self, 'args', tuple(args))
        object.__setattr__(self, 'kwargs', ldict(kwargs))
        return self
    def __new__(cls, *args, **kwargs):
        return cls._new(args, kwargs)
    def __rmatmul__(self, fn):
        return fn(*self.args, **self.kwargs)
    def passto(self, fn):
        return fn(*self.args, **self.kwargs)
    def copy(self, args=None, kwargs=None):
        if args is None:
            args = self.args
        if kwargs is None:
            kwargs = self.kwargs
        if args is self.args and kwargs is self.kwargs:
            return self
        return self._new(args, kwargs)
def argfilter(fn):
    """A decorator that creates decorators that filter function arguments.

    A function decorated with `@argfilter` or `@argfilter(k1=v1, k2=v2...)` is
    turned into a an argument filter, which itself can be used to decorate
    functions whose arguments need to be filtered.

    In the definition of the filter function, the names of arguments must match
    those of the arguments they are filtering on other functions. The function
    must return a tuple of the filtered values in the order they are defined in
    the function's argument list. The arguments may be given in any order, but a
    `*` in the arguments list indicates that any of the arguments following the
    `*` are not themselves being filtered and thus will not be returned from the
    filter function.

    The argument filter functions (i.e., the functions that are decorated by
    `@argfilter`) are of type `ArgFilter` and their base filter functions
    can be called using `function.filter(args...)`.

    Examples
    --------
    >>> @argfilter
    ... def fix_angle(angle, *, unit):
    ...     angle = np.asarray(angle)
    ...     if unit == 'degrees':
    ...         angle = np.pi / 180 * angle
    ...     elif unit != 'radians':
    ...         raise ValueError(f'unrecognized unit: {unit}')
    ...     return (angle,)
    ... @fix_angle
    ... def cos_halfangle(angle, unit='radians'):
    ...     return np.cos(angle / 2)

    >>> cos_halfangle([0, 360], 'degrees')
        array([1., -1.])

    """
    sig = signature(fn)
    pos_args = {}
    pok_args = {}
    for (k,u) in sig.parameters.items():
        if u.kind == u.POSITIONAL_ONLY or u.kind == u.POSITIONAL_OR_KEYWORD:
            pos_args[k] = u
        elif u.kind == u.KEYWORD_ONLY:
            pok_args[k] = u
        else:
            raise ValueError("variadic filter arguments are not supported")
    # We make a function that uses these when it is used as a decorator. There
    # are two ways to call this decorator: one as @filter_func and the other is
    # @filter_func(origname1=newname1, origname2=newname2...) in order to
    # indicate that some of the arguments need to be renamed. These are written
    # as the private functions below and are then wrapped up into a partial.
    def fn_decr(f=None, /, **kwargs):
        return _argfilter_decr(pos_args, pok_args, fn, f, **kwargs)
    return wraps(fn)(fn_decr)
def _argfilter_decr(pos_args, pok_args, filter_fn, fn=None, /, **kwargs):
    if len(kwargs) == 0:
        if fn is None:
            raise ValueError(
                "argfilter decorator requires either a function or options")
        else:
            return _argfilter_init(pos_args, pok_args, filter_fn, {}, fn)
    elif fn is None:
        return partial(
            _argfilter_init,
            pos_args, pok_args, filter_fn,
            kwargs)
    else:
        raise ValueError(
            "argfilter decorator requires a function or options, but not both")
def _argfilter_init(pos_args, pok_args, filter_fn, tr, f):
    sig = signature(f)
    params = sig.parameters
    # Prep some data structures so that we can quickly extract the filtered
    # arguments when the function is called.
    keys = []
    argdat = []
    kwdat = {}
    for (args, dat) in ((pos_args,argdat), (pok_args,kwdat)):
        for (k0,arg) in args.items():
            k = tr.get(k0, k0)
            dflt = arg.default
            if k in params:
                p = params[k]
                if p.default is not p.empty:
                    dflt = p.default
            else:
                if dflt is arg.empty:
                    if k0 != k:
                        k = f'{k0} ({k})'
                    raise ValueError(f"filtered parameter {k} not found")
            app = (k, dflt)
            if isinstance(dat, list):
                dat.append(app)
            else:
                dat[k0] = app
    # Now pass these along to the argilter initialization function.
    keys = tuple(u[0] for u in argdat)
    fn = partial(_argfilter_dispatch, filter_fn, f, sig, argdat, kwdat, keys)
    return wraps(f)(fn)
def _argfilter_dispatch(filter_fn, f, fsig,
                        filt_argdat, filt_kwdat, filt_keys,
                        *args, **kwargs):
    b = fsig.bind(*args, **kwargs)
    argmap = b.arguments
    filtered_vals = filter_fn(
        *(argmap.get(k, d) for (k,d) in filt_argdat),
        **{k0: argmap.get(k, d) for (k0,(k,d)) in filt_kwdat.items()})
    # The number of filtered items must be equal to the number we expect.
    if len(filtered_vals) != len(filt_keys):
        raise ValueError(
            f"filter on function {f.__name__} returned {len(filtered_vals)}"
            f" items but expected {len(filt_keys)}")
    argmap.update(zip(filt_keys, filtered_vals))
    return f(*b.args, **b.kwargs)


# unitregistry #################################################################

# We put the unitregistry here and not in the quantity namespace because we need
# it both for quantity and numeric and it causes a circular import if placed in
# the quantity file.
@docwrap
def unitregistry(obj, *args):
    """Returns the pint UnitRegistry object for the given unit or quantity.

    `unitregistry(u)` for a `pint.Unit` object `u` returns the
    `pint.UnitRegistry` object in which `u` is registered.

    `unitregistry(q)` for a `pint.Quantity` object `q` returns the
    `pint.UnitRegistry` object in which `q` is registered.

    `unitregistry(ureg)` returns `ureg` is `ureg` is itself a `UnitRegistry`.

    `unitregistry(Ellipsis)` returns `immlib.units`.

    `unitregistry(x)` raises a `TypeError` for any other type of object `x`.

    `unitregistry(x, default)` returns `unitregistry(x)` unless a `TypeError`
    would be raised, in which case it returns the given `default` value. If
    `default` is `Ellipsis`, then `immlib.units` is used as the default.
    """
    nargs = len(args)
    if nargs > 1:
        raise TypeError(
            f"unitregistry() takes from 1 to 2 positional arguments but"
            f" {1+len(args)} were given")
    elif isinstance(obj, (pint.Unit, pint.Quantity)):
        return obj._REGISTRY
    elif isinstance(obj, pint.UnitRegistry):
        return obj
    elif obj is Ellipsis:
        from immlib import units
        return units
    elif len(args) == 0:
        raise TypeError(
            f"unitregistry() cannot convert object of type {type(obj)} to a"
            f" pint.UnitRegistry")
    else:
        default = args[0]
        if default is Ellipsis:
            from immlib import units as default
        return default
