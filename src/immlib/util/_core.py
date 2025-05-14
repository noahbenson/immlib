# -*- coding: utf-8 -*-
###############################################################################
# immlib/util/_core.py


# Dependencies ################################################################

import operator as op
from inspect import (signature, getfullargspec)
from functools import (wraps, partial, lru_cache)
from joblib import Memory
from pathlib import Path

import pint
import numpy as np
import scipy.sparse as sps
from pcollections import holdlazy

from ..doc import docwrap


# Strings #####################################################################

@docwrap('immlib.is_str')
def is_str(obj):
    """Returns ``True`` if an object is a string and ``False`` otherwise.

    ``is_str(obj)`` returns ``True`` if the given object `obj` is an instance 
    of the ``str`` type and ``False`` otherwise.

    Parameters
    ----------
    obj : object
        The object whose quality as a string object is to be assessed.

    Returns
    -------
    bool
        `True` if `obj` is a string, otherwise `False`.
    """
    return isinstance(obj, str)
from unicodedata import normalize as unicodedata_normalize
@docwrap('immlib.strnorm')
def strnorm(s, /, case=False, *, unicode=True):
    """Normalizes a string using the ``unicodedata`` package.

    ``strnorm(s)`` returns a version of `s` that has been unicode-normalized
    using the ``unicodedata.normalize(s)`` function. Case-normalization can
    also be requested via the `case` option.

    Parameters
    ----------
    s : object
        The string to be normalized.
    case : bool, optional
        Whether to perform case-normalization (``case=True``) or not
        (``case=False``, the default). If two strings are case-normalized, then
        an equality comparison will reveal whether the original (unnormalized
        strings) were equal up to the case of the characters. Case
        normalization is performed using the ``str.casefold()`` method.
    unicode : bool or str, optional
        Whether to perform unicode normalization via the
        ``unicodedata.normalize`` function. The default behavior
        (``unicode=True``) is to perform normalization, but this can be
        disabled with ``unicode=False``. Alternatively, a string may be given,
        in which case it is passed to the ``unicodedata.normalize`` function as
        the first argument; when `unicode` is ``True``, the string used is
        ``'NFD'``.

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
    # When unicode is None, we do its normalization only when case
    # normalization is being done.
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
@docwrap('immlib.strcmp')
def strcmp(a, b, /, case=True, *, unicode=None, strip=False, split=False):
    """Determines if the given objects are strings and compares them if so.

    ``strcmp(a, b)`` returns ``None`` if either `a` or `b` is not a string;
    otherwise, it returns ``-1``, ``0``, or ``1`` if `a` is less than, equal to,
    or greater than `b`, respectively, subject to the constraints of the
    parameters.

    Parameters
    ----------
    a : object
        The first argument.
    b : object
        The second argument.
    case : bool, optional
        Whether to perform case-sensitive (``case=True``) or case-insensitive
        (``case=False``) string comparison. The default is ``False``.
    unicode : bool or None, optional
        Whether to run unicode normalization on `a` and `b` prior to
        comparison. By default, this is ``None``, which is interpreted as a
        `True` value when `case` is ``False`` and as ``False`` value when
        `case` is ``True``.  In other words, unicode normalization is performed
        when case-insensitive comparison is being performed but not when
        standard string comparison is being performed. Unicode normalization is
        always performed both before and after casefolding. Unicode
        normalization is performed using the ``unicodedata`` package's
        ``normalize(unicode, string)`` function. If this argument is a string,
        it is instead passed to the ``normalize`` function as the first
        argument. When `unicode` is not a string but normalization is
        performed, them the default string is ``'NFD'``.
    strip : bool, optional
        If set to ``True``, then ``a.strip()`` and ``b.strip()`` are used in
        place of `a` and `b`. If set to ``False`` (the default), then no
        stripping is performed. If a non-boolean value is given, then it is
        passed as an argument to the ``strip()`` method.
    split : bool, optional
        If set to ``True``, then ``a.split()`` and ``b.split()`` are used in
        place of `a` and `b`. The lists of strings that result from
        ``a.split()`` and ``b.split()`` are rejoined with no separator prior to
        comparison. If this option is set to `False` (the default), then no
        splitting is performed. If a non-boolean value is given, then it is
        passed as an argument to the ``split()`` method.

    Returns
    -------
    bool or None
        ``None`` if either `a` is not a string or `b` is not a string;
        otherwise, ``-1`` if `a` is lexicographically less than `b`, ``0`` if
        ``a == b``, and ``1`` if `a` is lexicographically greater than `b`,
        subject to the constraints of the optional parameters.

    See Also
    --------
    strnorm, streq

    """
    prep = _strbinop_prep(a, b, case=case, unicode=unicode, strip=strip)
    if prep is None:
        return None
    (a, b) = prep
    # If the split argument is true-ish, then we need to eliminate spaces.
    if split is not False:
        if split is True:
            a = a.split()
            b = b.split()
        else:
            a = a.split(split)
            b = b.split(split)
        # We can do this comparison by re-joining the strings with an empty
        # separator. The lexicographically smaller string (excepting the
        # spacers, which are now removed) will still be lexicographically
        # smaller.
        a = ''.join(a)
        b = ''.join(b)
    return (-1 if a < b else 1 if a > b else 0)
@docwrap('immlib.streq')
def streq(a, b, /, case=True, *, unicode=None, strip=False, split=False):
    """Determines if the given objects are equal strings or not.

    ``streq(a, b)`` returns ``True`` if `a` and `b` are both strings and are
    equal to each other, subject to the constraints of the options.

    Parameters
    ----------
    %(immlib.strcmp.parameters)s

    Returns
    -------
    bool or None
        If `a` and `b` are both strings then ``True`` is returned if `a` equals
        `b` and ``False`` is returned otherwise.  If either `a` or `b` is not a
        string, then ``None`` is returned.

    See Also
    --------
    strnorm, strcmp
    """
    cmpval = strcmp(a, b, case=case, unicode=unicode, strip=strip, split=split)
    return None if cmpval is None else (cmpval == 0)
@docwrap('immlib.strends')
def strends(a, b, /, case=True, *, unicode=None, strip=False):
    """Determines whether or not the string `a` ends with the string `b`.

    ``strends(a, b)`` returns ``True`` if `a` and `b` are both strings and if
    `a` ends with `b`, subject to the constraints of the parameters.

    Parameters
    ----------
    %(immlib.strcmp.parameters.case)s
    %(immlib.strcmp.parameters.unicode)s
    %(immlib.strcmp.parameters.strip)s

    Returns
    -------
    bool or None
        If `a` and `b` are both strings then ``True`` is returned if `a` ends
        with `b` and ``False`` is returned otherwise.  If either `a` or `b` is
        not a string, then ``None`` is returned.
    """
    prep = _strbinop_prep(a, b, case=case, unicode=unicode, strip=strip)
    if prep is None: return None
    else: (a, b) = prep
    # Check the ending
    return a.endswith(b)
@docwrap('immlib.strstarts')
def strstarts(a, b, /, case=True, *, unicode=None, strip=False):
    """Determines whether or not the string `a` starts with the string `b`.

    ``strstarts(a, b)`` returns ``True`` if `a` and `b` are both strings and if
    `a` starts with `b`, subject to the constraints of the parameters.

    Parameters
    ----------
    %(immlib.strcmp.parameters.case)s
    %(immlib.strcmp.parameters.unicode)s
    %(immlib.strcmp.parameters.strip)s

    Returns
    -------
    bool or None
        ``True` if `a` and `b` are both strings and if `a` startss with `b`,
        subject to the constraints of the optional parameters. If either `a` or
        `b` is not a string, then ``None`` is returned.
    """
    prep = _strbinop_prep(a, b, case=case, unicode=unicode, strip=strip)
    if prep is None: return None
    else: (a, b) = prep
    # Check the beginning.
    return a.startswith(b)
@docwrap('immlib.strissym')
def strissym(s):
    """Determines if the given string is a valid symbol (identifier).

    ``strissym(s)`` returns ``True`` if `s` is both a string and a valid
    identifier.  Otherwise, it returns ``False`` if `s` is a string and
    ``None`` if not.

    See also
    --------
    striskey, strisvar
    """
    return s.isidentifier() if is_str(s) else None
from keyword import iskeyword
@docwrap('immlib.striskey')
def striskey(s):
    """Determines if the given string is a valid keyword.

    ``strissym(s)`` returns ``True`` if `s` is both a string and a valid keyword
    (such as ``'if'`` or ``'while'``). Otherwise, it returns ``False`` if `s` is a
    string and ``None`` if not.

    See Also
    --------
    strissym, strisvar
    """
    return iskeyword(s) if is_str(s) else None
@docwrap('immlib.strisvar')
def strisvar(s):
    """Determines if the given string is a valid variable name.

    ``strissym(s)`` returns ``True`` if `s` is both a string and a valid name
    (i.e., a symbol but not a keyword). Otherwise, it returns ``False`` if `s`
    is a string and ``None`` if not.

    See Also
    --------
    strissym, striskey
    """
    return (
        None  if not is_str(s) else
        False if iskeyword(s)  else
        s.isidentifier())


# Builtin Python Abstract Types ###############################################

from collections.abc import Callable
@docwrap('immlib.is_acallable')
def is_acallable(obj):
    """Returns ``True`` if an object is a callable object like a function.

    ``is_acallable(obj)`` returns ``True`` if the given object `obj` is an
    instance of the abstract callable type, ``collections.abc.Callable``. Note
    that in general, it is more common and preferable to use the builtin
    ``callable`` function; however, ``is_acallable`` is included in ``immlib``
    for completeness.

    Parameters
    ----------
    obj : object
        The object whose quality as an ``Callable`` object is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``Callable``, otherwise ``False``.
    """
    return isinstance(obj, Callable)
from types import LambdaType
@docwrap('immlib.is_lambda')
def is_lambda(obj):
    """Returns ``True`` if an object is a lambda function, otherwise ``False``.

    ``is_lambda(obj)`` returns ``True`` if the given object `obj` is an
    instance of the ``types.LambdaType`` type.

    Parameters
    ----------
    obj : object
        The object whose quality as a ``LambdaType`` object is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``LambdaType``, otherwise
        ``False``.

    """
    return isinstance(obj, LambdaType)
from collections.abc import Sized
@docwrap('immlib.is_asized')
def is_asized(obj):
    """Returns ``True`` if an object implements ``len()``, otherwise ``False``.

    ``is_asized(obj)`` returns ``True`` if the given object `obj` is an
    instance of the abstract ``collections.abc.Sized`` type.

    Parameters
    ----------
    obj : object
        The object whose quality as a ``Sized`` object is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``Sized``, otherwise ``False``.

    """
    return isinstance(obj, Sized)
from collections.abc import Container
@docwrap('immlib.is_acontainer')
def is_acontainer(obj):
    """Returns ``True`` if an object implements ``__contains__``, otherwise
    ``False``.

    ``is_acontainer(obj)`` returns ``True`` if the given object `obj` is an
    instance of the abstract ``collections.abc.Container`` type.

    Parameters
    ----------
    obj : object
        The object whose quality as a ``Container`` object is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``Container``, otherwise ``False``.
    """
    return isinstance(obj, Container)
from collections.abc import Iterable
@docwrap('immlib.is_aiterable')
def is_aiterable(obj):
    """Returns ``True`` if an object implements ``__iter__``, otherwise
    ``False``.

    ``is_aiterable(obj)`` returns ``True`` if the given object `obj` is an
    instance of the abstract ``collections.abc.Iterable`` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an ``Iterable`` object is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``Iterable``, otherwise ``False``.
    """
    return isinstance(obj, Iterable)
from collections.abc import Iterator
@docwrap('immlib.is_aiterator')
def is_aiterator(obj):
    """Returns ``True`` if an object is an instance of
    ``collections.abc.Iterator``.

    ``is_aiterable(obj)`` returns ``True`` if the given object `obj` is an
    instance of the abstract ``collections.abc.Iterator`` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an ``Iterator`` object is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``Iterator``, otherwise ``False``.
    """
    return isinstance(obj, Iterator)
from collections.abc import Reversible
@docwrap('immlib.is_areversible')
def is_areversible(obj):
    """Returns ``True`` if an object is an instance of ``Reversible``.

    ``is_areversible(obj)`` returns ``True`` if the given object `obj` is an
    instance of the abstract ``collections.abc.Reversible`` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an ``Reversible`` object is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``Reversible``, otherwise
        ``False``.
    """
    return isinstance(obj, Reversible)
from collections.abc import Collection
@docwrap('immlib.is_acoll')
def is_acoll(obj):
    """Returns ``True`` if an object is a collection (a sized iterable
    container).

    ``is_acoll(obj)`` returns ``True`` if the given object `obj` is an instance
    of the abstract ``collections.abc.Collection`` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an ``Collection`` object is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``Collection``, otherwise
        ``False``.
    """
    return isinstance(obj, Collection)
from collections.abc import Sequence
@docwrap('immlib.is_aseq')
def is_aseq(obj):
    """Returns ``True`` if an object is a sequence, otherwise ``False``.

    ``is_aseq(obj)`` returns ``True`` if the given object `obj` is an instance
    of the abstract ``collections.abc.Sequence`` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an ``Sequence`` object is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``Sequence``, otherwise ``False``.
    """
    return isinstance(obj, Sequence)
from collections.abc import MutableSequence
@docwrap('immlib.is_amseq')
def is_amseq(obj):
    """Returns ``True`` if an object is a mutable sequence, otherwise
    ``False``.

    ``is_amseq(obj)`` returns ``True`` if the given object `obj` is an instance
    of the abstract ``collections.abc.MutableSequence`` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an ``MutableSequence`` object is to be
        assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``MutableSequence``, otherwise
        ``False``.
    """
    return isinstance(obj, MutableSequence)
from pcollections.abc import PersistentSequence
@docwrap('immlib.is_apseq')
def is_apseq(obj):
    """Returns ``True`` if an object is a persistent sequence, otherwise
    ``False``.

    ``is_apseq(obj)`` returns ``True`` if the given object `obj` is an instance
    of the abstract ``pcollections.abc.PersistentSequence`` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an ``PersistentSequence`` object is to be
        assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``PersistentSequence``, otherwise
        ``False``.
    """
    return isinstance(obj, PersistentSequence)
from collections.abc import ByteString
@docwrap('immlib.is_abytes')
def is_abytes(obj):
    """Returns ``True`` if an object is a byte-string, otherwise ``False``.

    ``is_abytes(obj)`` returns ``True`` if the given object `obj` is an
    instance of the abstract ``collections.abc.ByteString`` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an ``ByteString`` object is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``ByteString``, otherwise
        ``False``.
    """
    return isinstance(obj, ByteString)
@docwrap('immlib.is_bytes')
def is_bytes(obj):
    """Returns ``True`` if an object is a ``bytes`` object, otherwise
    ``False``.

    ``is_bytes(obj)`` returns ``True`` if the given object `obj` is an instance
    of the ``bytes`` type and returns ``False`` otherwise.

    Parameters
    ----------
    obj : object
        The object whose quality as an ``bytes`` object is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``bytes``, otherwise ``False``.
    """
    return isinstance(obj, bytes)
from collections.abc import Set
@docwrap('immlib.is_aset')
def is_aset(obj):
    """Returns ``True`` if an object is a set type, otherwise ``False``.

    ``is_aset(obj)`` returns ``True`` if the given object `obj` is an instance
    of the abstract ``collections.abc.Set`` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an ``Set`` object is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``Set``, otherwise ``False``.
    """
    return isinstance(obj, Set)
from collections.abc import MutableSet
@docwrap('immlib.is_amset')
def is_amset(obj):
    """Returns ``True`` if an object is a mutable set, otherwise ``False``.

    ``is_amset(obj)`` returns ``True`` if the given object `obj` is an instance
    of the abstract ``collections.abc.MutableSet`` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an ``MutableSet`` object is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``MutableSet``, otherwise
        ``False``.
    """
    return isinstance(obj, MutableSet)
from pcollections.abc import PersistentSet
@docwrap('immlib.is_apset')
def is_apset(obj):
    """Returns ``True`` if an object is a persistent set, otherwise ``False``.

    ``is_apset(obj)`` returns ``True`` if the given object `obj` is an instance
    of the abstract ``pcollections.abc.PersistentSet`` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an ``PersistentSet`` object is to be
        assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``PersistentSet``, otherwise
        ``False``.
    """
    return isinstance(obj, PersistentSet)
from collections.abc import Mapping
@docwrap('immlib.is_amap')
def is_amap(obj):
    """Returns ``True`` if an object is an abstract mapping, otherwise
    ``False``.

    ``is_amap(obj)`` returns ``True`` if the given object `obj` is an instance
    of the abstract ``collections.abc.Mapping`` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an ``Mapping`` object is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``Mapping``, otherwise ``False``.
    """
    return isinstance(obj, Mapping)
from collections.abc import MutableMapping
@docwrap('immlib.is_ammap')
def is_ammap(obj):
    """Returns ``True`` if an object is a mutable mapping, otherwise ``False``.

    ``is_ammap(obj)`` returns ``True`` if the given object ``obj`` is an
    instance of the abstract ``collections.abc.MutableMapping`` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an ``MutableMapping`` object is to be
        assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``MutableMapping``, otherwise
        ``False``.
    """
    return isinstance(obj, MutableMapping)
from pcollections.abc import PersistentMapping
@docwrap('immlib.is_apmap')
def is_apmap(obj):
    """Returns ``True`` if an object is a persistent mapping, otherwise
    ``False``.

    ``is_apmap(obj)`` returns ``True`` if the given object `obj` is an instance
    of the abstract ``pcollections.abc.PersistentMapping`` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an ``PersistentMapping`` object is to be
        assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``PersistentMapping``, otherwise
        ``False``.
    """
    return isinstance(obj, PersistentMapping)
from collections.abc import Hashable
@docwrap('immlib.is_ahashable')
def is_ahashable(obj):
    """Returns ``True`` if an object is a hashable object, otherwise ``False``.

    ``is_ahashable(obj)`` returns ``True`` if the given object `obj` is an
    instance of the abstract ``collections.abc.Hashable`` type. This differs
    from the ``can_hash`` function, which checks whehter calling ``hash`` on an
    object raises an exception.

    Parameters
    ----------
    obj : object
        The object whose quality as an ``Hashable`` object is to be assessed.

    Returns
    -------
    boolean
        ``True`` if `obj` is an instance of ``Hashable``, otherwise ``False``.

    See Also
    --------
    can_hash
    """
    return isinstance(obj, Hashable)


# Builtin Python Concrete Types ###############################################

@docwrap('immlib.is_list')
def is_list(obj):
    """Returns ``True`` if an object is a ``list`` object.

    ``is_list(obj)`` returns ``True`` if the given object `obj` is an instance
    of the ``list`` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an ``list`` object is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``list``, otherwise ``False``.
    """
    return isinstance(obj, list)
@docwrap('immlib.is_tuple')
def is_tuple(obj):
    """Returns ``True`` if an object is a ``tuple`` object.

    ``is_tuple(obj)`` returns ``True`` if the given object `obj` is an instance
    of the ``tuple`` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an ``tuple`` object is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``tuple``, otherwise ``False``.
    """
    return isinstance(obj, tuple)
from pcollections import plist
@docwrap('immlib.is_plist')
def is_plist(obj):
    """Returns ``True`` if an object is a persistent list object.

    ``is_plist(obj)`` returns ``True`` if the given object `obj` is an instance
    of the ``pcollections.plist`` type and ``False`` otherwise.

    Parameters
    ----------
    obj : object
        The object whose quality as a ``plist`` object is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``plist``, otherwise ``False``.
    """
    return isinstance(obj, plist)
from pcollections import tlist
@docwrap('immlib.is_tlist')
def is_tlist(obj):
    """Returns ``True`` if an object is a transient list object.

    ``is_tlist(obj)`` returns ``True`` if the given object `obj` is an instance
    of the ``pcollections.tlist`` type and ``False`` otherwise.

    Parameters
    ----------
    obj : object
        The object whose quality as a ``tlist`` object is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``tlist``, otherwise ``False``.
    """
    return isinstance(obj, tlist)
from pcollections import llist
@docwrap('immlib.is_llist')
def is_llist(obj):
    """Returns ``True`` if an object is a persistent lazy list object.

    ``is_llist(obj)`` returns ``True`` if the given object `obj` is an instance
    of the ``pcollections.llist`` type and ``False`` otherwise.

    Parameters
    ----------
    obj : object
        The object whose quality as a ``llist`` object is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``llist``, otherwise ``False``.
    """
    return isinstance(obj, llist)
@docwrap('immlib.is_set')
def is_set(obj):
    """Returns ``True`` if an object is a ``set`` object.

    ``is_set(obj)`` returns ``True`` if the given object `obj` is an instance
    of the ``set`` type. Note that this is not the same as ``is_aset`` which
    determines whether the object is of the ``collections.abc.Set`` abstract
    type.

    Parameters
    ----------
    obj : object
        The object whose quality as an ``set`` object is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``set``, otherwise ``False``.

    See Also
    --------
    is_aset, is_amset, is_apset, is_pset, is_tset, is_frozenset
    """
    return isinstance(obj, set)
@docwrap('immlib.is_frozenset')
def is_frozenset(obj):
    """Returns ``True`` if an object is a ``frozenset`` object.

    ``is_frozenset(obj)`` returns ``True`` if the given object `obj` is an
    instance of the ``frozenset`` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an ``frozenset`` object is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``frozenset``, otherwise ``False``.
    
    See Also
    --------
    is_set, is_aset, is_amset, is_apset, is_pset, is_tset
    """
    return isinstance(obj, frozenset)
from pcollections import pset
@docwrap('immlib.is_pset')
def is_pset(obj):
    """Returns ``True`` if an object is a persistent set object.

    ``is_pset(obj)`` returns ``True`` if the given object `obj` is an instance
    of the ``pcollections.pset`` type and ``False`` otherwise.

    Parameters
    ----------
    obj : object
        The object whose quality as a ``pset`` object is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``pset``, otherwise ``False``.
    
    See Also
    --------
    is_set, is_aset, is_amset, is_apset, is_tset, is_frozenset

    """
    return isinstance(obj, pset)
from pcollections import tset
@docwrap('immlib.is_tset')
def is_tset(obj):
    """Returns ``True`` if an object is a transient set object.

    ``is_tset(obj)`` returns ``True`` if the given object `obj` is an instance
    of the ``pcollections.tset`` type and ``False`` otherwise.

    Parameters
    ----------
    obj : object
        The object whose quality as a ``tset`` object is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``tset``, otherwise ``False``.
    """
    return isinstance(obj, tset)
@docwrap('immlib.is_dict')
def is_dict(obj):
    """Returns ``True`` if an object is a ``dict`` object.

    ``is_dict(obj)`` returns ``True`` if the given object `obj` is an instance
    of the ``dict`` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an ``dict`` object is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``dict``, otherwise ``False``.
    """
    return isinstance(obj, dict)
from collections import OrderedDict
@docwrap('immlib.is_odict')
def is_odict(obj):
    """Returns ``True`` if an object is an ``OrderedDict`` object.

    ``is_odict(obj)`` returns ``True`` if the given object `obj` is an instance
    of the ``collections.OrderedDict`` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an ``OrderedDict`` object is to be
        assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``OrderedDict``, otherwise
        ``False``.
    """
    return isinstance(obj, OrderedDict)
from collections import defaultdict
@docwrap('immlib.is_ddict')
def is_ddict(obj):
    """Returns ``True`` if an object is a ``defaultdict`` object.

    ``is_ddict(obj)`` returns ``True`` if the given object `obj` is an instance
    of the ``collections.defaultdict`` type.

    Parameters
    ----------
    obj : object
        The object whose quality as a ``defaultdict`` object is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``defaultdict``, otherwise
        ``False``.
    """
    return isinstance(obj, defaultdict)
from pcollections import pdict
@docwrap('immlib.is_pdict')
def is_pdict(obj):
    """Returns ``True`` if an object is a persistent dictionary object.

    ``is_pdict(obj)`` returns ``True`` if the given object `obj` is an instance
    of the ``pcollections.pdict`` type and ``False`` otherwise.

    .. Note:: The ``ldict`` type is a subtype of ``pdict``, so for
        ``is_pdict(ldict())`` returns ``True``.
    
    Parameters
    ----------
    obj : object
        The object whose quality as a ``pdict`` object is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``pdict``, otherwise ``False``.
    """
    return isinstance(obj, pdict)
from pcollections import tdict, tldict
@docwrap('immlib.is_tdict')
def is_tdict(obj):
    """Returns ``True`` if an object is a transient dictionary object.

    ``is_tdict(obj)`` returns ``True`` if the given object `obj` is an instance
    of the ``pcollections.tdict`` type and ``False`` otherwise.

    Parameters
    ----------
    obj : object
        The object whose quality as a ``tdict`` object is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``tdict``, otherwise ``False``.
    """
    return isinstance(obj, tdict)
from pcollections import ldict
@docwrap('immlib.is_ldict')
def is_ldict(obj):
    """Returns ``True`` if an object is a persistent lazy dictionary object.

    ``is_ldict(obj)`` returns ``True`` if the given object `obj` is an instance
    of the ``pcollections.ldict`` type and ``False`` otherwise.

    Parameters
    ----------
    obj : object
        The object whose quality as a ``ldict`` object is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``ldict``, otherwise ``False``.
    """
    return isinstance(obj, ldict)
@docwrap('immlib.hashsafe')
def hashsafe(obj):
    """Returns ``hash(obj)`` if `obj` is hashable, otherwise returns ``None``.

    This function attempts to hash an object and returns ``None`` when doing so
    raises a ``TypeError``.

    .. Note:: A fairly reliable test of whether an object is immutable or not
        in Python is whether it can be hashed.
        
    Parameters
    ----------
    obj : object
        The object to be hashed.

    Returns
    -------
    int or None
        If the object is hashable, returns the hashcode; otherwise, returns
        ``None``.

    See Also
    --------
    can_hash, is_ahashable
    """
    try:
        return hash(obj)
    except TypeError:
        return None
@docwrap('immlib.can_hash')
def can_hash(obj):
    """Returns ``True`` if `obj` is safe to hash and ``False`` otherwise.

    ``can_hash(obj)`` is equivalent to ``hashsafe(obj) is not None``. This
    differs from ``is_ahashable(obj)`` in that ``is_ahashable`` only checks
    whether `obj` is an instance of ``Hashable`` while ``hashsafe(obj)``
    attempts to hash `obj` and returns ``None`` when a ``TypeError`` is raised.

    .. Note:: A fairly reliable test of whether an object is immutable or not
        in Python is whether it can be hashed.

    See Also
    --------
    hashsafe, is_ahashable
    """
    return hashsafe(obj) is not None
@docwrap('immlib.itersafe')
def itersafe(obj):
    """Returns an iterator of the given object or ``None`` if it is not
    iterable.

    ``itersafe(obj)`` is equivalent to ``iter(obj)`` with the exception that,
    if `obj` is not iterable, it returns ``None`` instead of raising an
    exception.

    Parameters
    ----------
    obj : object
        The object to be iterated.

    Returns
    -------
    iterator or None
        If `obj` is iterable, returns ``iter(obj)``; otherwise, returns
        ``None``.

    See Also
    --------
    can_iter, is_aiterable
    """
    try:
        return iter(obj)
    except TypeError:
        return None
@docwrap('immlib.can_iter')
def can_iter(obj):
    """Returns ``True`` if `obj` is safe to iterate and ``False`` otherwise.

    ``can_iter(obj)`` is equivalent to ``itersafe(obj) is not None``. This
    differs from ``is_aiterable(obj)`` in that ``is_aiterable`` only checks
    whether `obj` is an instance of ``Iterable``; ``itersafe`` tries to run
    ``iter(obj)`` and returns ``None`` when a ``TypeError`` is raised.

    See Also
    --------
    itersafe, is_aiterable
    """
    return itersafe(obj) is not None
@docwrap('immlib.is_pcoll')
def is_pcoll(obj):
    """Detects if an object is a ``plist``, ``pset``, ``pdict``, ``llist`` or
    ``ldict``.

    ``is_pcoll(obj)`` returns ``True`` if the given object `obj` is an instance
    of the persistent collection types ``plist``, ``pset``, ``pdict``,
    ``llist``, or ``ldict``. Otherwise, ``False`` is returned.

    Note that this function tests against a specific set of concrete types;
    instances of objects whose types are subclasses of these types will be
    treated as instances of the base types; however other immutable types not
    defined in ``immlib`` or ``pcollections`` will not be recognized by this
    function.

    Parameters
    ----------
    obj : object
        The object whose quality as a persistent collection is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is a persistent collection and ``False`` otherwise.
    """
    return isinstance(obj, is_pcoll.types)
is_pcoll.types = (plist, pset, pdict, llist, ldict)
@docwrap('immlib.is_tcoll')
def is_tcoll(obj):
    """Returns ``True`` if an object is a transient ``tlist``, ``tset``, or
    ``tdict``.

    ``is_tcoll(obj)`` returns ``True`` if the given object `obj` is an instance
    of the ``tdict``, ``tset``, or ``tlist`` types, all of which are transient
    collections. Otherwise, ``False`` is returned.

    Parameters
    ----------
    obj : object
        The object whose quality as a transient collection is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is a ``tlist``, ``tset``, or ``tdict`` and ``False``
        otherwise.
    """
    return isinstance(obj, is_tcoll.types)
is_tcoll.types = (tlist, tset, tdict)
@docwrap('immlib.is_mcoll')
def is_mcoll(obj):
    """Returns ``True`` if an object is a mutable ``list``, ``set``, or
    ``dict``.

    ``is_mcoll(obj)`` returns ``True`` if the given object `obj` is an instance
    of the ``dict``, ``set``, or ``list`` types, all of which are mutable
    collections. Otherwise, ``False`` is returned.

    Parameters
    ----------
    obj : object
        The object whose quality as a mutable collection is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is a ``list``, ``set``, or ``dict`` and ``False``
        otherwise.
    """
    return isinstance(obj, is_mcoll.types)
is_mcoll.types = (list, set, dict)
@docwrap('immlib.to_pcoll')
def to_pcoll(obj):
    """Returns a persistent copy of `obj`.

    ``to_pcoll(obj)`` returns `obj` itself if `obj` is a persistent collection;
    otherwise, it returns a persistent copy of `obj`. If `obj` is not a
    collection that can be converted into a persistent collection, then an
    error is raised.

    Parameters
    ----------
    obj : collection
        An object that is to be converted into a persistent collection.

    Returns
    -------
    object
        A persistent version of `obj`.

    Raises
    ------
    TypeError
        If `obj` cannot be converted into a persistent collection.
    """
    if is_pcoll(obj):
        return obj
    if isinstance(obj, Sequence):
        return plist(obj)
    elif isinstance(obj, Set):
        return pset(obj)
    elif isinstance(obj, Mapping):
        return pdict(obj)
    else:
        raise TypeError(f"argument is not a recognized collection")
from pcollections.abc import (
    TransientSequence, TransientSet, TransientMapping)
@docwrap('immlib.to_tcoll')
def to_tcoll(obj, /, copy=True):
    """Returns a transient copy of `obj`.

    ``to_tcoll(obj)`` returns a copy of `obj` as a transient collection. If
    `obj` is not a collection that can be converted into a transient
    collection, then an error is raised.

    .. Note:: If `obj` is already a transient collection, then a copy of `obj`
        is returned.

    .. Note:: If ``to_tcoll`` is given a lazy dict (``pcollections.ldict``) or
        a lazi list (``pcollections.llist``), the resulting transient
        dictionary is made using ``obj.transient()`` and so respects the
        laziness of the elements.

    Parameters
    ----------
    obj : collection
        An object that is to be converted into a persistent collection.
    copy : boolean, optional
        If `obj` is already a transient collection, then a copy is made if and
        only if ``copy`` is ``True``; otherwise, `obj` is returned as-is when
        it is already a transient type. The default is ``True``.

    Returns
    -------
    object
        A transient version of `obj`. The returned value is always either a
        ``tlist``, ``tset``, or ``tdict`` object.

    Raises
    ------
    TypeError
        If `obj` cannot be converted into a transient collection.
    """
    if not copy and is_tcoll(obj):
        return obj
    if isinstance(obj, (PersistentSequence, PersistentSet, PersistentMapping)):
        return obj.transient()
    elif isinstance(obj, (TransientSequence, TransientSet, TransientMapping)):
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
@docwrap('immlib.to_mcoll')
def to_mcoll(obj, /, copy=True):
    """Returns a mutable copy of `obj`.

    ``to_mcoll(obj)`` returns a mutable copy of the given collection `obj`. If
    `obj` is already a mutable collection, then a duplicate is returned. If
    `obj` is not a collection that can be converted into a mutable collection,
    then an error is raised.

    A mutable collection, according to this function, is a Python ``list``,
    ``set``, or ``dict``, depending on the type of `obj`. When `obj` is a
    ``Sequence``, the result is a ``list``; when `obj` is a ``Set``, the result
    is a ``set``; and when `obj` is a ``Mapping``, the result is a ``dict``.

    Parameters
    ----------
    obj : collection
        An object that is to be converted into a persistent collection.
    copy : boolean, optional
        If `obj` is already a mutable collection, then a copy is made if and
        only if ``copy`` is ``True``; otherwise, `obj` is returned as-is when
        it is already a mutable type. The default is ``True``.

    Returns
    -------
    object
        A mutable version of `obj`; the return value's type will always be one
        of ``list``, ``set``, or ``dict``.

    Raises
    ------
    TypeError
        If `obj` cannot be converted into a mutable collection.
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
@docwrap('immlib.freezearray')
def freezearray(arr):
    """Freezes a NumPy array or SciPy sparse array in-place.

    ``freezearray(x)`` sets the ``'WRITEABLE'`` bit on the numpy array ``x`` or
    on ``x.data`` if ``x`` is a SciPy sparse array. If ``x`` is neither a NumPy
    array nor a SciPy sparse array, then a ``TypeError`` is raised. No value is
    returned.

    ``freezearray(q)`` is equivalent to ``freezearray(q.m)`` if ``q`` is a
    ``pint.Quantity`` object.

    .. Warning:: This function mutates its argument in-place.

    See Also
    --------
    frozenarray
    """
    if isinstance(arr, pint.Quantity):
        arr = arr.m
    if isinstance(arr, np.ndarray):
        arr.setflags(write=False)
    elif sps.issparse(arr):
        arr.data.setflags(write=False)
    else:
        raise TypeError(
            f"freezearray requires a numpy array or scipy sparse array,"
            f" but type {type(arr)} was given")
@docwrap('immlib.frozenarray')
def frozenarray(obj, /, dtype=None, *, copy=False, **kwargs):
    """Roughly equivalent to ``numpy.array`` but returns read-only arrays.

    ``frozenarray(obj)`` is equivalent to ``numpy.array(obj)`` with a small
    number of exceptions:
    
      - Primarily, the returned object is always a frozen array (i.e., an array
        with the ``'WRITEABLE'`` flag set to ``False``).
      - The default value of the ``copy`` option is ``False``, meaning that a
        copy of the array will only be made if required by the other parameters
        or if the array is not already read-only. If you wish to make an array
        read-only rather than obtaining a read-only copy of it, use the
        ``freezearray()`` function.
      - SciPy sparse arrays are also handled by setting the write flag on the
        ``obj.data`` member.
      - If ``obj`` is a ``pint.Quantity`` object, then an equivalent quantity
        with the magnitude made read-only is returned.

    If a PyTorch tensor is passed to ``frozenarray``, it will be converted into
    a frozen NumPy array; PyTorch tensors themselves cannot be frozen, however.

    See Also
    --------
    numpy.array : Create an array that is not frozen.
    freezearray : Convert an argument to a frozen array in-place.
    """
    if isinstance(obj, pint.Quantity):
        arr = frozenarray(obj.m, dtype=dtype, copy=copy, **kwargs)
        return obj if not copy and arr is obj.m else type(obj)(arr, obj.u)
    elif sps.issparse(obj):
        arr = frozenarray(obj.data, dtype=dtype, copy=copy, **kwargs)
        if not copy and obj.data is arr:
            return obj
        obj = obj.copy()
        obj.data = arr
        return obj
    elif isinstance(obj, np.ndarray):
        if obj.flags['WRITEABLE']:
            copy = True
        arr = np.array(obj, dtype=dtype, copy=copy, **kwargs)
        arr.setflags(write=False)
        return arr
    else:
        # PyTorch tensors might be sparse, so have to be treated specially.
        from ._numeric import torch, to_array
        if torch.is_tensor(obj):
            return frozenarray(to_array(obj), dtype=dtype, **kwargs)
        arr = np.array(obj, dtype=dtype, **kwargs)
        arr.setflags(write=False)
        return arr


# Mapping/Sequence Utilities ##################################################

@docwrap('immlib.get')
def get(d, k, /, *args, **kwargs):
    """Returns a value from either a mapping or a sequence.

    The ``get`` function is essentially a function version of the ``get``
    method that works for both ``Mapping`` and ``Sequence`` types (e.g.,
    ``dict``, ``list``, ``tuple``, and related types that implement their
    abstract bases).

    ``get(d, k)`` extracts element `k` from object `d` and returns it. If `k`
    is not a valid index of `d` (i.e., `k` is not a key of `d`, if `d` is a
    mapping, or is not an integer index of `d` if `d` is a sequence), then the
    optional value ``default`` is returned. If ``detault`` is not explicitly
    provided, then an error is raised. Note that if a non-integer key is
    provided for a sequence, this is treated as a missing index.

    .. Note:: The default value may be expressed as either a third positional
       argument or a named argument (``default``).

    Parameters
    ----------
    d : object
        The dict-like or list-like object from which an element is being
        extracted.
    k : object
        The key or index into `d` whose value is to be extracted.
    args
        The default value to be returned if an item is not found. The default
        value may be specified as a third positional argument.
    kwargs
        The default value to be returned if an item is not found. The default
        value may be specified as a named argument with the name ``"default"``.

    Returns
    -------
    object
        The object ``d[k]``, if the key `k` is found in the collection `d`.
        Otherwise, ``default`` is returned.
    
    Raises
    ------
    KeyError
        If the key or index `k` is not found in the collection `d` and no
        ``default`` option is given.
    """
    nargs = len(args)
    nkw = len(kwargs)
    if nargs + nkw > 1:
        raise TypeError(f"get takes 2 or 3 arguments; got {2 + nargs + nkw}")
    if nargs == 1:
        error = False
        default = args[0]
    elif nkw == 1:
        if "default" in kwargs:
            error = False
            default = kwargs.pop("default")
        else:
            k = repr(next(iter(kwargs.keys())))
            raise TypeError(f"get() got an unexpected keyword argument {k}")
    else:
        error = True
        default = None
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
@docwrap('immlib.nestget')
def nestget(d, /, *args, **kwargs):
    """Returns a value from a data structure of nested mappings and sequences.

    The ``nestget`` function is essentially a nested version of the ``get``
    function that works for both ``Mapping`` and ``Sequence`` types (e.g.,
    ``dict``, ``list``, ``tuple``, and related types that implement their
    abstract bases).

    ``nestget(data, k1, k2, k3...)`` extracts element ``k1`` from ``data`` then
    element ``k2`` from that value, then element ``k3`` from that value, etc.,
    until there are no more keys; the final value is returned. If any of the
    values are missing, then the optional value ``default`` is returned if it
    is provided and an error is raised if it is not. Note that the provided
    keys may be integer indices for list-like objects that may be included in
    the nesting. If a string key is given for a list-like container, then this
    is treated as a missing key, not an error.

    This function raises a ``KeyError` when a key or index is not found in the
    relevant container, but this behavior can be changed by passing the
    optional named parameter ``default``. If ``default`` is provided, then this
    value is returned if any keys are missing.

    Parameters
    ----------
    d : object
        The dict-like or list-like object from which an element is being
        extracted.
    args
        The list of keys and indices to be extracted.
    kwargs
        The default value to be returned if an item is not found can be
        specified using the named option ``default``. If this option is not
        provided, then an error is raised should the item not be found.

    Returns
    -------
    object
        The object found at the given nested position in the data structure
        `d`. If one of the provided keys does not exist in the associated
        sub-collection of `d`, then the ``default`` option is returned.

    Raises
    ------
    KeyError
        If the given sequence of keys cannot be found in the nested data
        structure and no ``default`` option is provided.
    """
    if "default" in kwargs:
        error = False
        default = kwargs.pop("default")
    else:
        error = True
        default = None
    if len(kwargs) > 0:
        k = repr(next(iter(kwargs.keys())))
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
@docwrap('immlib.lazyvalmap')
def lazyvalmap(f, d, /, *args, **kwargs):
    """Returns a dict object whose values are transformed by a function.

    ``lazyvalmap(f, d)`` returns a dict whose keys are the same as those of the
    given dict object and whose values, for each key ``k`` are ``f(d[k])``. All
    values are created lazily.

    ``lazyvalmap(f, d, *args, **kw)`` additionally passes the given arguments
    to the function `f`, such that in the resulting map, each key ``k`` is
    mapped to ``f(d[k], *args, **kw)``.

    Parameters
    ----------
    f : function
        The function used to create the values in the new dictionary.
    d : collections.abc.Mapping
        A mapping whose keys are to be preserved and remapped to a function of
        their values.
    args
        Additional positional arguments to pass to `f`.
    kwargs
        Additional named arguments to pass to `f`.
    
    Returns
    -------
    pcollections.ldict
        This function always returns a lazy dictionary object of type
        ``pcollections.ldict``.
    """
    t = tldict()
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
    return t.persistent()
@docwrap('immlib.valmap')
def valmap(f, d, /, *args, **kwargs):
    """Returns a dictionary object whose values are transformed by a function.

    ``valmap(f, d)`` returns a dict whose keys are the same as those of the
    given dict object and whose values, for each key ``k`` are ``f(d[k])``.

    ``valmap(f, d, *args, **kw)`` additionally passes the given arguments to
    the function `f`, such that in the resulting map, each key ``k`` is
    mapped to ``f(d[k], *args, **kw)``.

    Unlike ``lazyvalmap``, this function returns either a ``dict``, a
    ``pdict``, or an ``ldict`` depending on the input argument `d`. If `d` is a
    (lazy) ``ldict``, then an ``ldict`` is returned; if `d` is a ``pdict``, a
    ``pdict`` is returned, and otherwise, a ``dict`` is returnd.

    Parameters
    ----------
    f : function
        The function used to create the values in the new dictionary.
    d : collections.abc.Mapping
        A mapping whose keys are to be preserved and remapped to a function of
        their values.
    args
        Additional positional arguments to pass to `f`.
    kwargs
        Additional named arguments to pass to `f`.
    
    Returns
    -------
    collections.abc.Mapping object
        This function always returns a dictionary object whose type is either
        ``pcollections.pdict``, ``pcollections.ldict``, or ``dict``, depending
        on the type of `d`.
    """
    if is_ldict(d):
        return lazyvalmap(f, d, *args, **kwargs)
    elif is_pdict(d):
        t = tdict()
        for (k,v) in d.items():
            t[k] = f(v, *args, **kwargs)
        return t.persistent()
    else:
        return {k: f(v, *args, **kwargs) for (k,v) in d.items()}
@docwrap('immlib.lazykeymap')
def lazykeymap(f, d, /, *args, **kwargs):
    """Returns a object of type ``pcollections.ldict`` whose values are a
    function of the keys of the mapping `d`.

    ``keymap(f, d)`` returns a dict whose keys are the same as those of the
    given dict object and whose values, for each key ``k`` are ``f(k)``. If `d`
    is a sequence or iterable, then it is treated as a sequence of keys.

    ``keymap(f, d, *args, **kw)`` additionally passes the given arguments to
    the function `f`, such that in the resulting map, each key ``k`` is
    mapped to ``f(k, *args, **kw)``.

    Parameters
    ----------
    f : function
        The function used to create the values in the new dictionary.
    d : collections.abc.Mapping or iterable
        A mapping whose keys are to be preserved and remapped to a function of
        themselves. Alternatively, this may be an iterable of the keys instead
        of a dict with matching keys.
    args
        Additional positional arguments to pass to `f`.
    kwargs
        Additional named arguments to pass to `f`.
    
    Returns
    -------
    pcollections.ldict
        This function always returns an object of type ``ldict`` whose values
        are lazy.
    """
    if is_amap(d):
        keys = d.keys()
    else:
        keys = d
    t = tldict()
    for k in keys:
        t[k] = lazy(f, k, *args, **kwargs)
    return t.persistent()
@docwrap('immlib.keymap')
def keymap(f, d, /, *args, **kwargs):
    """Returns a dict object whose values are a function of a dict's keys.

    ``keymap(f, d)`` returns a dict whose keys are the same as those of the
    given dict object and whose values, for each key ``k`` are ``f(k)``.

    ``keymap(f, d, *args, **kw)`` additionally passes the given arguments to
    the function `f`, such that in the resulting map, each key ``k`` is mapped
    to ``f(k, *args, **kw)``.

    This function returns either a ``dict`` or a ``pdict``. If ``d`` is a
    ``pdict``, a ``pdict`` is returned, and otherwise, a ``dict`` is
    returnd. Unlike the ``valmap`` function, an ``ldict`` is never returned
    because the lazy values of such a dictionary are not accessed by
    ``keymap``; if a lazy dictionary is required, then the function
    ``lazykeymap`` should be used instead.

    Parameters
    ----------
    f : function
        The function used to create the values in the new dictionary.
    d : collections.abc.Mapping or iterable
        A mapping whose keys are to be preserved and remapped to a function of
        themselves. Alternatively, this may be an iterable of the keys instead
        of a dict with matching keys.
    args
        Additional positional arguments to pass to `f`.
    kwargs
        Additional named arguments to pass to `f`.
    
    Returns
    -------
    collections.abc.Mapping object
        This function always returns a dictionary object whose type is either
        ``pcollections.pdict`` or ``dict``, depending on the type of `d`.
    """
    if is_pdict(d):
        t = tdict()
        for k in d.keys():
            t[k] = f(k, *args, **kwargs)
        return t.persistent()
    elif is_amap(d):
        keys = d.keys()
    else:
        keys = d
    return {k: f(k, *args, **kwargs) for k in keys}
def _lazyitemmap_extract(f, d, k, *args, **kw):
    return f(k, d[k], *args, **kw)
@docwrap('immlib.lazyitemmap')
def lazyitemmap(f, d, /, *args, **kwargs):
    """Returns an ``ldict`` object whose values are a function of a dict's
    items.

    ``lazyitemmap(f, d)`` yields an ``ldict`` whose keys are the same as those
    of the given dict object and whose values, for each key ``k``, are lazily
    computed as ``f(k, d[k])``.

    ``itemmap(f, d, *args, **kw)`` additionally passes the given arguments to
    the function `f`, such that in the resulting map, each key ``k`` is
    mapped to ``f(k, d[k], *args, **kw)``.

    Parameters
    ----------
    f : function
        The function used to create the values in the new dictionary; it must
        accept two arguments (``f(k, d[k])``) plus any additional arguments
        provided in ``*args`` and ``**kwargs``.
    d : collections.abc.Mapping
        A mapping whose items are to be preserved and remapped to a function
        of their keys and values.
    args
        Additional positional arguments to pass to `f`.
    kwargs
        Additional named arguments to pass to `f`.
    
    Returns
    -------
    pcollections.ldict
        This function always returns a lazy dictionary object of type
        ``pcollections.ldict``.
    """
    t = tldict()
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
    return t.persistent()
@docwrap('immlib.itemmap')
def itemmap(f, d, /, *args, **kwargs):
    """Returns a dictionary object whose values are a function of a given
    dictionary's items.

    ``itemmap(f, d)`` returns a dict whose keys are the same as those of the
    given mapping object `d` and whose values, for each key ``k`` are ``f(k,
    d[k])``.

    ``itemmap(f, d, *args, **kw)`` additionally passes the given arguments to
    the function `f`, such that in the resulting map, each key ``k`` is
    mapped to ``f(k, d[k], *args, **kw)``.

    Unlike ``lazyitemmap``, this function returns either a ``dict``, a
    ``pdict``, or an ``ldict`` depending on the input argument `d`. If `d`
    is an ``ldict``, then an ``ldict`` is returned; if `d` is a ``pdict``, a
    ``pdict`` is returned, and otherwise, a ``dict`` is returnd.

    Parameters
    ----------
    f : function
        The function used to create the values in the new dictionary; it must
        accept two arguments (``f(k, d[k])``) plus any additional arguments
        provided in ``*args`` and ``**kwargs``.
    d : collections.abc.Mapping
        A mapping whose keys are to be preserved and remapped to a function of
        their values.
    args
        Additional positional arguments to pass to `f`.
    kwargs
        Additional named arguments to pass to `f`.
    
    Returns
    -------
    pcollections.pdict or pcollections.ldict or dict
        This function always returns a dictionary object whose type is either
        ``pcollections.pdict``, ``pcollections.ldict``, or ``dict``, depending
        on the type of `d`.
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
@docwrap('immlib.dictmap')
def dictmap(f, keys, /, *args, **kw):
    """Returns a dict with the given keys and the values ``map(f, keys)``.

    ``dictmap(f, keys)`` returns a dict object whose keys are the elements of
    ``iter(keys)`` and whose values are the elements of ``map(f, keys)``.

    ``dictmap(f, keys, *args, **kw)`` returns a dict object whose keys are the
    elements of ``iter(keys)`` and whose values are the elements of
    ``[f(k, *args, **kw) for k in iter(keys)]``.

    Parameters
    ----------
    f : function
        The function used to create the values in the new dictionary; it must
        accept one arguments (``f(k)``) plus any additional arguments provided
        in ``*args`` and ``**kwargs``.
    keys : iterable
        An iterable object whose values are to become the keys of the new
        dictionary.
    args
        Additional positional arguments to pass to `f`.
    kwargs
        Additional named arguments to pass to `f`.

    Returns
    -------
    dict
        A dictionary of the given `keys` with each key ``k`` mapped to
        ``f(k)``.
    """
    return {k: f(k, *args, **kw) for k in keys}
@docwrap('immlib.pdictmap')
def pdictmap(f, keys, /, *args, **kw):
    """Returns a ``pdict`` with the given keys and the values ``map(f, keys)``.

    ``pdictmap(f, keys)`` returns a ``pdict`` object whose keys are the
    elements of ``iter(keys)`` and whose values are the elements of
    ``map(f, keys)``.

    ``pdictmap(f, keys, *args, **kw)`` returns a dict object whose keys are
    the elements of ``iter(keys)`` and whose values are the elements of
    ``[f(k, *args, **kw) for k in iter(keys)]``.

    Parameters
    ----------
    f : function
        The function used to create the values in the new dictionary; it must
        accept one arguments (``f(k)``) plus any additional arguments provided
        in ``*args`` and ``**kwargs``.
    keys : iterable
        An iterable object whose values are to become the keys of the new
        dictionary.
    args
        Additional positional arguments to pass to `f`.
    kwargs
        Additional named arguments to pass to `f`.

    Returns
    -------
    pcollections.pdict
        A persistent dictionary of the given `keys` with each key ``k`` mapped
        to ``f(k)``.
    """
    t = tdict()
    for k in keys:
        t[k] = f(k, *args, **kw)
    return t.persistent()
@docwrap('immlib.ldictmap')
def ldictmap(f, keys, *args, **kw):
    """Returns a lazy dictionary with the given keys and the values
    ``map(f, keys)``.

    ``lazydictmap(f, keys)`` returns a ``pcollections.ldict`` object whose keys
    are the elements of ``iter(keys)`` and whose values are the elements of
    ``map(f, keys)``. All values are lazy.

    ``lazydictmap(f, keys, *args, **kw)`` returns a `pcollections.ldict` object
    whose keys are the elements of ``iter(keys)`` and whose values are the
    elements of ``[f(k, *args, **kw) for k in iter(keys)]``, lazily calculated.

    Parameters
    ----------
    f : function
        The function used to create the values in the new dictionary; it must
        accept one arguments (``f(k)``) plus any additional arguments provided
        in ``*args`` and ``**kwargs``.
    keys : iterable
        An iterable object whose values are to become the keys of the new
        dictionary.
    args
        Additional positional arguments to pass to `f`.
    kwargs
        Additional named arguments to pass to `f`.

    Returns
    -------
    pcollections.ldict
        A persistent lazy dictionary of the given `keys` with each key ``k``
        mapped to ``f(k)``.
    """
    t = tldict()
    for k in keys:
        t[k] = lazy(f, k, *args, **kw)
    return t.persistent()
@docwrap('immlib.merge')
def merge(*args, **kwargs):
    '''Merges dict-like objects left-to-right. See also ``rmerge``.

    ``merge(...)`` collapses all arguments, which must be ``Mapping`` objects
    of some kind (``dict``, ``pdict``, ``ldict``, or a similar type), into a
    single mapping from left-to-right (i.e., with values in dictionaries to the
    right in the argument list overwriting values to the left in the argument
    list). The mapping that is returned depends on the inputs: if any of the
    input mappings are ``ldict`` objects, then an ``ldict`` is returned (and
    the laziness of arguments is respected); otherwise, a ``pdict`` object is
    retuend.

    Named arguments may be passed after the dictionaries; these are
    collectively considered equivalent to one additional dictionary argument to
    the right of the positional mapping arguments.

    Parameters
    ----------
    args
        A sequence of ``collections.abc.Mapping`` objects such as ``dict``
        objects.
    kwargs
        Additional key-value pairs that are merged into the result last.

    Returns
    -------
    pcollections.pdict or pcollections.ldict
        A dictionary that represents the merger of all given dictionaries and
        key-value pairs. If any of the arguments are lazy dictionaries
        (``pcollections.ldict``) then the return value is also lazy in order to
        respect the laziness of the arguments.

    See Also
    --------
    rmerge : Merges dictionaries from right to left.
    '''
    if len(args) == 0:
        return pdict(kwargs)
    # Make the initial dictionary.
    res = args[0]
    lazy = is_ldict(res)
    res = tdict(holdlazy(res) if lazy else res)
    for d in args[1:]:
        if is_ldict(d):
            lazy = True
            res.update(holdlazy(d))
        else:
            res.update(d)
    res.update(kwargs)
    if not lazy:
        from pcollections import lazy as _lazy
        lazy = any(isinstance(u, _lazy) for u in kwargs.values())
    return ldict(res) if lazy else pdict(res)
def rmerge(*args, **kwargs):
    '''Merges dictionary objects right-to-left. See also ``merge``.

    ``rmerge(...)`` collapses all arguments, which must be python ``Mapping``
    objects of some kind, into a single mapping from right-to-left. The mapping
    that is returned depends on the inputs: if any of the input mappings are
    lazydict objects, then a lazydict is returned (and the laziness of
    arguments is respected); otherwise, a frozendict object is retuend.

    Named arguments may be passed after the dictionaries; these are
    collectively considered equivalent to one additional dictionary argument to
    the right of the positional mapping arguments.

    .. Note:: The ``rmerge`` function is identical to the ``merge`` function
        but with reversed arguments. In other words, ``merge(*args, **kw)`` is
        equivalent to ``rmerge(kw, **reversed(args))``.

    Parameters
    ----------
    args
        A sequence of ``collections.abc.Mapping`` objects such as ``dict``
        objectss.
    kwargs
        Additional key-value pairs that are merged into the result first.

    Returns
    -------
    pcollections.pdict or pcollections.ldict
        A dictionary that represents the merger of all given dictionaries and
        key-value pairs. If any of the arguments are lazy dictionaries
        (``pcollections.ldict``) then the return value is also lazy in order to
        respect the laziness of the arguments.

    See Also
    --------
    merge : Merges dictionaries from left to right.
    '''
    from pcollections import lazy as _lazy
    if len(args) == 0:
        return pdict(kwargs)
    # Make the initial dictionary.
    res = tdict(kwargs)
    lazy = any(isinstance(v, _lazy) for v in kwargs.values())
    for d in reversed(args):
        if is_ldict(d):
            lazy = True
            res.update(holdlazy(d))
        else:
            res.update(d)
    return ldict(res) if lazy else pdict(res)
@docwrap('immlib.assoc')
def assoc(d, /, *args, **kwargs):
    """Returns a copy of the given dictionary with additional key-value pairs.

    ``assoc(d, key, val)`` returns a copy of the dictionary `d` with the given
    key-value pair associated in the new copy. The return value is always the
    same type as the argument `d` but is always an updated copy. The argument
    `d` is never mutated.

    ``assoc(d, key1, val1, key2, val2 ...)`` associates all the given keys to
    the given values in the returned copy.

    ``assoc(d, key1=val1, key2=val2 ...)`` uses the keyword arguments as the
    arguments that are to be associated. These may be mixed with positional
    key-value pairs.

    ``assoc(d)`` returns a copy of `d`.

    Parameters
    ----------
    d : dict-like
        A dictionary that is to be copied and updated with the following
        arguments.
    args
        Sequential pairs of keys and values (i.e., ``len(args)`` must be even)
        that should be updated in the returned dictionary.
    kwargs
        Additional key-value pairs to be updated in the returned dictionary.

    Returns
    -------
    dict-like
        A copy of `d` with updated keys and values.
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
        for (k,v) in kwargs.items():
            d[k] = v
    elif is_apmap(d):
        nels = len(ks) + len(kwargs)
        if nels > 1:
            d = d.transient()
            for (k,v) in zip(ks,vs):
                d[k] = v
            for (k,v) in kwargs.items():
                d[k] = v
            d = d.persistent()
        else:
            for (k,v) in zip(ks,vs):
                d = d.set(k, v)
            for (k,v) in kwargs.items():
                d = d.set(k, v)
    else:
        raise TypeError(f"cannot assoc to type {type(d)}")
    return d
@docwrap('immlib.dissoc')
def dissoc(d, /, *args):
    """Returns a copy of the given dictionary with certain keys removed.

    ``dissoc(d, key)`` returns a copy of the dictionary `d` with the given
    ``key`` disssociated in the new copy. The return value is always the same
    type as the argument `d`.

    ``dissoc(d, key1, key2 ...)`` dissociates all the given keys from their
    values in the returned copy.

    ``dissoc(d)`` returns a copy of `d`.

    Parameters
    ----------
    d : dict-like
        A dictionary that is to be copied and updated according to the
        following arguments.
    args
        Keys that should be removed from the copy of `d` that is returned.

    Returns
    -------
    dict-like
        A copy of `d` with the given keys removed.
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
from pcollections import unlazy
def _lambdadict_call(data, fn):
    spec = getfullargspec(fn)
    dflts = spec.defaults or pdict()
    args = []
    kwargs = {}
    pos = True
    for k in spec.args:
        if k in data:
            v = unlazy(data[k])
            if pos:
                args.append(v)
            else:
                kwargs[k] = v
        else:
            pos = False
            if k in dflts:
                kwargs[k] = dflts[k]
    for k in spec.kwonlyargs:
        if k in data:
            kwargs[k] = unlazy(data[k])
        else:
            if k in dflts:
                kwargs[k] = dflts[k]
    return fn(*args, **kwargs)
def lambdadict(*args, **kwargs):
    """Builds and returns a ``ldict`` with lambda functions calculated lazily.

    ``lambdadict(args...)`` is equivalent to ``merge(args...)`` except that
    always returns an object of type ``pcollections.ldict`` and that any lambda
    function in the values provided by the merged arguments is made into a lazy
    partial function whose inputs come from the lambda-function variable names
    in the same resulting ``ldict``.

    .. Warning:: This function will gladly return an ``ldict`` that
        encapsulates an infinite loop if you are not careful. For example, the
        following lambdadict will infinitely loop when either key is requested:
        ``ld = lambdadict(a=lambda b:b, b=lambda a:a)``.
    
    Examples
    --------
    >>> d = lambdadict(a=1, b=2, c=lambda a,b: a + b)
    >>> d.is_lazy('c')
    True
    
    >>> d.is_ready('c')
    False
    
    >>> d['c']
    3
    
    >>> d
    {|'a': 1, 'b': 2, 'c': 3|}
    """
    d = merge(*args, **kwargs)
    finals = d.transient()
    if isinstance(d, ldict):
        d = d.as_pdict()
    for (k,v) in d.items():
        if isinstance(v, LambdaType):
            finals[k] = lazy(_lambdadict_call, finals, v)
        else:
            finals[k] = v
    return ldict(finals)


# Argument Utilities ##########################################################

from collections import namedtuple
argstuple = namedtuple('argstuple', ('args', 'kwargs'))
class args(argstuple):
    """An object type that represents a set of function arguments.

    ``args(x1, x2 ... k1=v1, k2=v2 ...)`` yields an ``args`` object that
    represents the positional arguments ``x1, x2 ...`` and the named arguments
    ``k1=v1``, ``k2=v2``, etc.

    If ``a`` is an instance of ``args`` and ``f`` is a function, then the
    arguments in ``a`` can be applied to ``f`` using either of the following
    methods:
    
     - ``f @ a``
     - ``a.passto(f)``

    Note that if ``f`` is an object that defines the ``__matmul__`` method,
    then the former syntax will call that method instead of the ``__rmatmul__``
    method of the ``args`` object ``a`` and thus won't work.
    """
    def __new__(cls, *args, **kwargs):
        return argstuple.__new__(cls, args, kwargs)
    def __rmatmul__(self, fn):
        return fn(*self.args, **self.kwargs)
    def passto(self, fn):
        return fn(*self.args, **self.kwargs)
    def copy(self, args=None, kwargs=None):
        """Returns a copy of the current ``args``, potentially with updates."""
        if args is None:
            args = self.args
        if kwargs is None:
            kwargs = self.kwargs
        if args is self.args and kwargs is self.kwargs:
            return self
        return argstuple.__new__(type(self), args, kwargs)
@docwrap('immlib.argfilter')
def argfilter(fn=None, /, **kwargs):
    """A decorator that creates decorators that filter function arguments.

    A function decorated with ``@argfilter`` is turned into a an argument
    filter function, which itself can be used to decorate functions whose
    arguments need to be filtered.

    In the definition of the filter function, the names of arguments must match
    those of the arguments they will be filtering on other functions. The
    filter function must return a tuple of the filtered values in the order
    they are defined in the function's argument list. The arguments may be
    given in any order, but a ``*`` in the arguments list indicates that any of
    the arguments following the ``*`` are not themselves being filtered and
    thus will not be returned from the filter function.

    When a filter function is used to decorate another function, the decorator
    can optionally be given named arguments where the name corresponds to one
    of the arguments to the origional filter definition and the value
    corresponds to the name that is used for this parameter in the decorated
    functions. In this way, the parameter names don't have to match exactly
    those of the filter function and can instead be specified in the
    decoration.

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

    >>> @fix_angle(angle='theta')
    ... def sin_halfangle(theta, unit='radians'):
    ...     return np.sin(theta / 2)

    >>> sin_halfangle([0, 360], 'degrees')
        array([0., 0.])
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


# unitregistry ################################################################

# We put the unitregistry here and not in the quantity namespace because we
# need it both for quantity and numeric and it causes a circular import if
# placed in the quantity file.
@docwrap('immlib.unitregistry')
def unitregistry(obj, /, *args):
    """Returns the ``pint.UnitRegistry`` object for the given unit or quantity.

    ``unitregistry(u)`` for a ``pint.Unit`` object ``u`` returns the
    ``pint.UnitRegistry`` object in which ``u`` is registered.

    ``unitregistry(q)`` for a ``pint.Quantity`` object ``q`` returns the
    ``pint.UnitRegistry`` object in which ``q`` is registered.

    ``unitregistry(ureg)`` returns ``ureg`` is ``ureg`` is itself a
    ``pint.UnitRegistry`` object.

    ``unitregistry(Ellipsis)`` returns ``immlib.units``, the default unit
    registry for ``immlib``.

    ``unitregistry(x)`` raises a ``TypeError`` for any other type of object.

    ``unitregistry(x, default)`` returns ``unitregistry(x)`` unless a
    ``TypeError`` would be raised, in which case it returns the given
    ``default`` value. If ``default`` is ``Ellipsis``, then ``immlib.units`` is
    used as the default.

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


# Caching #####################################################################

@docwrap('immlib.util.to_pathcache')
def to_pathcache(obj):
    """Returns a ``joblib.Memory`` object that corresponds to the given path
    object.

    ``to_pathcache(obj)`` converts the given object `obj` into a
    ``joblib.Memory`` cache manager. The object may be any of the following:
    
     - a ``joblib.Memory`` object;
     - a filename or pathlib object pointing to a directory; or
     - a tuple containing a filename or pathlib object followed by a dict-like
       object of options to ``joblib.Memory``.

    If the `obj` is ``None``, then ``None`` is returned. However, a
    ``joblib.Memory`` object whose location parameter is ``None`` can be
    created by using the object ``(None, opts)`` where ``opts`` may be ``None``
    or an empty dictionary.

    The ``joblib.Memory`` constructor takes certain arguments; this function
    makes one change to those arguments: the ``verbose`` option is by default 0
    when filtered through this function, meaning that no output will be printed
    unless a ``verbose`` argument of greater than 0 is explicitly given.

    See Also
    --------
    joblib.Memory, to_lrucache
    """
    # If we have been given a Memory object, just return it; otherwise, we
    # check to parse the object into path or path + options.
    if isinstance(obj, Memory):
        return obj
    elif is_tuple(obj):
        n = len(obj)
        if n == 1:
            (obj,opts) = (obj[0], {})
        elif n == 2:
            (obj,opts) = obj
        else:
            raise ValueError("only 1- or 2-tuples can become pathcaches")
        if opts is None:
            opts = {}
    elif isinstance(obj, args):
        (obj,opts) = (obj.args, obj.kwargs)
        if len(obj) == 1:
            obj = obj[0]
        else:
            raise TypeError(
                f"to_pathcache() takes exactly one argument ({len(obj)} given")
    else:
        opts = {}
    # We change the default argument of verbose into 0 in this function because
    # we don't want unintentional logging.
    if 'verbose' not in opts:
        opts['verbose'] = 0
    # Whether there were or were not any options, then we now have either a
    # string or pathlib path that we want to pass to the memory constructor.
    if isinstance(obj, Path) or isinstance(obj, str) or obj is None:
        return Memory(obj, **opts)
    else:
        raise TypeError(
            f"to_pathcache: arg must be path, str, or None; not {type(obj)}")
@docwrap('immlib.util.to_lrucache')
def to_lrucache(obj):
    """Returns an ``lru_cache`` function appropriate for the given object.

    ``to_lrucache(obj)`` converts the given object `obj` into either
    ``None``, the ``lru_cache`` function, or a function returned by
    ``lru_cache``. The object may be any of the following:
    
     - ``lru_cache`` itself, in which case it is just returned;
     - ``None`` or 0, indicating that no caching should be used (``None`` is
        returned in these cases);
     - ``inf``, indicating that an infinite cache should be returned; or
     - a positive integer indicating the number of most recently used items
       to keep in the cache.

    See Also
    --------
    functools.lru_cache
    """
    from ._numeric import (is_number, is_integer)
    if obj is lru_cache:
        return obj
    elif obj is None:
        return None
    elif is_number(obj):
        if obj == 0:
            return None
        elif obj == np.inf:
            return lru_cache(maxsize=None)
        elif not is_integer(obj):
            raise TypeError("to_lrucache size must be an int")
        elif obj < 1:
            raise ValueError("to_lrucache size must be > 0")
        else:
            return lru_cache(maxsize=int(obj))
    else:
        raise TypeError(f"bad type for to_lrucache: {type(obj)}")


# Other #######################################################################
    
def identfn(x):
    "The identify function; ``identfn(x)`` returns `x`."
    return x
