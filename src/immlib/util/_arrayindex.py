# -*- coding: utf-8 -*-
###############################################################################
# immlib/util/_arrayindex.py

"""The `ArrayIndex` type: an index of the (unique) elements of an array.

`ArrayIndex` was originally defined in the `immlib.types` subpackage; it lives
here because it is a general-purpose utility that depends only on the rest of
`immlib.util` (and NumPy), and because nothing in `immlib` used the other types
that subpackage defined.
"""


# Dependencies ################################################################

from __future__ import annotations

from collections import namedtuple
from threading import Lock
from typing import Any

import numpy as np
from docshare import docwrap

from ._core import (is_tuple, freezearray)
from ._numeric import (is_array, to_array)


# Globals #####################################################################

LockType = type(Lock())


# ArrayIndex ##################################################################

ArrayIndexFlatData = namedtuple(
    'ArrayIndexFlatData',
    ['ident', 'index'])
ArrayIndexFlatData.__doc__ = \
    """Flattened identity and index data used by ArrayIndex to search for IDs.

    Attributes
    ----------
    ident : read-only numpy array
        The sorted and flattened identities represented in the original array.
    index : read-only numpy array
        The argsort of the flattened original array object.
    """
class ArrayIndex:
    """A type that indexes the elements of an array for easy searching.

    The ``ArrayIndex`` class is a class that stores a (typically read-only)
    numpy array whose elements must all be unique and that creates an index of
    that array's elements. ``ArrayIndex`` objects primarily support a a
    ``find`` method that can be used to look up object indices.

    ``ArrayIndex`` objects require that the arrays they are given contain
    unique objects that are sortable and hashable.

    Examples
    --------
    >>> from immlib import ArrayIndex
    >>> labels = [['r1c1', 'r1c2'], ['r2c1', 'r2c2']]
    >>> index = ArrayIndex(labels)
    >>> index.find('r1c1')
        (0, 0)
    >>> index.find(['r2c1', 'r1c2'])
        (array([1, 0]), array([0, 1]))
    >>> index.find(['r2c2', 'r2c1', 'r1c2'], ravel=True)
        array([3, 1, 2])
    """
    # Class Methods -----------------------------------------------------------
    @classmethod
    def _make_flatdata(cls, array):
        flatins = np.argsort(array.flat)
        flatids = array.flat[flatins]
        freezearray(flatins)
        freezearray(flatids)
        return ArrayIndexFlatData(flatids, flatins)
    # Construction ------------------------------------------------------------
    __slots__ = ('array', '_flatdata')
    array: Any
    _flatdata: Any
    def __new__(cls, array: Any, freeze: bool = True) -> ArrayIndex:
        if not freeze and is_array(array, frozen=True):
            freeze = True
        else:
            array = to_array(array, frozen=freeze)
        self = object.__new__(cls)
        object.__setattr__(self, '_flatdata', Lock())
        object.__setattr__(self, 'array', array)
        return self
    # Public Methods -----------------------------------------------------------
    # `default` is documented although the signature does not name it: it is
    # taken from **kw, so that "not given" can be told from "given as None".
    @docwrap(format='numpy', extraparam='default')
    def find(self, ids: Any, *, ravel: bool = False, **kw: Any) -> Any:
        """Finds and returns the indices of the given identities.

        ``index.find(id)`` returns the index, in the original array on which
        ``index`` is based, of the identity ``id``. If ``id`` is not in the
        original array, then a ``KeyError`` is raised.

        Parameters
        ----------
        ids : array-like
            The identity or identities to look up in the index.
        ravel : boolean, optional
            Whether the return value should be an array representing a raveled
            index into the flattened version of the original indexed array
            (``True``) or a tuple representing an unraveled multi-index into
            the original indexed array (``False``). The default is ``False``.
        default : object, optional
            If `default` is not given, then an error is raised when an identity
            is not found. If `default` is given, however, the default value is
            inserted into the place of any missing indices and no error is
            raised. When `ravel` is ``False``, `default` may instead be a tuple
            with one element per dimension of the indexed array, in which case
            each element is the default for the corresponding coordinate.

        Returns
        -------
        indices
            The indices, into the original indexed array, of the given
            identities, `ids`.
        """
        if len(kw) == 1:
            try:
                default = kw.pop('default')
            except KeyError:
                err = TypeError(
                    f'unrecognized option for find: {next(iter(kw.keys()))}')
                raise err from None
            error = False
        else:
            default = None
            error = True
        if len(kw) > 0:
            k = next(iter(kw.keys()))
            raise TypeError(f"'{k}' is an invalid keyword argument for find()")
        ids = to_array(ids)
        (flatids, flatins) = self.flatdata
        # flatids is the ids in sorded order; flatins is the argsort of the
        # original argsort--how to put the sorted ids back in canonical
        # order; flatarg is the argsort itself.
        ii = np.searchsorted(flatids, ids)
        try:
            ins = flatins[ii]
            ok = flatids[ii] == ids
        except IndexError:
            ok = np.asarray((ii < len(flatids)) & (ii >= 0))
            ok[ok] &= flatids[ii[ok]] == ids[ok]
            ins = np.empty_like(ii, dtype=flatins.dtype)
            ins[ok] = flatins[ii[ok]]
        bad = ~ok
        anymissing = np.any(bad)
        # If some were not found, we might need to raise an error.
        if anymissing:
            if error:
                raise KeyError(ids[bad].flat[0])
            elif ravel or not is_tuple(default):
                ins[bad] = default
            # A per-dimension default (a tuple) cannot be written into
            # `ins`, which is a flat index; it is applied below, after the
            # unravelling, which rewrites these entries anyway.
        if not ravel:
            if anymissing:
                unrav = np.unravel_index(ins[ok], self.array.shape)
                if is_tuple(default):
                    if len(default) != len(unrav):
                        raise ValueError(
                            f"find: default has {len(default)} elements but"
                            f" the indexed array has {len(unrav)} dimensions")
                else:
                    # One default per dimension of the indexed array--not
                    # per identity looked up, which is a different number
                    # except by coincidence.
                    default = (default,) * len(unrav)
                ins = tuple(np.empty_like(ins) for u in unrav)
                for (u,r,d) in zip(ins, unrav, default):
                    u[bad] = d
                    u[ok] = r
            else:
                ins = np.unravel_index(ins, self.array.shape)
        return ins
    @property
    def flatdata(self) -> Any:
        """Returns a named tuple containing the flattened data used by the
        ``ArrayIndex`` type to lookup identities.

        ``index.flatdata`` returns a named 2-tuple with keys ``ident`` and
        ``index``. The ``ident`` element is a read-only numpy array containing
        the sorted and flattened identities represented in the original
        array. The ``index`` element is a read-only numpy array containing the
        argsort of the flattened original array object.
        """
        flatdata = self._flatdata
        if isinstance(flatdata, LockType):
            with flatdata:
                # Make sure that once we've acquired the lock we still need to
                # calculate the flatdata (i.e., we didn't check then acquire
                # after another thread ran).
                lock = flatdata
                flatdata = self._flatdata
                if flatdata is lock:
                    flatdata = self._make_flatdata(self.array)
                    object.__setattr__(self, '_flatdata', flatdata)
        return flatdata
    # Disabled Methods --------------------------------------------------------
    def __setattr__(self, k, v):
        raise TypeError(f"{type(self)} is immutable")
    def __delattr__(self, k):
        raise TypeError(f"{type(self)} is immutable")
    def __setitem__(self, k, v):
        raise TypeError(f"{type(self)} is immutable")
    def __delitem__(self, k):
        raise TypeError(f"{type(self)} is immutable")
