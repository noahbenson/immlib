# -*- Coding: utf-8 -*-
###############################################################################
# pimms/types/_core.py

"""The pimms subpackage containing various utility types.

The utility types included in pimms are:
 * `MetaObject` is a `planobject` type that implements metadata via the
   value `metadata` (a lazy dictionary) and the `withmeta` and `dropmeta`
   methods.
 * `PropTable` is a lazy table similar to a pandas `DataFrame` but built around
   the abilities to lazily load properties (columns) and to annotating multiple
   values for each property along arbitrary dimensions such as time or depth.
"""


# Dependencies ################################################################

import math
import operator as op
from functools import partial
from warnings import warn
from collections import namedtuple
from threading import Lock
LockType = type(Lock())

import numpy as np
import scipy as sp
from pcollections import *
from numpy.lib.mixins import NDArrayOperatorsMixin

from ..doc import docwrap
from ..util import (
    freezearray,
    frozenarray,
    is_integer,
    is_tuple,
    is_amap,
    is_array,
    is_realdata,
    is_tensor,
    is_array,
    merge,
    to_array)
from ..workflow import *


# MetaObject ##################################################################

class MetaObject(planobject):
    """Base planobject type for objects that keep track of metadata.

    Parameters
    ----------
    metadata : None or Mapping, optional
        The dictionary of metadata that is to be attached to the object. If the
        argument ``None`` is provided (the default), then the empty dictionary
        is used.

    Attributes
    ----------
    metadata : ldict
        A lazy dictionary of the metadata tracked by the object.
    """
    def __init__(self, metadata=None):
        if metadata is None:
            metadata = ldict.empty
        self.metadata = metadata
    @calc('metadata', lazy=False)
    def filter_metadata(metadata):
        return ldict.empty if metadata is None else ldict(metadata)
    def withmeta(self, *args, **kwargs):
        """Return a duplicate object with updated metadata.

        The arguments and keyword arguments to ``withmeta`` are merged,
        left-to-right, into the current metadata; this new dictionary is used
        as the metadata parameter of the new object.
        """
        new_metadata = merge(self.metadata, *args, **kwargs)
        return self.set_metadata(new_metadata)
    def dropmeta(self, *args):
        """Returns a duplicate object with given metadata keys cleared.

        The arguments must be keys, which are dropped from the metadata of
        the duplicate object.
        """
        md = self.metadata
        for k in args:
            md = md.drop(k)
        return self.set_metadata(md)
    def set_metadata(self, md):
        """Returns a duplicate object with the given metadata dictionary.

        The argument must be a dict-like object.
        """
        if md is self.metadata:
            return self
        return self.copy(metadata=md)
    def clear_metadata(self):
        """Returns a duplicate object with its metadata dictionary cleared."""
        if len(self.metadata) == 0:
            return self
        return self.copy(metadata=None)


# Immutable ###################################################################

class ImmutableType(type):
    """A meta-class for types that are immutable.

    When this metaclass is used in a class, objects of the type become
    immutable immediately after the ``__init__`` method is run. Such types
    should not overload the ``__new__`` classmethod and instead should see to
    their initialization in ``__init__`` as usual. Once the ``__init__`` method
    has finished, the ``__setattr__``, ``__delattr__``, ``__setitem__``, and
    ``__delitem__`` methods will raise a ``TypeError``.
    """
    class ImmutableBase:
        "The base class of all immlib immutable classes."
        __slots__ = ('__init_status',)
        def __setattr__(self, k, v):
            if self.__init_status:
                raise TypeError(f"{type(self)} is immutable")
            else:
                return object.__setattr__(self, k, v)
        def __delattr__(self, k):
            if self.__init_status:
                raise TypeError(f"{type(self)} is immutable")
            else:
                return object.__delattr__(self, k)
        def __init_wrap(self, *args, **kw):
            # Find the correct __init__ function to run:
            initfn = next(
                filter(
                    None,
                    (getattr(c, f'_{c.__name__}__init__', None)
                     for c in type(self).__mro__)),
                None)
            if initfn:
                initfn(self, *args, **kw)
            # Note that we have now initialized everything.
            self.__init_status = True
        def __new__(cls, *args, **kw):
            self = object.__new__(cls)
            object.__setattr__(self, '_ImmutableBase__init_status', False)
            return self
    def __new__(cls, name, bases, attrs, **kwargs):
        init_orig = attrs.get('__init__')
        if init_orig:
            attrs[f'_{name}__init__'] = init_orig
        base = ImmutableType.ImmutableBase
        attrs['__init__'] = base._ImmutableBase__init_wrap
        if base not in bases:
            bases = bases + (base,)
        return type.__new__(cls, name, bases, attrs, **kwargs)
class Immutable(ImmutableType.ImmutableBase, metaclass=ImmutableType):
    """A type that becomes immutable immediately after initialization.

    Any class that inherits from ``Immutable`` should implement an ``__init__``
    method, within which it is allowed to change the attributes of the ``self``
    object normally. After the ``__init__`` method terminates, the object
    becomes read-only and can no longer be updated.
    """
    __slots__ = ()


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
    def __new__(cls, array, freeze=True):
        if not freeze and is_array(array, frozen=True):
            freeze = True
        else:
            array = to_array(array, frozen=freeze)
        self = object.__new__(cls)
        object.__setattr__(self, '_flatdata', Lock())
        object.__setattr__(self, 'array', array)
        return self
    # Public Methods -----------------------------------------------------------
    @docwrap(indent=8)
    def find(self, ids, *, ravel=False, **kw):
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
            raised.

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
            else:
                ins[bad] = default
        if not ravel:
            if anymissing:
                unrav = np.unravel_index(ins[ok], self.array.shape)
                if not is_tuple(default):
                    default = (default,) * np.size(ins)
                ins = tuple(np.empty_like(ins) for u in unrav)
                for (u,r,d) in zip(ins, unrav, default):
                    u[bad] = d
                    u[ok] = r
            else:
                ins = np.unravel_index(ins, self.array.shape)
        return ins
    @property
    def flatdata(self):
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
