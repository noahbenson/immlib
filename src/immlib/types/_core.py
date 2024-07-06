# -*- Coding: utf-8 -*-
################################################################################
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


# Dependencies #################################################################

import math
import operator as op
from functools import partial

import numpy as np
import scipy as sp
from pcollections import *
from numpy.lib.mixins import NDArrayOperatorsMixin

from ..doc import docwrap
from ..util import (
    to_frozenarray,
    is_integer,
    is_tuple,
    is_amap,
    is_array,
    merge,
    to_array,
    unbroadcast_index)
from ..workflow import *


# MetaObject ###################################################################

class MetaObject(planobject):
    """Base planobject type for objects that keep track of metadata.

    Parameters
    ----------
    metadata : None or Mapping, optional
        The dictionary of metadata that is to be attached to the object. If the
        argument `None` is provided (the default), then the empty dictionary is
        used.

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

        The arguments and keyword arguments to `withmeta` are merged,
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


# larray #######################################################################
    
class larray(NDArrayOperatorsMixin):
    """A lazy array type that simulates ndarray and does not cache values.

    The `larray` type is a lazy array type that computes values from a stored
    function only when requested.
    """
    # Static and Class Methods -------------------------------------------------
    index_dtypes = (np.uint8, np.uint16, np.uint32, np.uint64)
    @classmethod
    def _calc_dtype(cls, shape):
        m = np.prod(shape) - 1 # max value an index will take
        return next(t for t in cls.index_dtypes if m <= np.iinfo(t).max)
    @classmethod
    def _make_index(cls, shape, order):
        dtype = cls._calc_dtype(shape)
        m = np.prod(shape) # max value an index will take + 1
        idx = np.reshape(np.arange(m, dtype=dtype), shape, order=order)
        return to_frozenarray(idx, copy=False)
    @classmethod
    def _filter_shape_dtype(cls, shape, dtype, order):
        if order == 'K':
            order = None
        if isinstance(shape, int):
            if shape >= 0:
                shape = (shape,)
                idx = cls._make_index(shape, 'C' if order is None else order)
                dt = object
            else:
                raise ValueError("shape dimensions must be >= 0")
        elif isinstance(shape, tuple):
            if all(isinstance(s, int) and s >= 0 for s in shape):
                idx = cls._make_index(shape, 'C' if order is None else order)
                dt = object
            else:
                raise ValueError("shape dimensions must be non-negative ints")
        elif isinstance(shape, larray):
            idx = shape._index
            dt = shape.dtype
            shape = shape.shape
            if order is not None and order != idx.order:
                idx = to_frozenarray(np.array(idx, order=order), copy=False)
        else:
            raise ValueError("shape must be a tuple of non-negative ints")
        if dtype is not None:
            dt = dtype
        return (shape, dt, idx)
    # Private Methods ----------------------------------------------------------
    def _raise_immerr(self, *args, **kwargs):
        raise TypeError(f"{type(self)} is immutable")
    # Construction -------------------------------------------------------------
    __slots__ = (
        'dtype', 'shape', 'ndim', 'size',
        'fn',
        'index_array',
        'refshape')
    def __new__(cls, fn, shape, dtype=None, order=None):
        (shape, dtype, idx) = cls._filter_shape_dtype(shape, dtype, order)
        self = object.__new__(cls)
        object.__setattr__(self, 'dtype', np.dtype(dtype))
        object.__setattr__(self, 'shape', shape)
        object.__setattr__(self, 'ndim', len(shape))
        object.__setattr__(self, 'size', np.prod(shape))
        object.__setattr__(self, 'fn', fn)
        object.__setattr__(self, 'index_array', idx)
        object.__setattr__(self, 'refshape', shape)
        return self
    # Public Methods -----------------------------------------------------------
    def __array__(self, dtype=None, *, copy=None):
        if copy == False:
            raise ValueError(
                f"type {type(self)} cannot be converted to an array with"
                f" copy=False")
        return self.toarray(dtype)
    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        # We can't handle 'out' options:
        nikeys = ('out','where','axes','axis','keepdims','casting','signature')
        if ( all(kwargs.get(k) is None for k in nikeys)
             and kwargs.get('subok') != False
             and method == '__call__'):
            dtype = kwargs.get('dtype')
            order = kwargs.get('order')
            if dtype is None:
                args = tuple(
                    arg if isinstance(arg, larray) else np.asarray(arg)
                    for arg in args)
                dtypes = (arg.dtype for arg in args)
                dtype = ufunc.resolve_dtypes((*dtypes, None))[-1]
            return self.narymap(
                ufunc, *args, kwargs=kwargs,
                dtype=dtype,
                order=order)
        else:
            arr = self.toarray()
            args = tuple(
                arr if arg is self else arg
                for arg in args)
            return arr.__array_ufunc__(ufunc, method, *args, **kwargs)
    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        # Okay, we need to get the subindex via the index array.
        idx = self.index_array
        newidx = idx[index]
        if isinstance(newidx, idx.dtype.type):
            # This is a valid single integer; calculate it.
            return self._callfn_rav(newidx)
        if np.array_equal(newidx, idx):
            return self
        # Create a new larray with this index
        newidx = to_frozenarray(newidx, copy=False)
        return self._clone(index=newidx)
    def __getattr__(self, k):
        if k == 'T':
            return self._clone(index=self.index_array.T)
        elif k == 'conj':
            return self._clone(fn=lambda ii:np.conj(self.fn(ii)))
        elif k == 'real':
            return self._clone(fn=lambda ii:np.real(self.fn(ii)))
        elif k == 'imag':
            return self._clone(fn=lambda ii:np.imag(self.fn(ii)))
        elif k == 'ndim':
            return len(self.shape)
        else:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute {k}")
    def __str__(self):
        return f"[<lazy {'x'.join(map(str, self.shape))} {self.dtype}>]"
    def __repr__(self):
        return f"larray([<{'x'.join(map(str, self.shape))} {self.dtype}>])"
    def __len__(self):
        if self.shape == ():
            raise TypeError("len() of unsized larray")
        return self.shape[0]
    def __contains__(self, *args):
        raise TypeError(
            "larray does not support `a in b` queries; use `a in b.toarray()`")
    def __iter__(self):
        if self.shape == ():
            raise TypeError("iteration over 0-d larray")
        return map(self.__getitem__, range(self.shape[0]))
    def __eq__(self, x):
        return self.narymap(op.eq, self, x, dtype=bool)
    def __neq__(self, x):
        return self.narymap(op.ne, self, x, dtype=bool)
    def __le__(self, x):
        return self.narymap(op.le, self, x, dtype=bool)
    def __ge__(self, x):
        return self.narymap(op.ge, self, x, dtype=bool)
    def __lt__(self, x):
        return self.narymap(op.lt, self, x, dtype=bool)
    def __gt__(self, x):
        return self.narymap(op.gt, self, x, dtype=bool)
    def __add__(self, x):
        return self.narymap(
            op.add, self, x,
            dtype=np.result_type(self.dtype, x))
    def __sub__(self, x):
        return self.narymap(
            op.sub, self, x,
            dtype=np.result_type(self.dtype, x))
    def __mul__(self, x):
        return self.narymap(
            op.mul, self, x,
            dtype=np.result_type(self.dtype, x))
    def __matmul__(self, x):
        return self.toarray() @ x
    def __truediv__(self, x):
        return self.narymap(
            op.truediv, self, x,
            dtype=np.result_type(self.dtype, x))
    def __floordiv__(self, x):
        return self.narymap(
            op.floordiv, self, x,
            dtype=np.result_type(self.dtype, x))
    def __mod__(self, x):
        return self.narymap(
            op.mod, self, x,
            dtype=np.result_type(self.dtype, x))
    def __divmod__(self, x):
        return self.narymap(
            op.divmod, self, x,
            dtype=np.result_type(self.dtype, x))
    def __pow__(self, x):
        return self.narymap(
            op.pow, self, x,
            dtype=np.result_type(self.dtype, x))
    def __lshift__(self, x):
        return self.narymap(
            op.lshift, self, x,
            dtype=np.result_type(self.dtype, x))
    def __rshift__(self, x):
        return self.narymap(
            op.rshift, self, x,
            dtype=np.result_type(self.dtype, x))
    def __and__(self, x):
        return self.narymap(
            op.and_, self, x,
            dtype=np.result_type(self.dtype, x))
    def __or__(self, x):
        return self.narymap(
            op.or_, self, x,
            dtype=np.result_type(self.dtype, x))
    def __xor__(self, x):
        return self.narymap(
            op.xor, self, x,
            dtype=np.result_type(self.dtype, x))
    def __radd__(self, x):
        return self.narymap(
            op.add, x, self,
            dtype=np.result_type(self.dtype, x))
    def __rsub__(self, x):
        return self.narymap(
            op.sub, x, self,
            dtype=np.result_type(self.dtype, x))
    def __rmul__(self, x):
        return self.narymap(
            op.mul, x, self,
            dtype=np.result_type(self.dtype, x))
    def __rmatmul__(self, x):
        return x @ self.toarray()
    def __rtruediv__(self, x):
        return self.narymap(
            op.truediv, x, self,
            dtype=np.result_type(self.dtype, x))
    def __rfloordiv__(self, x):
        return self.narymap(
            op.floordiv, x, self,
            dtype=np.result_type(self.dtype, x))
    def __rmod__(self, x):
        return self.narymap(
            op.mod, x, self,
            dtype=np.result_type(self.dtype, x))
    def __rdivmod__(self, x):
        return self.narymap(
            op.divmod, x, self,
            dtype=np.result_type(self.dtype, x))
    def __rpow__(self, x):
        return self.narymap(
            op.pow, x, self,
            dtype=np.result_type(self.dtype, x))
    def __rlshift__(self, x):
        return self.narymap(
            op.lshift, x, self,
            dtype=np.result_type(self.dtype, x))
    def __rrshift__(self, x):
        return self.narymap(
            op.rshift, x, self,
            dtype=np.result_type(self.dtype, x))
    def __rand__(self, x):
        return self.narymap(
            op.and_, x, self,
            dtype=np.result_type(self.dtype, x))
    def __ror__(self, x):
        return self.narymap(
            op.or_, x, self,
            dtype=np.result_type(self.dtype, x))
    def __rxor__(self, x):
        return self.narymap(
            op.xor, x, self,
            dtype=np.result_type(self.dtype, x))
    __iadd__ = _raise_immerr
    __isub__ = _raise_immerr
    __imul__ = _raise_immerr
    __imatmul__ = _raise_immerr
    __itruediv__ = _raise_immerr
    __ifloordiv__ = _raise_immerr
    __imod__ = _raise_immerr
    __idivmod__ = _raise_immerr
    __ipow__ = _raise_immerr
    __ilshift__ = _raise_immerr
    __irshift__ = _raise_immerr
    __iand__ = _raise_immerr
    __ior__ = _raise_immerr
    __ixor__ = _raise_immerr
    def __neg__(self):
        return self.map(op.neg)
    def __pos__(self):
        return self.map(op.pos)
    def __abs__(self):
        return self.map(op.abs)
    def __invert__(self):
        return self.map(op.invert)
    def _toscalar(self, tt):
        if self.size != 1:
            raise ValueError(
                "can only convert an larray of size 1 to a Python scalar")
        else:
            return tt(self.item())
    def __complex__(self):
        return self._toscalar(complex)
    def __int__(self):
        return self._toscalar(int)
    def __float__(self):
        return self._toscalar(float)
    def __index__(self):
        if self.size != 1 or not np.issubdtype(self.dtype, np.integer):
            raise ValueError(
                "only integer scalar arrays can be converted into a scalar"
                " index")
        else:
            return int(self.item())
    def __round__(self):
        x = self._toscalar(float)
        return round(x)
    def __trunc__(self):
        x = self._toscalar(float)
        return math.trunc(x)
    def __floor__(self):
        x = self._toscalar(float)
        return math.floor(x)
    def __ceil__(self):
        x = self._toscalar(float)
        return math.ceil(x)
    def astype(self, dtype):
        return self._clone(dtype=np.dtype(dtype))
    def toarray(self, dtype=None):
        if dtype is None:
            dtype = self.dtype
        return np.reshape(
            np.fromiter(
                map(self._callfn_rav, self.index_array.flat),
                dtype=dtype,
                count=self.size),
            self.shape)
    def flatten(self):
        idx = self.index_array
        newidx = to_frozenarray(idx.flatten(), copy=False)
        return self._clone(index=newidx)
    def squeeze(self, axis=None):
        newidx = self.index_array.squeeze(axis=axis)
        if newidx.shape == self.index_array.shape:
            return self
        return self._clone(index=newidx)
    def broadcast_to(self, shape):
        """Broadcasts the given `larray` object to the given shape."""
        return self._clone(index=np.broadcast_to(self.index_array, shape))
    def map(self, f, dtype=None):
        """Maps the given function over the values of the lazy-array.

        For an object `a` of type `larray`, the method call `a.map(f)`
        immediately returns a new `larray` object whose values are equal to the
        application of `f` to the values of `a`. In other words, if `b =
        a.map(f)`, then `b[u] == f(a[u])` for any valid index `u`.
        """
        return self._clone(fn=lambda ii:f(self[ii]), dtype=dtype)
    @staticmethod
    def _narymap_apply(fn, args, kwargs, shape, bc_ii):
        un_ii = (
            unbroadcast_index(bc_ii, arg.shape, shape)
            for arg in args)
        fn_args = (
            a._callfn(ii) if isinstance(a, larray) else a[ii]
            for (a,ii) in zip(args, un_ii))
        return fn(*fn_args, **kwargs)
    @classmethod
    def narymap(cls, fn, *args,
                kwargs={}, dtype=None, order=None, copyargs=False):
        """Returns a lazily-calculated mapping of an n-ary function over the
        given `larray` objects.

        For a set of objects `a1, a2 ... an`, each of which is of type `larray`,
        `self.narymap(f, a1, a2, ... an)` immediately returns a new `larray`
        object whose values are equal to the application of `f` to the values of
        the `larray` objects. In other words, if `b = larray.narymap(f, a1, a2,
        ... an)` then `b[u] == f(a1[u], a2[u] ... an[u])` for any valid index u.

        This function raises an error if the dimensions of the various arguments
        are not broadcastable or if any of the arguments `a2, a3 ... an` are
        neither `larray` objects nor frozen `numpy.ndarray` objects.
        """
        nargs = len(args)
        if nargs == 0:
            raise ValueError("larray.narymap must have at least 2 arguments")
        # First of all, make sure we can broadcast the shapes.
        f = np.array if copyargs else np.asarray
        args = tuple(a if isinstance(a,larray) else f(a) for a in args)
        shape = np.broadcast_shapes(*(a.shape for a in args))
        wrapper_fn = partial(larray._narymap_apply, fn, args, kwargs, shape)
        if dtype is None:
            dtype = np.result_type(*(a.dtype for a in args))
        return cls(wrapper_fn, shape, dtype=dtype, order=order)
    def replace_fn(self, fn, dtype=None):
        return self._clone(fn=fn, dtype=dtype)
    def item(self):
        return self._callfn_rav(self.index_array.item())
    # Private Methods ----------------------------------------------------------
    def _clone(self, dtype=None, fn=None, index=None, refshape=None):
        cls = type(self)
        index = self.index_array if index is None else index
        obj = object.__new__(cls)
        setf = object.__setattr__
        setf(obj, 'dtype', self.dtype if dtype is None else dtype)
        setf(obj, 'shape', index.shape)
        setf(obj, 'ndim', len(index.shape))
        setf(obj, 'size', np.prod(index.shape))
        setf(obj, 'fn', self.fn if fn is None else fn)
        setf(obj, 'index_array', index)
        setf(obj, 'refshape', self.refshape if refshape is None else refshape)
        return obj
    def _callfn(self, tt):
        # The index tt must be an unraveled tuple-based index.
        if all(is_integer(u) for u in tt) or tt == ():
            return self.fn(tt)
        else:
            shape = np.broadcast_shapes(*(np.shape(u) for u in tt))
            n = np.prod(shape)
            tt = (np.broadcast_to(u, shape) for u in tt)
            return np.fromiter(
                map(self.fn, *(u.flat for u in tt)),
                count=n,
                dtype=self.dtype)
    def _callfn_rav(self, ii):
        # index must be an integer from the index whose value is to be
        # calculated.
        tt = np.unravel_index(ii, self.refshape)
        return self._callfn(tt)
    # Disabled Methods ---------------------------------------------------------
    def __setattr__(self, k, v):
        self._raise_immerr()
    def __delattr__(self, k):
        self._raise_immerr()
    def __setitem__(self, k, v):
        self._raise_immerr()
    def __delitem__(self, k):
        self._raise_immerr()


# ArrayIndex ###################################################################

class ArrayIndex:
    """A type that indexes the elements of an array for easy searching.

    The `ArrayIndex` class is a class that stores a (typically read-only) numpy
    array whose elements must all be unique and that creates an index of that
    array's elements. `ArrayIndex` objects primarily support a a `find` method
    that can be used to look up object indices.

    `ArrayIndex` objects require that the arrays they are given contain unique
    objects that are sortable and hashable.

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
    # Class Methods ------------------------------------------------------------
    @classmethod
    def _make_flatdata(cls, array):
        ii = np.argsort(array.flat)
        flatids = to_frozenarray(array.flat[ii], copy=False)
        flatins = to_frozenarray(ii, copy=False)
        return (flatids, flatins)
    # Construction -------------------------------------------------------------
    __slots__ = ('array', '_flatdata')
    def __new__(cls, array, freeze=True):
        if not freeze and is_array(array, frozen=True):
            freeze = True
        else:
            array = to_array(array, frozen=freeze)
        self = object.__new__(cls)
        object.__setattr__(self, '_flatdata', None)
        object.__setattr__(self, 'array', array)
        return self
    # Public Methods -----------------------------------------------------------
    @docwrap(indent=8)
    def find(self, ids, ravel=False, **kw):
        """Finds and returns the indices of the given identities.
        
        `index.find(id)` returns the index, in the original array on which
        `index` is based, of the identity `id`. If `id` is not in the original
        array, then a `KeyError` is raised.
        
        Parameters
        ----------
        ids : array-like
            The identity or identities to look up in the index.
        ravel : boolean, optional
            Whether the return value should be an array representing a raveled
            index into the flattened version of the original indexed array
            (`True`) or a tuple representing an unraveled multi-index into the
            original indexed array (`False`). The default is `False`.
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
            default = kw.pop('default')
            error = False
        else:
            default = None
            error = True
        if len(kw) > 0:
            k = next(iter(kw.keys()))
            raise TypeError(f"'{k}' is an invalid keyword argument for find()")
        ids = to_array(ids)
        (flatids, flatins) = self._get_flatdata()
        # flatids is the ids in sorded order; flatins is the argsort of the
        # original argsort--how to put the sorted ids back in canonical
        # order; flatarg is the argsort itself.
        ii = np.searchsorted(flatids, ids)
        ins = flatins[ii]
        notfound = flatids[ii] != ids
        anymissing = np.any(notfound)
        # If some were not found, we might need to raise an error.
        if error and anymissing:
            k = np.atleast_1d(ids[notfound])[0]
            raise KeyError(k)
        if ravel:
            if anymissing:
                ins = np.asarray(ins)
                ins[notfound] = default
        else:
            ins = np.unravel_index(ins, self.array.shape)
            if anymissing:
                if not is_tuple(default):
                    default = (default,) * len(ins)
                ins = tuple(np.asarray(u) for u in ins)
                for (u,d) in zip(ins, default):
                    u[notfound] = default
        return ins
    # Private Methods ----------------------------------------------------------
    def _get_flatdata(self):
        flatdata = self._flatdata
        if flatdata is None:
            flatdata = self._make_flatdata(self.array)
            object.__setattr__(self, '_flatdata', flatdata)
        return flatdata
    # Disabled Methods ---------------------------------------------------------
    def __setattr__(self, k, v):
        raise TypeError(f"{type(self)} is immutable")
    def __delattr__(self, k):
        raise TypeError(f"{type(self)} is immutable")
    def __setitem__(self, k, v):
        raise TypeError(f"{type(self)} is immutable")
    def __delitem__(self, k):
        raise TypeError(f"{type(self)} is immutable")


# LazyFrame ####################################################################

class LazyFrame(MetaObject):
    """A DataFrame-like table type for qualities arranged along arbitrary axes.

    A `LazyFrame` is similar to a pandas `DataFrame` object but is organized
    around lazily loading properties (columns) and around placing propertties on
    arbitrary axes, such as time or depth, that can then be queried against.

    A `LazyFrame` represents each column of the data as a "property" and
    represents each row as an "identity". The member value `frame.ident` is an
    array of the unique identities in the `LazyFrame` object `frame`. The
    `ident` array does not need to be a vector of identities, but whatever shape
    it has, the properties in the frames must have the same shape, and all the
    values in `ident` must be hashable and unique.


    Parameters
    ----------
    *arg : tuple
        The arguments that are to be represented as a `LazyFrame`. This argument
        list would typically be a mapping object (such as a `dict` or a
        `pcollections.ldict`) or a sequence of such mapping objects that are to
        be merged together to make the `LazyFrame`. In these mappings,
        the keys must be the property names, and the values must be the valid
        properties, whose format is described here:

        All properties passed in these arguments must match, along their initial
        axis or axes, the `shape` parameter. Additional axes may be present in
        the properties, and if they are present, they may be indexed for
        interpolation by providing them as a tuple `(array, axes)` instead of
        as an array alone. The `axes` member of the tuple must be a dict-like
        mapping with one entry per additional axis in `array` (i.e.,
        `len(shape) + len(axes) == array.ndim`). Each key in `coords` is
        interpreted as the name of the additional axis, and the value must be
        an array-like of real numeric values indicating the coordinates of each
        entry along the axis. The ordering of the coordinates in the dictionary
        must match the ordering of the additional axes. All axes are assumed to
        be independent for the purposes of interpolation.

        For example, to construct a lazy-frame with 5 identities (0 through 4)
        and one property named "intensity" that was measured at 10 timepoints
        each a half of a second apart, one might use the constructor:
        `LazyFrame({'intensity': (intensity, {'time': arange(0, 5.0, 0.5)})})`
        where `intensity` is a numpy array with shape `(5,10)`.

        If a property has additional axes but does not wish to expose them to
        interpolation, then a value of `None` may be associated with the
        coordinate in question. Such coordinates are always returned as
        dimensions in the result instead of being interpolated.

        If a property has a default value for when no explicit position along
        its interpolation axis is requested, then it can be specified in the
        axes matrix using the entry `axis_name: (coords, default)`. The value
        `default` provides the default value taken when no explicit value is
        requested for this interpolation item. Note that axes with specified
        defaults can still be obtained as entire dimensions by requesting a
        specific value for the axis of `...` (`Ellipsis`).
    ident : array or None, optional
        An array of unique hashable identities whose shape defines the required
        shape for all properties. If `ident` is `None` (the default), then the
        shape tuple provided by the `shape` option is used to create an identity
        matrix equivalent to `reshape(arange(prod(shape)), shape)`. If the
        `shape` option is also not provided, then a ready (non-lazy) entry of
        the arguments is used to deduce the shape; if no such non-lazy entry is
        found, then an error is raised unless the argument `readycount` is set
        to `True`, in which case lazy entries are examined. If no entries are
        included, then the shape is assumed to be `(0,)`.
    shape : tuple of ints or None, optional
        The shape of the identity array of the `LazyFrame`. If `None` is given
        (the default), then the shape is deduced from the `ident` matrix of
        from the properties themselves (see the `ident` option). If the shape
        and the `ident` options are both provided explicitly, then they must
        match.
    metadata : dict-like or None, optional
        An optional dictionary of metadata to attach to the frame.
    readycount : boolean, optional
        If neither `ident` nor `shape` are explicitly provided and none of the
        properties provided to the `LazyFrame` constructor are ready (i.e., not
        lazy) then this option determines whether an error is raised. If
        `readycount` is `True`, then no error is raised and instead the lazy
        properties are checked for sizes. If `readycount` is `False` (the
        default), then a `RuntimeError` is raised.
    """
    # Static Methods -----------------------------------------------------------
    @classmethod
    def _get_propshape(cls, ld, key):
        "Returns the shape implied by a property entry in the lazy-dict."
        arr = ld[key]
        if is_tuple(arr) and len(arr) == 2:
            (arr, coords) = val
            return np.shape(arr)[:np.ndim(arr) - len(coords)]
        else:
            return np.shape(val)
    @classmethod
    def _make_ident(cls, shape, uint=True):
        "Returns a default identity array for the given shape option."
        numel = np.prod(shape)
        maxel = numel - 1
        if uint:
            if maxel <= np.iinfo(np.uint8).max:
                dtype = np.uint8
            elif maxel <= np.iinfo(np.uint16).max:
                dtype = np.uint16
            elif maxel <= np.iinfo(np.uint32).max:
                dtype = np.uint32
            else:
                dtype = np.uint
        else:
            if maxel <= np.iinfo(np.int8).max:
                dtype = np.int8
            elif maxel <= np.iinfo(np.int16).max:
                dtype = np.int16
            elif maxel <= np.iinfo(np.int32).max:
                dtype = np.int32
            else:
                dtype = int
        ident = np.arange(numel, dtype=dtype)
        ident.setflags(write=False)
        return ident
    @classmethod
    def _fix_axis(cls, propkey, axiskey, axis):
        if isinstance(axis, tuple) and len(axis) == 2:
            (x0, default) = axis
            x = to_frozenarray(x0)
            if x is not x0:
                axis = (x, default)
        else:
            default = Ellipsis
            x0 = None
            x = to_frozenarray(axis)
            axis = (x, default)
        if len(x.shape) != 1:
            raise ValueError(
                f"axis {axiskey} given for property {propkey} in LazyFrame has"
                f" invalid shape {x.shape}")
        return axis
    @classmethod
    def _fix_prop(cls, shape, prop, key):
        "Checks that a property matches the frame's shape or raises an error."
        if isinstance(prop, lazy):
            prop = prop()
        if is_tuple(prop) and len(prop) == 2:
            (array, coords) = prop
            if not is_amap(coords):
                raise ValueError(
                    f"(array, coords) tuple given for property {key} uses type"
                    f" {type(coords)} for coords, but coords must be a mapping")
            array = to_frozenarray(array)
            coords = tdict(coords)
            csh = []
            for (axiskey,axis) in coords.items():
                ax = cls._fix_axis(key, axiskey, axis)
                if ax is not axis:
                    coords[axiskey] = ax
                csh.append(ax[0].shape[0])
            coords = coords.persistent()
            csh = tuple(csh)
            if shape != (array.shape + csh):
                raise ValueError(
                    f"LazyFrame key {key} has array shape {array.shape}, but"
                    f" ident shape is {shape} and coords have shape {csh}")
            if array is prop[0] and coords is prop[1]:
                return prop
        else:
            array = to_frozenarray(prop)
            coords = pdict.empty
            if shape != array.shape:
                raise ValueError(
                    f"LazyFrame key {key} has array shape {array.shape}, but"
                    f" ident shape is {shape} and coords were given")
        return (array, coords)
    # Constructor --------------------------------------------------------------
    def __init__(self, *args,
                 ident=None,
                 shape=None,
                 metadata=None,
                 readycount=False):
        # Initialize the MetaObject data.
        MetaObject.__init__(self, metadata)
        # Make a lazy dictionary from the provided arguments.
        self.datadict = ldict(merge(*args))
        self.ident = ident
        self.shape = shape
        self.readycount = readycount
    # If we are updated via a copy operation, we want to make sure that we parse
    # the new values we are given; since we keep a lot of our initial checks in
    # the init function, we just allocate a new object instead of going the
    # plandict route.
    def copy(self, **kwargs):
        if len(kwargs) == 0:
            return self
        readycount = kwargs.pop('readycount', False)
        if len(kwargs) == 1:
            k = next(kwargs.keys())
            if k == 'metadata':
                return MetaObject.copy(self, **kwargs)
        dd = kwargs.pop('datadict', self.datadict)
        ident = kwargs.pop('ident', self.ident)
        md = kwargs.pop('metadata', self.metadata)
        cls = type(self)
        return cls(dd, ident=ident, metadata=md, readycount=readycount)
    # Calculations -------------------------------------------------------------
    @calc('ident', 'shape', 'datadict', 'readycount', lazy=False)
    def filter_params(datadict, ident=None, shape=None, readycount=False):
        readycount = bool(readycount)
        # If we have been provided with an identity array, process it.
        if ident is None:
            if shape is None:
                if len(datadict) == 0:
                    shape = (0,)
                else:
                    # We try to deduce the row count
                    key = next(
                        filter(datadict.is_ready, datadict.keys()),
                        datadict)
                    if key is datadict:
                        # We weren't given a ready key we can use; if readycount
                        # is false, we stop here.
                        if not readycount: 
                            raise RuntimeError(
                                "LazyFrame cannot deduce shape: no shape,"
                                " ident, or ready properties given and"
                                " readycount is False")
                        # Otherwise, we can try the unready keys.
                        for key in datadict.keys():
                            # We try to extract a shape; on failure we keep
                            # going, because it is possible that not all lazy
                            # entries are intended to be used.
                            try:
                                shape = LazyFrame._get_propshape(props, key)
                                break
                            except Exception:
                                pass
                        # If shape is still None here, all keys failed!
                        if shape is None:
                            raise RuntimeError(
                                f"LazyFrame cannot deduce shape: no shape or"
                                f" ident given, and all {len(datadict)} lazy"
                                f" properties raised errors when loaded")
            # At this point, we have either been provided with or deduced a
            # shape for the ident matrix. We can go ahead and make a default
            # matrix.
            ident = LazyFrame._make_ident(shape)
        else:
            ident = to_frozenarray(ident)
            # If we were given a shape option, it must match.
            if shape is None:
                shape = ident.shape
            else:
                if not isinstance(shape, tuple):
                    shape = (shape,)
                if shape != ident.shape:
                    raise ValueError(
                        f"shape of ident [{ident.shape}] does not match the"
                        f" given shape option [{shape}]")
        # For all ready keys, we want to check that they are appropriate shapes
        # and have appropriate coordinates now.
        datadict = datadict.to_pdict()
        res = tdict(datadict)
        for (k,v) in datadict.items():
            if isinstance(v, lazy):
                vv = lazy(LazyFrame._fix_prop, shape, v, k)
            else:
                vv = LazyFrame._fix_prop(shape, v, k)
            if v is not vv:
                res[k] = vv
        datadict = ldict(res)
        return (ident, shape, datadict, readycount)
    @calc('ident_index')
    def calc_ident_index(ident):
        return ArrayIndex(ident, freeze=True)
    # Methods ------------------------------------------------------------------
    @docwrap(indent=8)
    def find(self, ident, ravel=False):
        """Returns the index/indices of the given id/ids in the lazy frame.

        Parameters
        ----------
        ident : array-like
            The identity or identities (if an array is provided) whose indices
            in the original array are to be looked up.
        ravel : boolean, optional
            Whether to return a raveled integer index (`True`) or an unraveled
            tuple index (`False`). The default is `False`.

        Returns
        -------
        index : array-like
            The index or indices corresponding to the `ident` argument.

        Raises
        ------
        RuntimeError
            If the given identity value(s) cannot be found in the original
            array.
        """
        return self.ident_index.find(ident, ravel=ravel)
    
