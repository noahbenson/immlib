# -*- coding: utf-8 -*-
###############################################################################
# immlib/util/_numeric.py


# Dependencies ################################################################

import inspect
from functools import (partial, wraps, update_wrapper)
from collections import namedtuple

import pint
import numpy as np
import scipy as sp
import scipy.sparse as sps
from scipy.sparse import issparse as scipy__is_sparse
from pcollections import *

from ..doc import docwrap
from ._core import (
    is_tuple, is_list, is_aseq, is_aset,
    is_str, streq, strnorm, 
    frozenarray, freezearray,
    unitregistry)



# PyTorch Configuration #######################################################

# If torch isn't imported or configured, that's fine, we just write our methods
# to generate errors. We want these errors to explain the problem, so we create
# our own error type, then have a wrapper for the functions that follow that
# automatically raise the error when torch isn't found.

class TorchNotFound(Exception):
    """Exception raised when PyTorch is requested but is not installed."""
    def __str__(self):
        return (
            "pytorch not found.\n\n"
            "Immlib does not require pytorch, but it must be installed for\n"
            "certain operations to work.\n\n"
            "See https://pytorch.org/get-started/locally/ for help\n"
            "installing pytorch.")
    @staticmethod
    def raise_self(*args, **kw):
        """Raises a `TorchNotFound` error."""
        raise TorchNotFound()
class FakeTorchPackage:
    """A class that raises errors for the keras package if it cannot be loaded.
    """
    __slots__ = ('__version__')
    def __new__(cls):
        self = object.__new__(cls)
        object.__setattr__(self, '__version__', '0.0.0')
        return self
    def __getattr__(self, k):
        raise TorchNotFound()
    @classmethod
    def is_tensor(cls, arg):
        return False
try:
    import torch
    torch_found = True
    @docwrap('immlib.util.checktorch', indent=8)
    def checktorch(f):
        """Decorator, ensures that PyTorch functions throw an informative error
        when PyTorch isn't found.
        
        A function that is wrapped with the ``@checktorch`` decorator will
        always throw a descriptive error message when PyTorch isn't found on
        the system rather than raising a complex exception. Any ``immlib``
        function that uses the ``torch`` library should use this decorator.

        The ``torch`` library was found on this system, so ``checktorch(f)``
        always returns ``f``.
        """
        return f
    @docwrap('immlib.util.alttorch', indent=8)
    def alttorch(f_alt):
        """Decorator that runs an alternative function when PyTorch isn't
        found on the system.
        
        A function ``f`` that is wrapped with the ``@alttorch(f_alt)``
        decorator will always run `f_alt` instead of ``f`` when called if
        PyTorch is not found on the system and will always run ``f`` when
        `PyTorch` is found.

        The ``torch`` library was found on this system, so
        ``alttorch(f)(f_alt)`` always returns ``f``.
        """
        return (lambda f: f)
except (ModuleNotFoundError, ImportError) as e:
    torch = FakeTorchPackage()
    torch_found = False
    @docwrap('immlib.util.checktorch', indent=8)
    def checktorch(f):
        """Decorator that ensures that PyTorch functions throw an informative
        error when PyTorch isn't found.
        
        A function that is wrapped with the ``@checktorch`` decorator will
        always throw a descriptive error message when PyTorch isn't found on
        the system rather than raising a complex exception. Any ``immlib``
        function that uses the ``torch`` library should use this decorator.

        The ``torch`` library was not found on this system, so
        ``checktorch(f)`` always returns a function with the same docstring as
        `f` but which raises a ``TorchNotFound`` exception.
        """
        from functools import wraps
        return wraps(f)(TorchNotFound.raise_self)
    @docwrap('immlib.util.alttorch', indent=8)
    def alttorch(f_alt):
        """Decorator that runs an alternative function when PyTorch isn't
        found on the system.
        
        A function ``f`` that is wrapped with the ``@alttorch(f_alt)``
        decorator will always run `f_alt` instead of ``f`` when called if
        PyTorch is not found on the system and will always run ``f`` when
        PyTorch is found.

        The ``torch`` library was not found on this system, so
        ``alttorch(f)(f_alt)`` always returns `f_alt`, or rather a version of
        `f_alt` wrapped to ``f``.
        """
        from functools import wraps
        return (lambda f: wraps(f)(f_alt))
    _sparse_torch_types = pdict()
    _sparse_torch_layouts = pdict()
# Get the torch version setup.
_torch_version = torch.__version__.split('.')
try:
    _torch_version = (
        int(_torch_version[0]),
        int(_torch_version[1]),
        int(_torch_version[2]))
except Exception:
    _torch_version = (
        int(_torch_version[0]),
        int(_torch_version[1]),
        _torch_version[2])


# Numerical Types #############################################################

from numpy import ndarray
def _is_numtype(obj, numtype, dtypes):
    if isinstance(obj, numtype):
        return True
    elif isinstance(obj, ndarray):
        return any(map(partial(np.issubdtype, obj.dtype), dtypes))
    elif torch.is_tensor(obj):
        return any(map(partial(np.issubdtype, obj.numpy().dtype), dtypes))
    else:
        return False
from numbers import Number
_number_dtypes = (np.number, np.bool_)
@docwrap('immlib.is_numberdata')
def is_numberdata(obj, /):
    """Returns ``True`` if an object is a Python number, otherwise ``False``.

    ``is_numberdata(obj)`` returns ``True`` if the given object ``obj`` is an
    instance of the ``numbers.Number`` type or if it is an instance of a
    numeric NumPy array or PyTorch tensor.

    Except in special cases, ``is_numberdata(x)`` is equivalent to
    ``is_complexdata(x)``.

    ``is_numberdata`` is related to the function ``is_numeric``: if
    ``is_numeric(x)`` is ``True`` then ``is_numberdata(x)`` is also
    ``True``. However, ``is_numberdata(10)`` is ``True`` while
    ``is_numeric(10)`` is not. ``is_numberdata`` is designed for determining
    whether an object represents numbers, whereas ``is_numeric`` is designed
    for querying the properties of NumPy arrays and PyTorch tensors such as
    their shapes and data types.

    Parameters
    ----------
    obj : object
        The object whose quality as a ``Number`` object or numerical array or
        tensor is to be assessed.

    Returns
    -------
    boolean
        ``True`` if `obj` is an instance of ``Number`` or is a numerical
        array or tensor, otherwise ``False``.

    See Also
    --------
    is_booldata, is_intdata, is_realdata, is_complexdata
    """
    return _is_numtype(obj, Number, _number_dtypes)
_bool_dtypes = (np.bool_,)
@docwrap('immlib.is_booldata')
def is_booldata(obj, /):
    """Returns ``True`` if an object is a boolean, otherwise ``False``.

    ``is_booldata(obj)`` returns ``True`` if the given object `obj` is an
    instance of the ``bool`` type or if it is an instance of a boolean NumPy
    array or PyTorch tensor.

    Parameters
    ----------
    obj : object
        The object whose quality as a ``bool`` object or boolean array or
        tensor is to be assessed.

    Returns
    -------
    boolean
        ``True`` if `obj` is an instance of ``bool`` or is a boolean array or
        tensor, otherwise ``False``.

    See Also
    --------
    is_intdata, is_realdata, is_complexdata, is_numberdata
    """
    return _is_numtype(obj, bool, _bool_dtypes)
from numbers import Integral
_integer_dtypes = (np.integer, np.bool_)
@docwrap('immlib.is_intdata')
def is_intdata(obj, /):
    """Returns ``True`` if an object is a Python integer, otherwise ``False``.

    ``is_intdata(obj)`` returns ``True`` if the given object `obj` is an
    instance of the ``numbers.Integral`` type or if it is an instance of a
    numeric NumPy array or PyTorch tensor whose dtype is an integer type.

    Parameters
    ----------
    obj : object
        The object whose quality as a ``Integral`` object or integer-valued
        array or tensor is to be assessed.

    Returns
    -------
    boolean
        ``True`` if `obj` is an instance of ``Integral`` or is an integer numpy
        array, otherwise ``False``.

    See Also
    --------
    is_booldata, is_realdata, is_complexdata, is_numberdata
    """
    return _is_numtype(obj, Integral, _integer_dtypes)
from numbers import Real
_real_dtypes = (np.floating, np.integer, np.bool_)
@docwrap('immlib.is_intdata')
def is_realdata(obj, /):
    """Returns ``True`` if an object is a Python number, otherwise ``False``.

    ``is_realdata(obj)`` returns ``True`` if the given object `obj` is an
    instance of the ``numbers.Real`` type or of a real-valued NumPy array or
    PyTorch tensor.

    Parameters
    ----------
    obj : object
        The object whose quality as a ``Real`` object or real-values NumPy
        array ot PyTorch tensor is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``Real`` or is a real-valued array
        or tensor, otherwise ``False``.

    See Also
    --------
    is_booldata, is_intdata, is_complexdata, is_numberdata
    """
    return _is_numtype(obj, Real, _real_dtypes)
from numbers import Complex
_complex_dtypes = (np.number, np.bool_)
@docwrap('immlib.is_complexdata')
def is_complexdata(obj):
    """Returns ``True`` if an object is a complex number, otherwise ``False``.

    ``is_complexdata(obj)`` returns ``True`` if the given object `obj` is an
    instance of the ``numbers.Complex`` type or an instance of a complex-valued
    NumPy array or PyTorch tensor.

    Parameters
    ----------
    obj : object
        The object whose quality as a ``Complex`` object is to be assessed.

    Returns
    -------
    boolean
        ``True`` if `obj` is an instance of ``Complex``, otherwise ``False``.

    See Also
    --------
    is_booldata, is_intdata, is_realdata, is_numberdata
    """
    return _is_numtype(obj, Complex, _complex_dtypes)


# Scalar Utilities ############################################################

def _is_scalar(obj, numtype):
    if isinstance(obj, np.ndarray) or torch.is_tensor(obj):
        if obj.shape != ():
            return False
        obj = obj.item()
    return isinstance(obj, numtype)
@docwrap('immlib.is_number')
def is_number(obj, /, dtype=None):
    """Determines whether the argument is a scalar number or not.

    ``is_number(obj)`` returns ``True`` if `obj` is a scalar number and
    ``False`` otherwise. The following are considered scalar numbers:
    
      - Any instances of ``numbers.Number``,
      - Any numpy array ``x`` whose shape is ``()`` such that ``x.item()`` is a
        scalar.

    Parameters
    ----------
    obj : object
        The object whose quality as a scalar number is to be tested.
    dtype : bool, int, float, complex, or None, optional
        The type of the scalar. If this is ``None`` (the default)``, then the
        type of the scalar must be a number but it needn't be any particular
        number.  Otherwise, it must match the given type.
    
    Returns
    -------
    bool
        ``True`` if `obj` is a scalar number value and ``False`` otherwise.

    See Also
    --------
    like_number, is_numberdata, is_numeric, is_bool, is_integer, is_real,
    is_complex
    """
    if dtype is None:
        return _is_scalar(obj, Number)
    elif dtype is bool:
        return _is_scalar(obj, bool)
    elif dtype is int:
        return _is_scalar(obj, Integral)
    elif dtype is float:
        return _is_scalar(obj, Real)
    elif dtype is complex:
        return _is_scalar(obj, Complex)
    else:
        raise ValueError(f"invalid dtype: {dtype}")
@docwrap('immlib.is_bool')
def is_bool(obj, /):
    """Determines whether the argument is a scalar boolean or not.

    ``is_bool(obj)`` returns ``True`` if `obj` is a scalar boolean and
    ``False`` otherwise.

    See Also
    --------
    is_scalar, is_booldata
    """
    return _is_scalar(obj, bool)
@docwrap('immlib.is_integer')
def is_integer(obj, /):
    """Determines whether the argument is a scalar integer or not.

    ``is_integer(obj)`` returns ``True`` if `obj` is a scalar integer and
    ``False`` otherwise. Note that booleans are considered integers.

    See Also
    --------
    is_scalar, is_intdata
    """
    return _is_scalar(obj, Integral)
@docwrap('immlib.is_real')
def is_real(obj, /):
    """Determines whether the argument is a scalar real number or not.

    ``is_real(obj)`` returns ``True`` if `obj` is a scalar real number and
    ``False`` otherwise. Note that booleans and integers are considered real
    numbers.

    See Also
    --------
    is_scalar, is_realdata
    """
    return _is_scalar(obj, Real)
@docwrap('immlib.is_complex')
def is_complex(obj, /):
    """Determines whether the argument is a scalar complex number or not.

    ``is_complex(obj)`` returns ``True`` if `obj` is a scalar complex number
    and ``False`` otherwise. Note that booleans, integers, and real numbers are
    all considered valid complex numbers.

    See Also
    --------
    is_number, is_complexdata
    """
    return _is_scalar(obj, Complex)
@docwrap('immlib.like_number')
def like_number(obj, /):
    """Determines whether the argument holds a scalar number value or not.

    ``like_number(x)`` returns ``True`` if ``x`` is already a scalar number, if
    ``x`` is a single-element numpy array or tensor, or if ``x`` is a sequence
    or set that has only one numerical element; otherwise, it returns
    ``False``.

    If ``like_number(x)`` returns ``True``, then ``to_number(x)`` will always
    return a valid Python number (i.e., an object of type ``numbers.Number``).

    See Also
    --------
    is_number, to_number
    """
    if isinstance(obj, Number):
        return True
    if torch.is_tensor(obj):
        return torch.numel(obj) == 1
    if not isinstance(obj, np.ndarray):
        try:
            obj = np.asarray(obj)
        except (TypeError, ValueError):
            return False
    return obj.size == 1 and is_numberdata(obj)
@docwrap('immlib.to_number')
def to_number(obj, /, unit=Ellipsis, *, ureg=None):
    """Converts the argument into a simple Python number.

    ``to_number(obj)`` returns a simple Python number representation of `obj`
    (in other words, `obj` will be a subtype of Python's ``numbers.Number``
    type). Any number, any NumPy array with only one element, and any PyTorch
    tensor with only one element can be converted into a scalar. If `obj` is a
    ``pint.Quantity`` then the return value is a quantity with the same unit as
    `obj` and whose magnitude is ``to_number(obj.m)``.

    Parameters
    ----------
    obj : object
        The object that is to be converted into a scalar number.
    unit : unit-like, bool, or None, optional
        If `obj` is a ``pint.Quantity`` object, the `unit` parameter determines
        how it is handled by ``to_number``. If `unit` is ``None`` and `obj` is
        a quantity, then an error will be raised. If `unit` is a valid
        ``pint.Unit`` or the an object that can be converted into a unit vis
        that ``immlib.unit`` function. then an error is raised if `obj` is not
        a quantity with alike units. If `unit` is ``Ellipsis`` (the default
        value), then the behavior depends on whether `obj` is a quantity: if
        `obj` is a quantity, the ``to_number`` function is run on its magnitude
        and a quantity with the same unit is returned; if `obj` is not a
        quantity, then the a non-quantity is returned.
    ureg : pint.UnitRegistry, None, or Ellipsis, optional
        The ``pint.UnitRegistry`` object to use for units. If `ureg` is
        ``Ellipsis``, then ``immlib.units`` is used. If `ureg` is ``None`` (the
        default), then the registry of `obj` is used if `obj` is a quantity,
        and ``immlib.units`` is used if not.


    Returns
    -------
    number or pint.Quantity
        A scalar number that is an object whose class is a subtype of
        ``numbers.Number`` or a ``pint.Quantity`` object whose magnitude is
        such a number.

    Raises
    ------
    TypeError
        If the argument is not like a scalar number.
    """
    if ureg is Ellipsis:
        from immlib import units as ureg
    # If obj is a quantity, we handle things differently.
    if isinstance(obj, pint.Quantity):
        if ureg is None:
            from ._quantity import unitregistry
            ureg = unitregistry(obj)
        if unit is None:
            raise ValueError("to_number: unit is None but Quantity given")
        q = ureg.Quantity(to_number(obj.m, unit=None), obj.u)
        if unit is not Ellipsis:
            from ._quantity import unit as to_unit
            q = q.to(to_unit(unit))
        return q
    elif isinstance(obj, Number):
        return obj
    elif torch.is_tensor(obj):
        if torch.numel(obj) == 1:
            return obj.item()
    else:
        u = np.asarray(obj)
        if u.size == 1 and is_numberdata(u):
            return u.item()
    raise TypeError(f"given object is not scalar-like: {obj}")


# Numerical Collection Suport #################################################

# Numerical collections include numpy arrays and torch tensors. These objects
# are handled similarly due to their overall functional similarity, and certain
# support functions are used for both.
def _numcoll_match(numcoll_shape, numcoll_dtype, ndim, shape, numel, dtype):
    """Checks that the actual numcoll shape and the actual numcol dtype match
    the requirements of the ndim, shape, and dtype parameters.
    """
    # Parse the shape int front and back requirements and whether middle values
    # are allowed.
    if shape is None:
        (sh_pre, sh_mid, sh_suf) = ((), True, ())
    elif shape == ():
        (sh_pre, sh_mid, sh_suf) = ((), False, ())
    elif np.shape(shape) == ():
        (sh_pre, sh_mid, sh_suf) = ((shape,), False, ())
    else:
        # We add things to the prefix until we get to an ellipsis...
        sh_pre = []
        for d in shape:
            if d is Ellipsis: break
            sh_pre.append(d)
        sh_pre = tuple(sh_pre)
        # We might have finished with just that; otherwise, note the ellipsis
        # and move on.
        if len(sh_pre) == len(shape):
            (sh_mid, sh_suf) = (False, ())
        else:
            sh_suf = []
            for d in reversed(shape):
                if d is Ellipsis: break
                sh_suf.append(d)
            sh_suf = tuple(sh_suf) # We leave this reversed!
            sh_mid = len(sh_suf) + len(sh_pre) < len(shape)
        assert len(sh_suf) + len(sh_pre) + int(sh_mid) == len(shape), \
            "only one Ellipsis may be used in the shape filter"
    # Parse ndim.
    if not (is_tuple(ndim) or is_aset(ndim) or is_list(ndim) or ndim is None):
        ndim = (ndim,)
    # See if we match in terms of numel, ndim, and shape
    sh = numcoll_shape
    if ndim is not None and len(sh) not in ndim:
        return False
    if numel is not None:
        n = np.prod(sh)
        if is_tuple(numel):
            if n not in numel:
                return False
        elif n != numel:
            return False
    ndim = len(sh)
    if ndim < len(sh_pre) + len(sh_suf):
        return False
    (npre, nsuf) = (0,0)
    for (s,p) in zip(sh, sh_pre):
        if p != -1 and p != s: return False
        npre += 1
    for (s,p) in zip(reversed(sh), sh_suf):
        if p != -1 and p != s: return False
        nsuf += 1
    # If there are extras in the middle and we don't allow them, we fail the
    # match.
    if not sh_mid and nsuf + npre != ndim:
        return False
    # See if we match the dtype.
    if dtype is not None:
        if is_numpydtype(numcoll_dtype):
            if is_aseq(dtype) or is_aset(dtype):
                dtype = [to_numpydtype(dt) for dt in dtype]
            else:
                # If we have been given a torch dtype, we convert it, but
                # otherwise we let np.issubdtype do the converstion so that
                # users can pass in things like np.integer meaningfully.
                if is_torchdtype(dtype):
                    dtype = to_numpydtype(dtype)
                if not np.issubdtype(numcoll_dtype, dtype):
                    return False
                dtype = (numcoll_dtype,)
        elif is_torchdtype(numcoll_dtype):
            if is_aseq(dtype) or is_aset(dtype):
                dtype = [to_torchdtype(dt) for dt in dtype]
            else:
                dtype = [to_torchdtype(dtype)]
        if numcoll_dtype not in dtype:
            return False
    # We match everything!
    return True


# Numpy Arrays ################################################################

# For testing whether numpy arrays or pytorch tensors have the appropriate
# dimensionality, shape, and dtype, we use some helper functions.
from numpy import dtype as numpy_dtype
@docwrap('immlib.util.is_numpydype')
def is_numpydtype(obj, /):
    """Returns ``True`` for a ``numpy.dtype`` object and ``False`` otherwise.

    ``is_numpydtype(obj)`` returns ``True`` if the given object `obj` is an
    instance of the ``numpy.dtype`` class.

    Parameters
    ----------
    obj : object
        The object whose quality as a NumPy ``dtype`` object is to be assessed.

    Returns
    -------
    boolean
        ``True`` if `obj` is a valid ``numpy.dtype``, otherwise ``False``.
    """
    return isinstance(obj, numpy_dtype)
@docwrap('immlib.util.like_numydtype')
def like_numpydtype(obj, /):
    """Returns ``True`` for any object that can be converted into a
    ``numpy.dtype`` object.

    ``like_numpydtype(obj)`` returns ``True`` if the given object `obj` is an
    instance of the ``numpy.dtype`` class, is a string that can be used to
    construct a ``numpy.dtype`` object, or is a ``torch.dtype`` object.

    Parameters
    ----------
    obj : object
        The object whose quality as a NumPy ``dtype`` object is to be assessed.

    Returns
    -------
    bool
        ``True`` if `obj` can be converted into a valid numpy ``dtype``,
        otherwise ``False``.
    """
    if is_numpydtype(obj) or is_torchdtype(obj):
        return True
    else:
        try:
            return is_numpydtype(np.dtype(obj))
        except TypeError:
            return False
@docwrap('immlib.util.to_numpydtype')
def to_numpydtype(obj, /):
    """Returns a ``numpy.dtype`` object equivalent to the given argument.

    ``to_numpydtype(obj)`` attempts to coerce the given `obj` into a
    ``numpy.dtype`` object. If `obj` is already a ``numpy.dtype`` object, then
    `obj` itself is returned. If the object cannot be converted into a
    ``numpy.dtype`` object, then an error is raised.

    The following kinds of objects can be converted into a ``numpy.dtype`` (see
    also ``like_numpydtype()``):
     - ``numpy.dtype`` objects;
     - ``torch.dtype`` objects;
     - ``None`` (the default ``numpy.dtype``);
     - strings that name ``numpy.dtype`` objects; or
     - any object that can be passed to ``numpy.dtype()``, such as
       ``numpy.int32``.

    Parameters
    ----------
    obj : object
        The object whose quality as a NumPy ``dtype`` object is to be assessed.

    Returns
    -------
    numpy.dtype
        The ``numpy.dtype`` object that is equivalent to the argument `obj`.

    Raises
    ------
    TypeError
        If the given argument `obj` cannot be converted into a ``numpy.dtype``
        object.
    """
    if is_numpydtype(obj):
        return obj
    elif is_torchdtype(obj):
        return torch.as_tensor((), dtype=obj).numpy().dtype
    else:
        return np.dtype(obj)
# Sparse Array/Tensor stuff.
def sparray_isfrozen(obj):
    return not obj.data.flags['WRITEABLE']
def sparray_freeze(obj):
    obj.data.setflags(write=False)
def sparray_frozen(obj):
    if not obj.data.flags['WRITEABLE']:
        return obj
    obj = obj.copy()
    obj.data.setflags(write=False)
    return obj
def ndarray_isfrozen(obj):
    return not obj.flags['WRITEABLE']
def ndarray_freeze(obj):
    obj.setflags(write=False)
def ndarray_frozen(obj):
    if not obj.flags['WRITEABLE']:
        return obj
    obj = obj.copy()
    obj.setflags(write=False)
    return obj
SparseLayout = namedtuple(
    'SparseLayout', 
    ('name',
     'scipy_type', 'scipy_matrix_type', 'scipy_tomethod',
     'torch_constructor', 'torch_layout', 'torch_tomethod'))
_sparse_layouts = tdict(
    bsr=SparseLayout(
        'bsr',
        sps.bsr_array, sps.bsr_matrix, 'tobsr',
        'sparse_bsr_tensor', 'sparse_bsr', 'to_sparse_bsr'),
    bsc=SparseLayout(
        'bsc',
        None, None, None,
        'sparse_bsc_tensor', 'sparse_bsc', 'to_sparse_bsc'),
    coo=SparseLayout(
        'coo',
        sps.coo_array, sps.coo_matrix, 'tocoo',
        'sparse_coo_tensor', 'sparse_coo', 'to_sparse_coo'),
    csr=SparseLayout(
        'csr',
        sps.csr_array, sps.csr_matrix, 'tocsr',
        'sparse_csr_tensor', 'sparse_csr', 'to_sparse_csr'),
    csc=SparseLayout(
        'csc',
        sps.csc_array, sps.csc_matrix, 'tocsc',
        'sparse_csc_tensor', 'sparse_csc', 'to_sparse_csc'),
    dia=SparseLayout(
        'dia',
        sps.dia_array, sps.dia_matrix, 'todia',
        None, None, None),
    dok=SparseLayout(
        'dok',
        sps.dok_array, sps.dok_matrix, 'todok',
        None, None, None),
    lil=SparseLayout(
        'lil',
        sps.lil_array, sps.lil_matrix, 'tolil',
        None, None, None))
for (k,v) in tuple(_sparse_layouts.items()):
    try:
        con = getattr(torch, v.torch_constructor)
        lay = getattr(torch, v.torch_layout)
        cas = getattr(torch.Tensor, v.torch_tomethod)
        _sparse_layouts[k] = SparseLayout(
            v.name, v.scipy_type, v.scipy_matrix_type, v.scipy_tomethod,
            con, lay, cas)
    except Exception:
        _sparse_layouts[k] = SparseLayout(
            v.name, v.scipy_type, v.scipy_matrix_type, v.scipy_tomethod,
            None, None, None)
_sparse_layouts = _sparse_layouts.persistent()
# Indices for going from type or layout to SparseLayout:
_sparse_index = tdict()
for (k,st) in _sparse_layouts.items():
    if st.scipy_type is not None:
        _sparse_index[st.scipy_type] = st
    if st.scipy_matrix_type is not None:
        _sparse_index[st.scipy_matrix_type] = st
    if st.torch_layout is not None:
        _sparse_index[st.torch_layout] = st
_sparse_index = _sparse_index.persistent()
_sparse_torch_layouts = frozenset(
    st.torch_layout
    for st in _sparse_layouts.values()
    if st.torch_layout is not None)
def torch__is_sparse(obj):
    if not torch.is_tensor(obj):
        return False
    return obj.layout in _sparse_torch_layouts
@docwrap('immlib.util.sparse_layout')
def sparse_layout(obj, /):
    """Returns a tuple containing data about a sparse array layout.

    ``sparse_layout(name)`` returns the ``SparseLayout`` tuple for the sparse
    array layout with the given ``name``. The ``name`` must be one of the
    following (see ``scipy.sparse`` for more information about layouts):
      - ``'bsr'``
      - ``'bsc'``
      - ``'coo'``
      - ``'csr'``
      - ``'csc'``
      - ``'dia'``
      - ``'dok'``
      - ``'lil'``

    Alternatively, ``sparse_layout(obj)`` returns the sparse layout information
    for the given sparse array or sparse tensor `obj`.

    Whether the argument is a string or another type, the value ``None`` is
    returned if the object does not correspond to a sparse layout.

    The ``SparseLayout`` namedtuple that is returned has the following
    elements:
     - ``scipy_type``: the scipy type (e.g., ``scipy.sparse.csr_array``).
     - ``scipy_matrix_type``: the matrix type (e.g.,
       ``scipy.sparse.csr_matrix``).
     - ``scipy_tomethod``: the scipy casting method name (e.g., ``'tocsr'``).
     - ``torch_constructor``: The torch constructor function (e.g.,
       ``torch.sparse_csr_tensor``).
     - ``torch_layout``: The torch layout object (e.g., ``torch.sparse_csr``).
     - ``torch_tomethod``: The name of the torch ``Tensor`` method for casting
       (e.g., ``'to_sparse_csr'``).
    """
    if isinstance(obj, SparseLayout):
        return obj
    elif isinstance(obj, pint.Quantity):
        return sparse_layout(obj.m)
    elif isinstance(obj, str):
        return _sparse_layouts.get(obj, None)
    elif torch.is_tensor(obj):
        if obj.layout in _sparse_torch_layouts:
            return _sparse_index.get(obj.layout, None)
    elif scipy__is_sparse(obj):
        return _sparse_index.get(type(obj), None)
    else:
        return _sparse_index.get(obj, None)
@docwrap('immlib.util.sparse_haslayout')
def sparse_haslayout(arr, layout):
    """Returns ``True`` if the given sparse array or tensor has the given
    layout.

    If the first argument ``arr`` is not a sparse array nor a sparse tensor,
    then the return value is ``False`` (i.e., no, the object does not have the
    given sparse array or sparse tensor layout).

    The second argument ``layout`` may be any valid argument to the
    ``sparse_layout`` function.
    """
    if isinstance(arr, pint.Quantity):
        return sparse_haslayout(arr.m, layout)
    dstlay = sparse_layout(layout)
    if dstlay is None:
        raise ValueError(f"invalid layout: {layout}")
    if not (scipy__is_sparse(arr) or torch__is_sparse(arr)):
        return False
    srclay = sparse_layout(arr)
    return srclay.name == dstlay.name
@docwrap('immlib.sparse_find')
def sparse_find(arr, /):
    """Returns the indices and values of nonzero elements of a sparse object.
    
    ``sparse_find(sp_array)`` is equivalent to ``scipy.sparse.find(sp_array)``
    for a sparse array ``sp_array``.
    
    ``sparse_find(sp_tensor)`` is equivalent to ``s.indices() + (s.values(),)``
    for a sparse PyTorch tensor ``sp_tensor`` and a version of it that has been
    coalesced, ``s = sp_tensor.coalesce()``. Note that the ``s.values()``
    tensor is cloned and detached before being returned.

    ``sparse_find(q)`` for a quantity ``q`` returns the equivalent of
    ``sparse_find(q.m)`` except that the returned value array will have the
    same magnitude as ``q``.

    Raises
    ------
    TypeError
        If `arr` is not a sparse array or sparse tensor.

    See Also
    --------
    sparse_data, sparse_indices
    """
    if isinstance(arr, pint.Quantity):
        from ._quantity import quant
        f = sparse_find(arr.m)
        return f[:-1] + (quant(f[-1], arr.u),)
    elif scipy__is_sparse(arr):
        return sps.find(arr)
    elif torch__is_sparse(arr):
        arr = arr.coalesce()
        return tuple(arr.indices()) + (arr.values().clone().detach(),)
    else:
        raise TypeError(f"sparse_find requires a sparse array or sparse tensor")
@docwrap('immlib.util.sparse_indices')
def sparse_indices(arr, /):
    """Returns the indices of the nonzero values in the given sparse object.

    ``sparse_indices(arr)`` is roughly equivalent to the expression
    ``stack(sparse_find(arr)[:-1])``---i.e., it returns a numpy array or a
    pytorch tensor of the index matrix of the nonzero values in `arr`.

    See Also
    --------
    sparse_find, sparse_data
    """
    if isinstance(arr, pint.Quantity):
        return sparse_indices(arr.m)
    elif scipy__is_sparse(arr):
        return np.stack(sps.find(arr)[:-1])
    elif torch__is_sparse(arr):
        arr = arr.coalesce()
        return arr.indices()
    else:
        raise TypeError(f"sparse_data requires a sparse array or sparse tensor")
@docwrap('immlib.util.sparse_data')
def sparse_data(arr, /):
    """Returns the data vector for the given sparse array or sparse tensor.

    ``sparse_data(arr)`` is equivalent to ``sparse_find(arr)[-1]``---i.e., it
    returns a vector of non-zero values in the sparse array or sparse tensor
    `arr`---with the exception that it returns the actual vector itself and
    not a copy of the data vector. Changes to the return value of this function
    will be reflected in `arr`.

    See Also
    --------
    sparse_find, sparse_indices
    """
    if isinstance(arr, pint.Quantity):
        from ._quantity import quant
        return quant(sparse_data(arr.m), arr.u)
    elif scipy__is_sparse(arr):
        return arr.data
    elif torch__is_sparse(arr):
        arr = arr.coalesce()
        return arr.values()
    else:
        raise TypeError(f"sparse_data requires a sparse array or sparse tensor")
@docwrap('immlib.util.sparse_tolayout')
def sparse_tolayout(obj, layout):
    """Copies a sparse object into another sparse object with a given layout.

    ``sparse_tolayout(sparr, layout)`` copies the given sparse SciPy array or
    sparse PyTorch tensor ``sparr`` into an equivalent array or tensor that
    uses the sparse layout given by the ``layout`` argument, which must be
    compatible with the ``sparse_layout`` function. The backend (PyTorch or
    SciPy) will not be changed.

    See Also
    --------
    sparse_layout
    """
    if isinstance(obj, pint.Quantity):
        from ._quantity import quant
        arr = sparse_tolayout(obj.m, layout)
        return obj if arr is obj.m else quant(arr, obj.u)
    lay = sparse_layout(layout)
    if scipy__is_sparse(obj):
        method = getattr(obj, lay.scipy_tomethod)
        if method is None:
            raise ValueError(f"layout is invalid for scipy arrays: {layout}")
        arr = method()
        # If obj was frozen, duplicate that.
        if arr is not obj and sparray_isfrozen(obj):
            sparray_freeze(arr)
        return arr
    elif torch__is_sparse(obj):
        obj = obj.coalesce()
        mtd = lay.torch_tomethod
        if mtd is None:
            raise ValueError(f"layout is invalid for pytorch tensors: {layout}")
        return mtd(obj)
    else:
        raise TypeError(
            "sparse_tolayout requires a sparse scipy array or"
            " a sparse pytorch tensor")
@docwrap('immlib.is_array')
def is_array(obj, /, *,
             dtype=None, shape=None, ndim=None, numel=None, frozen=None,
             sparse=None, quant=None, unit=Ellipsis, ureg=None):
    """Returns ``True`` if an object is a ``numpy.ndarray`` object, otherwise
    returns ``False``.

    ``is_array(obj)`` returns ``True`` if the given object `obj` is an instance
    of the ``numpy.ndarray`` class or is a ``scipy.sparse`` array, or if `obj`
    is a ``pint.Quantity`` object whose magnitude is one of these. Additional
    constraints may be placed on the object via the optional argments.

    Note that to ``immlib``, both ``numpy.ndarray`` arrays and ``scipy.sparse``
    arrays are considered "arrays". This behavior can be changed with the
    ``sparse`` parameter.

    Parameters
    ----------
    obj : object
        The object whose quality as a NumPy array object is to be assessed.
    dtype : dtype-like or None, optional
        The NumPy `dtype` that is required of the `obj` in order to be
        considered a valid ``ndarray``. The ``obj.dtype`` matches the given
        `dtype` parameter if either `dtype` is ``None`` (the default) or if
        ``obj.dtype`` is a sub-dtype of `dtype` according to
        ``numpy.issubdtype``. Alternately, `dtype` can be a tuple, in which
        case, `obj` is considered valid if its dtype is any of the dtypes in
        `dtype`. Note that in the case of a tuple, the dtype of `obj` must
        appear exactly in the tuple rather than be a subtype of one of the
        objects in the tuple.
    ndim : int, tuple or ints, or None, optional
        The number of dimensions that the object must have in order to be
        considered a valid numpy array. If ``None``, then any number of
        dimensions is acceptable (this is the default). If this is an integer,
        then the number of dimensions must be exactly that integer. If this is
        a list or tuple of integers, then the dimensionality must be one of
        these numbers.
    shape : int, tuple of ints, or None, optional
        If the ``shape`` parameter is not ``None``, then the given `obj` must
        have a shape that matches the parameter value. The value `shape` must
        be a tuple that is equal to the `obj`'s shape tuple with the following
        additional rules: a ``-1`` value in the ``shape`` tuple will match any
        value in the `obj`'s shape tuple, and a single ``Ellipsis`` may appear
        in `shape`, which matches any number of values in the `obj`'s shape
        tuple. The default value of ``None`` indicates that no restriction
        should be applied to the `obj`'s shape.
    numel : int, tuple of ints, or None, optional
        If the `numel` parameter is not ``None``, then the given `obj` must
        have the same number of elements as given by `numel`. If `numel` is a
        tuple, then the number of elements in `obj` must be in the `numel`
        tuple. The number of elements is the product of its shape.
    frozen : bool or None, optional
        If ``None``, then no restrictions are placed on the ``'WRITEABLE'``
        flag of `obj`. If ``True``, then the data in `obj` must be read-only in
        order for `obj` to be considered a valid array. If ``False``, then the
        data in `obj` must not be read-only.
    sparse : boolean or False, optional
        If the `sparse`` parameter is ``None``, then no requirements are placed
        on the sparsity of `obj` for it to be considered a valid array. If
        `sparse` is ``True`` or ``False``, then `obj` must either be sparse or
        not be sparse, respectively, for `obj` to be considered valid. If
        ``sparse`` is a string, then it must be either ``'coo'``, ``'lil'``,
        ``'csr'``, or ``'csr'``, indicating the required sparse array
        type. Only ``scipy.sparse`` matrices are considered valid sparse
        arrays.
    quant : bool, optional
        Whether ``Quantity`` objects should be considered valid arrays or not.
        If ``quant=True`` then `obj` is considered a valid array only when
        ``obj`` is a quantity object with a ``numpy`` array as the
        magnitude. If ``False``, then `obj` must be a ``numpy`` array itself
        and not a ``Quantity`` to be considered valid. If ``None`` (the
        default), then either quantities or ``numpy`` arrays are considered
        valid arrays.
    unit : unit-like, Ellipsis, or None, optional
        A unit with which the object `obj`'s unit must be compatible in order
        for `obj` to be considered a valid array. An `obj` that is not a
        quantity is considered to have a unit of ``None``, which is not the
        same as being a quantity with a dimensionless unit. In other words,
        ``is_array(array, quant=None)`` will return ``True`` for a numpy array
        while ``is_array(arary, quant='dimensionless')`` will return
        ``False``. If ``unit=Ellipsis`` (the default), then the object's unit
        is ignored.
    ureg : pint.UnitRegistry, None, or Ellipsis, optional
        The ``pint.UnitRegistry`` object to use for units. If `ureg` is
        ``Ellipsis``, then ``immlib.units`` is used. If `ureg` is ``None`` (the
        default), then the registry of `obj` is used if `obj` is a quantity,
        and ``immlib.units`` is used if not.

    Returns
    -------
    bool
        ``True`` if `obj` is a valid numpy array, otherwise ``False``.

    See Also
    --------
    is_tensor, is_numeric
    """
    if ureg is Ellipsis:
        from immlib import units as ureg
    # If this is a quantity, just extract the magnitude.
    if isinstance(obj, pint.Quantity):
        if quant is False:
            return False
        if ureg is None:
            from ._quantity import unitregistry
            ureg = unitregistry(obj)
        u = obj.u
        obj = obj.m
    elif quant is True:
        return False
    else:
        if ureg is None:
            from immlib import units as ureg
        u = None
    # At this point we want to check if this is a valid numpy array or scipy
    # sparse matrix; however how we handle the answer to this question depends
    # on the sparse parameter.
    if sparse is True:
        if not scipy__is_sparse(obj):
            return False
    elif sparse is False:
        if not isinstance(obj, ndarray):
            return False
    elif sparse is None:
        # Set sparse to True/False:
        sparse = scipy__is_sparse(obj)
        # Also, it still has to be either a numpy array or a scipy sparse array.
        if not (isinstance(obj, ndarray) or sparse):
            return False
    else:
        if is_str(sparse):
            sparse = strnorm(sparse.strip(), case=True, unicode=False)
            layout = sparse_layout(sparse)
            if layout is None:
                raise ValueError(f"invalid sparse array type: {sparse}")
        else:
            layout = sparse_layout(sparse)
            if layout is None:
                tt = type(sparse)
                raise ValueError(f"invalid sparse parameter of type {tt}")
        ltypes = (layout.scipy_type, layout.scipy_matrix_type)
        if not isinstance(obj, ltypes):
            return False
        sparse = True
    # At this point, the sparse parameter has been checked against the object
    # and sparse is now True if the object is a sparse array and False if not.
    # Next, check that the object is read-only or not.
    if frozen is True:
        if sparse:
            if not sparray_isfrozen(obj):
                return False
        elif not ndarray_isfrozen(obj):
            return False
    elif frozen is False:
        if sparse:
            if sparray_isfrozen(obj):
                return False
        else:
            if ndarray_isfrozen(obj):
                return False
    elif frozen is None:
        frozen = sparray_isfrozen(obj) if sparse else ndarray_isfrozen(obj)
    else:
        raise ValueError(
            f"frozen option must be boolean or None; got type {type(frozen)}")
    # Next, check compatibility of the units.
    if unit is None:
        # We are required to not be a quantity.
        if u is not None:
            return False
    elif unit is not Ellipsis:
        from ._quantity import alike_units
        if not is_tuple(unit):
            unit = (unit,)
        if not any(map(partial(alike_units, u), unit)):
            return False
    # Check the match to the numeric collection last.
    if dtype is None and shape is None and ndim is None and numel is None:
        return True
    return _numcoll_match(obj.shape, obj.dtype, ndim, shape, numel, dtype)
def to_array(obj, /, dtype=None, *,
             order=None, copy=False, sparse=None, frozen=None,
             quant=None, ureg=None, unit=Ellipsis, detach=True):
    """Reinterprets `obj` as a NumPy array or quantity with an array magnitude.

    ``immlib.to_array`` is roughly equivalent to the ``numpy.asarray`` function
    with a few exceptions:
    
      - ``to_array(obj)`` allows quantities for `obj` and, in such a case, will
        return a quantity whose magnitude has been reinterpreted as an array,
        though this behavior can be altered with the `quant` parameter;
      - ``to_array(obj)`` can extract the ``numpy`` array from ``torch`` tensor
        objects.

    Parameters
    ----------
    obj : object
        The object that is to be reinterpreted as, or if necessary covnerted
        to, a NumPy array object.
    dtype : data-type, optional
        The `dtype` that is passed to ``numpy.asarray()``.
    order : {'C', 'F'}, optional
        The array order that is passed to ``numpy.asarray()``.
    copy : boolean, optional
        Whether to copy the data in `obj` or not. If ``False``, then `obj` is
        only copied if doing so is required by the optional parameters. If
        ``True``, then `obj` is always copied if possible.
    sparse : bool, 'csr', 'coo', or None, optional
        If ``None``, then the sparsity of `obj` is the same as the sparsity of
        the array that is returned. Otherwise, the return value will always be
        either a ``scipy.spase`` matrix (``sparse=True``) or a
        ``numpy.ndarray`` (``sparse=False``) based on the given value of
        `sparse`. The `sparse` parameter may also be set to ``'bsr'``,
        ``'coo'``, ``'csc'``, ``'csr'``, ``'dia'``, or ``'dok'`` to return
        specific sparse matrix types.
    frozen : bool or None, optional
        Whether the return value should be read-only or not. If ``None``, then
        no changes are made to the return value; if a new array is allocated in
        the ``to_array()`` function call, then it is returned in a writeable
        form. If ``frozen=True``, then the return value is always a read-only
        array; if `obj` is not already read-only, then a copy of `obj` is
        always returned in this case. If ``frozen=False``, then the
        return-value is never read-only.
    quant : bool or None, optional
        Whether the return value should be a ``Quantity`` object wrapping the
        array (``quant=True``) or the array itself (``quant=False``). If
        `quant` is ``None`` (the default) then the return value is a quantity
        if either `obj` is a quantity or an explicit `unit` parameter is given
        and is not a quantity if `obj` is not a quantity.
    ureg : pint.UnitRegistry or None, optional
        The ``pint.UnitRegistry`` object to use for units. If `ureg` is
        ``Ellipsis``, then ``immlib.units`` is used. If `ureg` is ``None`` (the
        default), then no specific coersion to a ``UnitRegistry`` is performed
        (i.e., the same quantity class is returned).
    unit : unit-like, bool, or Ellipsis, optional
        The unit that should be used in the return value. When the return value
        of this function is a ``Quantity`` (see the `quant` parameter), the
        returned quantity always has a unit matching the `unit` parameter; if
        the provided `obj` is not a quantity, then its unit is presumed to be
        that requested by `unit`. When the return value of this function is not
        a ``Quantity`` object and is instead is a NumPy array object, then when
        `obj` is not a quantity the `unit` parameter is ignored, and when `obj`
        is a quantity, its magnitude is returned after conversion into
        `unit`. The default value of `unit`, ``Ellipsis``, indicates that, if
        `obj` is a quantity, its unit should be used, and `unit` should be
        considered dimensionless otherwise.
    detach : bool, optional
        If the argument is a PyTorch tensor that requires gradient tracking,
        then it must be detached from the gradient tracking system before it
        can be turned into an array. If `detach` is ``True`` (the default),
        then this detachment is performed automatically. Otherwise, an error is
        raised if a tensor would need to be detached.

    Returns
    -------
    numpy.ndarray or pint.Quantity
        Either a NumPy array equivalent to `obj` or a ``Quantity`` whose
        magnitude is a NumPy array equivalent to `obj`.

    Raises
    ------
    ValueError
        If invalid parameter values are given or if the parameters conflict.
    
    See Also
    --------
    to_tensor, to_numeric
    """
    if ureg is Ellipsis:
        from immlib import units as ureg
    # If obj is a quantity, we handle things differently.
    if isinstance(obj, pint.Quantity):
        q = obj
        obj = q.m
        if ureg is None:
            from ._quantity import unitregistry
            ureg = unitregistry(q)
    else:
        q = None
        if ureg is None:
            from immlib import units as ureg
    # Translate obj depending on whether it's a pytorch array / scipy sparse
    # matrix.  We need to think about whether the output array is being
    # requested in sparse format. If so, we handle the conversion differently.
    obj_is_spsparse = scipy__is_sparse(obj)
    obj_is_tensor = not obj_is_spsparse and torch.is_tensor(obj)
    obj_is_sparse = obj_is_spsparse or torch__is_sparse(obj)
    # If this is a tensor and it requires grad, we can check whether we can
    # duplicate it now or not.
    if obj_is_tensor and obj.requires_grad:
        if detach:
            obj = obj.detach()
        else:
            raise ValueError(
                f"to_array: tensor requires grad but detach is non-true")
    newarr = False # True means we own the memory of arr; False means we don't.
    if sparse is not False and (sparse is not None or obj_is_sparse):
        # That condition is rough to parse; essentially, in this then-clause:
        #  * the user isn't requesting a dense output explicitly, and
        #  * the inputs tell us that we need a sparse output (because either
        #    the user is requesting a sparse output explicitly or they have
        #    requested no change in output sparsity and the input is sparse).
        if sparse is None or sparse is True:
            layout = sparse_layout(obj if obj_is_sparse else 'coo')
        elif isinstance(sparse, str):
            sparse = strnorm(sparse.strip(), case=True, unicode=False)
            layout = sparse_layout(sparse)
            if layout is None:
                raise ValueError(
                    f"invalid scipy sparse array layout name: {sparse}")
        else:
            layout = sparse_layout(sparse)
            if layout is None:
                raise ValueError(
                    f"invalid scipy sparse array layout type: {type(sparse)}")
        # We now have a layout that we are converting into.
        if obj_is_sparse:
            # We're creating a scipy sparse output from a sparse input.
            if obj_is_tensor:
                # We're creating a scipy sparse output from a sparse tensor.
                arr = obj.coalesce()
                if obj is not arr:
                    newarr = True
                ii = arr.indices().detach().numpy()
                uu = arr.values().detach().numpy()
                if copy:
                    vv = np.array(uu, dtype=dtype, order=order)
                else:
                    vv = np.asarray(uu, dtype=dtype, order=order)
                newarr = newarr or (uu is not vv)
                arr = layout.scipy_type(
                    (vv, tuple(ii)),
                    shape=arr.shape,
                    dtype=dtype)
            else:
                # We're creating a scipy sparse output from another scipy
                # sparse matrix. The scipy.sparse API (i.e., the find()
                # function) does not let us get access to the data itself, and
                # using obj.data along with the first two return values of
                # sps.find (the indices) can cause problems because find
                # doesn't always return the data in the same order as
                # obj.data. So we must make a copy of the data in this case
                # unless we know that the dtype and order have not been
                # changed; since order doesn't apply to 1d vectors we can
                # ignore it.
                if dtype is None or dtype == obj.dtype:
                    arr = obj
                else:
                    (rr,cc,uu) = sps.find(obj)
                    vv = np.asarray(uu, dtype=dtype)
                    if vv is uu:
                        arr = obj
                    else:
                        arr = layout.scipy_type(
                            (vv, (rr,cc)),
                            shape=obj.shape,
                            dtype=dtype)
        else:
            # We're creating a scipy sparse matrix from a dense matrix.
            arr = obj.numpy() if obj_is_tensor else obj
            # Make sure our dtype and order match.
            arr = np.asarray(arr, dtype=dtype, order=order)
            # We just call the appropriate constructor.
            arr = layout.scipy_type(arr)
            newarr = True
        # We mark sparse as True so that below we know that the output is
        # sparse.
        sparse = True
    else:
        # We are creating a dense array output.
        if obj_is_sparse:
            # We are creating a dense array output from a sparse input.
            if obj_is_tensor:
                # We are creating a dense array output from a sparse tensor
                # input.
                arr = obj.to_dense().numpy()
            else:
                # We are creating a dense array output from a scipy sparse
                # array input.
                arr = obj.todense()
            # In both of these cases, a copy has already been made.
            arr = np.asarray(arr, dtype=dtype, order=order)
            newarr = True
        else:
            # We are creating a dense array output from a dense input.
            if obj_is_tensor:
                # We are creating a dense array output from a dense tensor
                # input.
                arr = obj.numpy()
            else:
                arr = obj
            # Whether we call array() or asarray() depends on the copy
            # parameter.
            if copy:
                tmp = np.array(arr, dtype=dtype, order=order)
            else:
                tmp = np.asarray(arr, dtype=dtype, order=order)
            newarr = tmp is not arr
            arr = tmp
        # We mark sparse as False so that below we know that the output is
        # dense.
        sparse = False
    # If a read-only array is requested, we either return the object itself (if
    # it is already a read-only array), or we make a copy and make it
    # read-only.
    if frozen is True:
        if sparse:
            arr = sparray_frozen(arr)
        else:
            arr = ndarray_frozen(arr)
    elif frozen is False:
        if sparse:
            if sparray_isfrozen(arr):
                if not newarr:
                    arr = arr.copy()
                arr.data.setflags(write=True)
        elif ndarray_isfrozen(arr):
            arr = np.array(arr)
    elif frozen is None:
        frozen = sparray_isfrozen(arr) if sparse else ndarray_isfrozen(arr)
    else:
        raise ValueError(f"bad parameter value for frozen: {frozen}")
    # Next, we switch on whether we are being asked to return a quantity or
    # not.
    if quant is None:
        quant = (q if unit is Ellipsis else unit) is not None
    if quant is True:
        if unit is None:
            raise ValueError(
                "to_array: cannot make a quantity (quant=True) without a unit"
                " (unit=None)")
        if q is None:
            if unit is Ellipsis:
                raise ValueError(
                    "to_array(x): cannot make a quantity (quant=True) with the"
                    " same unit as x (unit=...) when the x is not a quantity")
            return ureg.Quantity(arr, unit)
        else:
            from ._quantity import unitregistry
            if ureg is not unitregistry(q) or obj is not arr:
                q = ureg.Quantity(arr, q.u)
            if unit is not Ellipsis and ureg.Unit(unit) != q.u:
                return q.to(unit)
            else:
                return q
    elif quant is False:
        # Don't return a quantity, whatever the input argument.
        if unit is Ellipsis:
            # We return the current array/magnitude whatever its unit.
            return arr
        elif q is None:
            # We just pretend this was already in the given unit (i.e., ignore
            # unit).
            return arr
        elif unit is None:
            raise ValueError(
                "to_tensor: cannot extract unit None from quantity; to get the"
                " native unit, use unit=Ellipsis")
        else:
            if obj is not arr:
                q = ureg.Quantity(arr, q.u)
            # We convert to the given unit and return that.
            return q.m_as(unit)
    else:
        raise ValueError(
            f"to_array: quant must be boolean or None;"
            f" got object of type {type(quant)}")


# PyTorch Tensors #############################################################

# At this point, either torch has been imported or it hasn't, but either way,
# we can use @checktorch to make sure that errors are thrown when torch isn't
# present. Otherwise, we can just write the functions assuming that torch is
# imported.
@docwrap('immlib.unit.is_torchdtype')
@alttorch(lambda dt: False)
def is_torchdtype(obj, /):
    """Returns ``True`` for a PyTroch ``dtype`` object and ``False`` otherwise.
    
    ``is_torchdtype(obj)`` returns ``True`` if the given object `obj` is an
    instance of the ``torch.dtype`` class.
    
    Parameters
    ----------
    obj : object
        The object whose quality as a PyTorch ``dtype`` object is to be
        assessed.
    
    Returns
    -------
    bool
        ``True`` if `obj` is a valid ``torch.dtype``, otherwise ``False``.
    """
    return isinstance(obj, torch.dtype)
@docwrap('immlib.unit.like_torchdtype')
def like_torchdtype(obj, /):
    """Returns ``True`` for any object that can be converted into a
    ``torch.dtype``.
    
    ``like_torchdtype(obj)`` returns ``True`` if the given object `obj` is an
    instance of the ``torch.dtype`` class, is a string that names a
    ``torch.dtype`` object, or is a ``numpy.dtype`` object that is compatible
    with PyTorch. Note that ``None`` is equivalent to ``torch``'s default
    ``dtype``.
    
    Parameters
    ----------
    obj : object
        The object whose quality as a PyTorch ``dtype`` object is to be
        assessed.
    
    Returns
    -------
    bool
        ``True`` if `obj` can be converted into a valid ``torch.dtype``,
        otherwise ``False``.
    """
    if is_torchdtype(obj):
        return True
    elif is_numpydtype(obj):
        try:
            return None is not torch.from_numpy(np.array((), dtype=obj))
        except TypeError:
            return False
    elif is_str(obj):
        try:
            return is_torchdtype(getattr(torch, obj))
        except AttributeError:
            return False
    elif obj is None:
        return True
    else:
        try:
            return None is not torch.from_numpy(np.array((), dtype=obj))
        except Exception:
            return False
@docwrap('immlib.unit.to_torchdtype')
@checktorch
def to_torchdtype(obj, /):
    """Returns a ``torch.dtype`` object equivalent to the given argument `obj`.

    ``to_torchdtype(obj)`` attempts to coerce the given `obj` into a
    ``torch.dtype`` object. If `obj` is already a ``torch.dtype`` object,
    then `obj` itself is returned. If the object cannot be converted into a
    ``torch.dtype`` object, then an error is raised.

    The following kinds of objects can be converted into a ``torch.dtype`` (see
    also ``like_numpydtype()``):
     - ``torch.dtype`` objects;
     - ``numpy.dtype`` objects with compatible (numeric) types;
     - strings that name ``torch.dtype`` objects; or
     - any object that can be passed to ``numpy.dtype()``, such as
       ``numpy.int32``, that also creates a compatible (numeric) type.

    Parameters
    ----------
    obj : object
        The object whose quality as a NumPy ``dtype`` object is to be assessed.

    Returns
    -------
    numpy.dtype
        The ``numpy.dtype`` object that is equivalent to the argument `obj`.

    Raises
    ------
    TypeError
        If the given argument `obj` cannot be converted into a ``numpy.dtype``
        object.
    """
    if is_torchdtype(obj):
        return obj
    else:
        return torch.as_tensor(np.array([], dtype=obj)).dtype
def _is_never_tensor(obj,
                     dtype=None, shape=None, ndim=None, numel=None,
                     device=None, requires_grad=None,
                     sparse=None, quant=None, unit=Ellipsis, ureg=None):
    return False
@docwrap('immlib.is_tensor')
@alttorch(_is_never_tensor)
def is_tensor(obj, /, dtype=None, *,
              shape=None, ndim=None, numel=None,
              device=None, requires_grad=None,
              sparse=None, quant=None, unit=Ellipsis, ureg=None):
    """Returns ``True`` if the argument is a ``torch.tensor`` object, otherwise
    returns ``False``.

    ``is_tensor(obj)`` returns ``True`` if the given object `obj` is an
    instance of the ``torch.Tensor`` class or is a ``pint.Quantity`` object
    whose magnitude is an instance of ``torch.Tensor``. Additional constraints
    may be placed on the object via the optional argments.

    Parameters
    ----------
    obj : object
        The object whose quality as a PyTorch tensor object is to be assessed.
    dtype : dtype-like or None, optional
        The PyTorch `dtype` or a dtype-like object that is required to match
        that of the `obj` in order to be considered a valid tensor. The
        ``obj.dtype`` matches the given `dtype` parameter if either `dtype` is
        ``None`` (the default) or if ``obj.dtype`` is equal to the PyTorch
        equivalent ot `dtype`. Alternately, `dtype` can be a tuple, in which
        case, `obj` is considered valid if its dtype is any of the dtypes in
        `dtype`.  ndim : int or tuple or ints or None, optional The number of
        dimensions that the object must have in order to be considered a valid
        tensor. If ``None``, then any number of dimensions is acceptable (this
        is the default). If this is an integer, then the number of dimensions
        must be exactly that integer. If this is a list or tuple of integers,
        then the dimensionality must be one of these numbers.
    shape : int, tuple of ints, None, optional
        If the `shape` parameter is not ``None``, then the given `obj` must
        have a shape shape that matches the parameter value. The value of
        `shape` must be a tuple that is equal to `obj`'s shape tuple with the
        following additional rules: a ``-1`` value in the `shape` tuple will
        match any value in the `obj`'s shape tuple, and a single ``Ellipsis``
        may appear in `shape`, which matches any number of values in the
        `obj`'s shape tuple. The default value of ``None`` indicates that no
        restriction should be applied to the `obj`'s shape.
    numel : int, tuple of ints, or None, optional
        If the `numel` parameter is not ``None``, then the given `obj` must
        have the same number of elements as given by `numel`. If `numel` is a
        tuple, then the number of elements in `obj` must be in the `numel`
        tuple. The number of elements is the product of its shape.
    device : device-name or None, optional
        If `device` is ``None``, then a tensor with any `device` field is
        considered valid; otherwise, the `device` parameter must equal
        ``obj.device`` for `obj` to be considered a valid tensor. The default
        value is ``None``.
    requires_grad : bool or None, optional
        If ``None``, then a tensor with any `requires_grad` field is considered
        valid; otherwise, the `requires_grad` parameter must equal
        ``obj.requires_grad`` for `obj` to be considered a valid tensor. The
        default value is ``None``.
    sparse : bool or None, optional
        If the `sparse` parameter is ``None``, then no requirements are placed
        on the sparsity of `obj` for it to be considered a valid tensor. If
        `sparse` is ``True`` or ``False``, then `obj` must either be sparse or
        not be sparse, respectively, for `obj` to be considered valid. If
        `sparse` is a string, then it must be either ``'coo'`` or ``'csr'``,
        indicating the required sparse array type.
    quant : bool, optional
        Whether ``pint.Quantity`` objects should be considered valid tensors or
        not.  If `quant` is ``True`` then `obj` is considered a valid array
        only when `obj` is a quantity object with a ``torch`` tensor as the
        magnitude. If `quant` is ``False``, then `obj` must be a ``torch``
        tensor itself and not a ``Quantity`` to be considered valid. If `quant`
        is ``None`` (the default), then either quantities or ``torch`` tensors
        are considered valid.
    unit : unit-like, None, Ellipsis, optional
        A unit with which the object obj's unit must be compatible in order for
        `obj` to be considered a valid tensor. An `obj` that is not a quantity
        is considered to have a unit of ``None``. If ``unit=Ellipsis`` (the
        default), then the object's unit is ignored.
    ureg : UnitRegistry, None, or Ellipsis, optional
        The ``pint.UnitRegistry`` object to use for units. If `ureg` is
        ``Ellipsis``, then ``immlib.units`` is used. If `ureg` is ``None`` (the
        default), then the registry of `obj` is used if `obj` is a quantity,
        and ``immlib.units`` is used if not.

    Returns
    -------
    boolean
        ``True`` if `obj` is a valid PyTorch tensor whose properties match the
        requirements spelled out by the optional parameters, otherwise
        ``False``.

    See Also
    --------
    is_array, is_numeric
    """
    # If so, we can process the arguments.
    if ureg is Ellipsis:
        from immlib import units as ureg
    # If this is a quantity, just extract the magnitude.
    if isinstance(obj, pint.Quantity):
        if quant is False:
            return False
        if ureg is None:
            from ._quantity import unitregistry
            ureg = unitregistry(obj)
        u = obj.u
        obj = obj.m
    else:
        if quant is True:
            return False
        if ureg is None:
            from immlib import units as ureg
        u = None
    # Right away: is this a torch tensor or not?
    if not torch.is_tensor(obj):
        return False
    # Do we match the various torch field requirements?
    if device is not None:
        device = torch.device(device)
        if obj.device != device:
            return False
    if requires_grad is not None:
        if obj.requires_grad != requires_grad:
            return False
    # Do we match the sparsity requirement?
    if sparse is True:
        if not torch__is_sparse(obj):
            return False
    elif sparse is False:
        if torch__is_sparse(obj):
            return False
    elif sparse is not None:
        layout = sparse_layout(sparse)
        if layout is None:
            if isinstance(sparse, str):
                raise ValueError(f"invalid sparse option: {sparse}")
            else:
                raise ValueError(
                    f"invalid sparse option of type {type(sparse)}")
        if not sparse_haslayout(obj, layout):
            return False
    # Next, check compatibility of the units.
    if unit is None:
        # We are required to not be a quantity.
        if u is not None:
            return False
    elif unit is not Ellipsis:
        from ._quantity import alike_units
        if not is_tuple(unit):
            unit = (unit,)
        if not any(alike_units(u, uu) for uu in unit):
            return False
    # Check the match to the numeric collection last.
    if dtype is None and shape is None and ndim is None and numel is None:
        return True
    return _numcoll_match(obj.shape, obj.dtype, ndim, shape, numel, dtype)
@docwrap('immlib.to_tensor')
def to_tensor(obj, /, dtype=None, *,
              device=None, requires_grad=None, copy=False,
              sparse=None, quant=None, ureg=None, unit=Ellipsis):
    """Reinterprets `obj` as a PyTorch tensor or as a ``pint`` quantity with
    a tensor magnitude.

    ``immlib.to_tensor`` is roughly equivalent to the ``torch.as_tensor``
    function with a few exceptions:
    
      - ``to_tensor(obj)`` allows quantities for `obj` and, in such a case,
        will return a quantity whose magnitude has been reinterpreted as a
        tensor, though this behavior can be altered with the `quant`
        parameter;
      - ``to_tensor(obj)`` can convet a SciPy sparse matrix into a sparse
        tensor.

    Parameters
    ----------
    obj : object
        The object that is to be reinterpreted as or covnerted to, a PyTorch
        tensor object.
    dtype : dtype-like, optional
        The `dtype` that is passed to ``torch.as_tensor(obj)``.
    device : device name or None, optional
        The `device` parameter that is passed to ``torch.as_tensor(obj)``,
        ``None`` by default.
    requires_grad : bool or None, optional
        Whether the returned tensor should require gradient calculations or
        not.  If ``None`` (the default), then the objecct `obj` is not changed
        from its current gradient settings, if `obj` is a tensor, and `obj` is
        not made to track its gradient if it is converted into a tensor. If the
        `requires_grad` parameter does not match the given tensor's
        `requires_grad` field, then a copy is always returned.
    copy : bool, optional
        Whether to copy the data in `obj` or not. If ``False``, then `obj` is
        only copied if doing so is required by the optional parameters. If
        ``True``, then `obj` is always copied if possible.
    sparse : bool, {'csr','csc','bsr','bsc','coo'}, or None, optional
        If ``None``, then the sparsity of `obj` is the same as the sparsity of
        the tensor that is returned. Otherwise, the return value will always be
        either a spase tensor (``sparse=True``) or a dense tensor
        (``sparse=False``) based on the given value of ``sparse``. The
        ``sparse`` parameter may also be set to the name of a sparse layout in
        order to convert the object into that layout. Possible sparse layouts
        include ``'coo'``, ``'csr'``, ``'csc'``, ``'bsr'``, and ``'bsc'``.
    quant : bool or None, optional
        Whether the return value should be a ``Quantity`` object wrapping the
        array (``quant=True``) or the tensor itself (``quant=False``). If
        `quant` is ``None`` (the default) then the return value is a quantity
        if either `obj` is a quantity or an explicit `unit` parameter is given
        and is not a quantity if `obj` is not a quantity.
    ureg : pint.UnitRegistry or None, optional
        The ``pint.UnitRegistry`` object to use for units. If `ureg` is
        ``Ellipsis``, then ``immlib.units`` is used. If `ureg` is ``None`` (the
        default), then no specific coersion to a ``pint.UnitRegistry`` is
        performed (i.e., the specific subclass of ``pint.Quantity`` used by
        `obj` is not changed).
    unit : unit-like, bool, None, or Ellipsis, optional
        The unit that should be used in the return value. When the return value
        of this function is a ``pint.Quantity`` (see the `quant` parameter),
        the returned quantity always has units matching the `unit` parameter;
        if the provided `obj` is not a quantity, then its unit is presumed to
        be that requested by `unit`. When the return value of this function is
        not a ``pint.Quantity`` object and is instead a PyTorch tensor object,
        then when `obj` is not a quantity the `unit` parameter is ignored, and
        when `obj` is a quantity, its magnitude is returned after conversion
        into `unit`. The default value of `unit`, ``Ellipsis``, indicates that,
        if `obj` is a quantity, its unit should be used, and `unit` should be
        considered dimensionless otherwise.

    Returns
    -------
    torch.Tensor or pint.Quantity
        Either a PyTorch tensor equivalent to `obj` or a ``pint.Quantity``
        whose magnitude is a PyTorch tensor equivalent to `obj`.

    Raises
    ------
    ValueError
        If invalid parameter values are given or if the parameters conflict.

    See Also
    --------
    to_array, to_numeric
    """
    if ureg is Ellipsis:
        from immlib import units as ureg
    if dtype is not None:
        dtype = to_torchdtype(dtype)
    # If obj is a quantity, we handle things differently.
    if isinstance(obj, pint.Quantity):
        q = obj
        obj = q.m
        if ureg is None:
            from ._quantity import unitregistry
            ureg = unitregistry(q)
    else:
        q = None
        if ureg is None:
            from immlib import units as ureg
    # Translate obj depending on whether it's a pytorch tensor already or a
    # scipy sparse matrix.
    if torch.is_tensor(obj):
        if requires_grad is None:
            requires_grad = obj.requires_grad
        if device is None:
            device = obj.device
        if dtype is None:
            dtype = obj.dtype
        needs_copy = device != obj.device or dtype != obj.dtype
        prefs_copy = copy or requires_grad != obj.requires_grad
        if copy is False:
            if needs_copy:
                if device == obj.device:
                    msg = "dtype change"
                elif dtype == obj.dtype:
                    msg = "device change"
                else:
                    msg = "device and dtype change"
                raise ValueError(
                    f"copy=False requested, but copy required by {msg}")
            else: 
                arr = obj
        elif needs_copy:
            arr = obj.to(dtype=dtype, device=device)
        elif copy:
            arr = obj.detach().clone()
        else:
            arr = obj
        if arr.requires_grad != requires_grad:
            arr = arr.requires_grad_(requires_grad)
    else:
        if requires_grad is None:
            requires_grad = False
        if scipy__is_sparse(obj):
            (rows, cols, vals) = sps.find(obj)
            # Process these into a PyTorch COO matrix.
            ii = torch.as_tensor(
                np.array([rows, cols], dtype=np.int_),
                dtype=torch.long,
                device=device)
            vals = torch.as_tensor(vals, dtype=dtype, device=device)
            if dtype is None:
                dtype = vals.dtype
            arr = torch.sparse_coo_tensor(
                ii, vals, obj.shape,
                dtype=dtype,
                device=device,
                requires_grad=requires_grad)
            # If possible, convert to the layout we want.
            arr = sparse_tolayout(arr, obj.format)
        elif (copy or requires_grad is True or 
              (isinstance(obj, np.ndarray) and not obj.flags['WRITEABLE'])):
            arr = torch.tensor(
                obj,
                dtype=dtype,
                device=device,
                requires_grad=requires_grad)
            dtype = arr.dtype
        else:
            arr = torch.as_tensor(obj, dtype=dtype, device=device)
            dtype = arr.dtype
    # If there is an instruction regarding the output's sparsity, handle that
    # now.
    if sparse is True:
        # arr must be sparse (COO by default); make sure it is.
        if not torch__is_sparse(arr):
            arr = arr.to_sparse()
    elif sparse is False:
        # arr must not be a sparse array; make sure it isn't.
        if torch__is_sparse(arr):
            arr = arr.to_dense()
    elif sparse is not None:
        layout = sparse_layout(sparse)
        if layout is None:
            if isinstance(sparse, str):
                raise ValueError(f"invalid pytorch sparse layout: {sparse}")
            else:
                raise ValueError(
                    f"to_tensor: invalid sparse option of type {type(sparse)}")
        if arr.layout != layout.torch_layout:
            arr = layout.torch_tomethod(arr)
    # Next, we switch on whether we are being asked to return a quantity or
    # not.
    if quant is None:
        quant = (q if unit is Ellipsis else unit) is not None
    if quant is True:
        if unit is None:
            raise ValueError(
                "to_tensor: cannot make a quantity (quant=True) without a unit"
                " (unit=None)")
        if q is None:
            if unit is Ellipsis:
                raise ValueError(
                    "to_tensor(x): cannot make a quantity (quant=True) with"
                    " the same unit as x (unit=Ellipsis) when x is not a"
                    " quantity")
            return ureg.Quantity(arr, unit)
        else:
            from ._quantity import unitregistry
            if ureg is not unitregistry(q) or obj is not arr:
                q = ureg.Quantity(arr, q.u)
            if unit is not Ellipsis and ureg.Unit(unit) != q.u:
                return q.to(unit)
            else:
                return q
    elif quant is False:
        # Don't return a quantity, whatever the input argument.
        if unit is Ellipsis:
            # We return the current array/magnitude whatever its unit.
            return arr
        elif q is None:
            # We just pretend this was already in the given unit (i.e., ignore
            # unit).
            return arr
        elif unit is None:
            raise ValueError(
                "to_tensor: cannot extract unit None from quantity; to get the"
                " native unit, use unit=Ellipsis")
        else:
            if obj is not arr:
                q = ureg.Quantity(arr, q.u)
            # We convert to the given unit and return that.
            return q.m_as(unit)
    else:
        raise ValueError(
            f"to_tensor: quant must be boolean or None;"
            f" got object of type {type(quant)}")


# General Numeric Collection Functions ########################################

@docwrap('immlib.is_numeric')
def is_numeric(obj, /, dtype=None, *,
               shape=None, ndim=None, numel=None,
               sparse=None, quant=None, unit=Ellipsis, ureg=None):
    """Returns ``True`` if an object is a numerical collection type and
    ``False`` otherwise.

    ``is_numeric(obj)`` returns ``True`` if the given object `obj` is an
    instance of the ``torch.Tensor`` class, the ``numpy.ndarray`` class, one
    one of the ``scipy.sparse`` array classes, or is a ``pint.Quantity``
    object whose magnitude is an instance of one of these types. Additional
    constraints may be placed on the object via the optional argments.

    .. Note:: The ``is_numeric`` function is similar to the ``is_array`` and
        ``is_tensor`` functions butis agnostic about whether its argument is a
        PyTorch tensor, a NumPy array, or an object that can be converted into
        one of these types.

    Parameters
    ----------
    obj : object
        The object whose quality as a numeric object is to be assessed.
    dtype : dtype-like or None, optional
        The NumPy or PyTorch `dtype` or dtype-like object that is required to
        match that of the `obj` in order to be considered valid. The
        ``obj.dtype`` matches the given `dtype` parameter if either `dtype` is
        ``None`` (the default) or if ``obj.dtype`` is equivalent to
        `dtype`. Alternately, `dtype` can be a tuple, in which case, `obj` is
        considered valid if its dtype is any of the dtypes in `dtype`.
    ndim : int, tuple or ints, or None, optional
        The number of dimensions that the object must have in order to be
        considered valid. If `ndim` is ``None``, then any number of dimensions
        is acceptable (this is the default). If it is an integer, then the
        number of dimensions must be exactly that integer. If this is a list or
        tuple of integers, then the dimensionality must be one of the listedn
        numbers.
    shape : int, tuple of ints, or None, optional
        If the `shape` parameter is not ``None``, then the given `obj` must
        have a shape that matches the parameter value. The value of `shape`
        must be a tuple that is equal to the shape of `obj` with the following
        additional rules: a ``-1`` value in the `shape` tuple will match any
        value in the shape of `obj`, and a single ``Ellipsis`` may appear in
        `shape`, which matches any number of values in the shape tuple of
        `obj`. The default value of ``None`` indicates that no restriction
        should be applied to the shape of `obj`.
    sparse : bool or False, optional
        If the ``sparse`` parameter is ``None``, then no requirements are
        placed on the sparsity of `obj` for it to be considered valid. If
        `sparse` is ``True`` or ``False``, then `obj` must either be sparse or
        not be sparse, respectively, for `obj` to be considered valid. If
        `sparse` is a string, then it must be a valid sparse array type that
        matches the type of `obj` for `obj` to be considered valid.
    numel : int, tuple of ints, or None, optional
        If the `numel` parameter is not ``None``, then the given `obj` must
        have the same number of elements as given by `numel`. If `numel` is a
        tuple, then the number of elements in `obj` must be in the `numel`
        tuple. The number of elements is the product of its shape.
    quant : bool, optional
        Whether ``pint.Quantity`` objects should be considered valid or not.
        If ``quant=True`` then `obj` is considered a valid numerical object
        only when `obj` is a quantity object with a valid numerical object as
        the magnitude. If ``quant=False``, then `obj` must be a numerical
        object itself and not a ``pint.Quantity`` to be considered valid. If
        ``quant=None`` (the default), then either quantities or numerical
        objects are considered valid.
    unit : unit-like or Ellipsis, optional
        A unit with which the unit of `obj` must be compatible in order for
        `obj` to be considered a valid numerical object. An `obj` that is not a
        quantity is considered to have dimensionless units. If
        ``unit=Ellipsis`` (the default), then the object's unit is ignored.
    ureg : UnitRegistry, None, or Ellipsis, optional
        The ``pint.UnitRegistry`` object to use for units. If `ureg` is
        ``Ellipsis``, then ``immlib.units`` is used. If `ureg` is ``None`` (the
        default), then the registry of `obj` is used if `obj` is a quantity,
        and ``immlib.units`` is used if not.

    Returns
    -------
    bool
        ``True`` if `obj` is a valid numerical object, otherwise ``False``.

    See Also
    --------
    is_array, is_tensor
    """
    if isinstance(obj, pint.Quantity):
        istns = torch.is_tensor(obj.m)
    else:
        istns = torch.is_tensor(obj)
    if istns:
        return is_tensor(obj,
                         dtype=dtype, shape=shape, ndim=ndim, numel=numel,
                         sparse=sparse, quant=quant, unit=unit, ureg=ureg)
    else:
        return is_array(obj,
                        dtype=dtype, shape=shape, ndim=ndim, numel=numel,
                        sparse=sparse, quant=quant, unit=unit, ureg=ureg)
@docwrap('immlib.to_numeric')
def to_numeric(obj, /, dtype=None, *,
               copy=False, sparse=None, quant=None, ureg=None, unit=Ellipsis):
    """Reinterprets `obj` as a numeric type or quantity with such a magnitude.

    ``immlib.to_numeric`` is roughly equivalent to the ``torch.as_tensor`` or
    ``numpy.asarray`` function with a few exceptions:

      - ``to_numeric(obj)`` allows quantities for `obj` and, in such a case,
        will return a quantity whose magnitude has been reinterpreted as a
        numeric object, though this behavior can be altered with the ``quant``
        parameter;
      - ``to_numeric(obj)`` correctly handles SciPy sparse matrices, NumPy
        arrays, and PyTorch tensors.

    If the object `obj` passed to ``immlib.to_numeric(obj)`` is a PyTorch
    tensor or a quantity whose magnitude is a PyTorch tensor, then a PyTorch
    tensor or a quantity with a PyTorch tensor magnitude is
    returned. Otherwise, a NumPy array, SciPy sparse matrix, or quantity with a
    magnitude matching one of these types is returned.

    Parameters
    ----------
    obj : object
        The object that is to be reinterpreted as, or if necessary covnerted
        to, a numeric object.
    dtype : dtype-like or None, optional
        The dtype that is passed to ``torch.as_tensor(obj)`` or
        ``np.asarray(obj)``.
    copy : bool, optional
        Whether to copy the data in `obj` or not. If ``False``, then `obj` is
        only copied if doing so is required by the optional parameters. If
        ``True``, then `obj` is always copied if possible.
    sparse : bool, {'csr','csc','bsr','bsc','coo'}, or None, optional
        If ``None``, then the sparsity of `obj` is the same as the sparsity of
        the object that is returned. Otherwise, the return value will always be
        either a spase object (``sparse=True``) or a dense object
        (``sparse=False``) based on the given value of `sparse`. The `sparse`
        parameter may also be set to ``'coo'``, ``'csr'``, or other sparse
        matrix names to return specific sparse layouts.
    quant : bool or None, optional
        Whether the return value should be a ``pint.Quantity`` object wrapping
        wrapping the object (``quant=True``) or the object itself
        (``quant=False``). If `quant` is ``None`` (the default) then the return
        value is a quantity if `obj` is a quantity and is not a quantity if
        `obj` is not a quantity.
    ureg : pint.UnitRegistry or None, optional
        The ``pint.UnitRegistry`` object to use for units. If `ureg` is
        ``Ellipsis``, then ``immlib.units`` is used. If `ureg` is ``None`` (the
        default), then no specific coersion to a ``pint.UnitRegistry`` is
        performed (i.e., the same quantity class is returned).
    unit : unit-like, bool, None, or Ellipsis, optional
        The unit that should be used in the return value. When the return value
        of this function is a ``pint.Quantity`` (see the `quant` parameter),
        the returned quantity always has a unit matching the `unit` parameter;
        if the provided `obj` is not a quantity, then its unit is presumed to
        be those requested by `unit`. When the return value of this function is
        not a ``pint.Quantity`` object and is instead a numeric object, then
        when `obj` is not a quantity the `unit` parameter is ignored, and when
        `obj` is a quantity, its magnitude is returned after conversion into
        `unit`. The default value of `unit`, ``Ellipsis``, indicates that, if
        `obj` is a quantity, its unit should be used, and `unit` should be
        considered to be ``None`` otherwise.

    Returns
    -------
    NumPy array or PyTorch tensor or Quantity
        Either a NumPy array or PyTorch tensor equivalent to `obj` or a 
        ``pint.Quantity`` whose magnitude is such an object.

    Raises
    ------
    ValueError
        If invalid parameter values are given or if the parameters conflict.

    See Also
    --------
    to_array, to_tensor
    """
    if torch.is_tensor(obj):
        return to_tensor(obj,
                         dtype=dtype, sparse=sparse,
                         quant=quant, unit=unit, ureg=ureg)
    else:
        return to_array(obj,
                        dtype=dtype, sparse=sparse,
                        quant=quant, unit=unit, ureg=ureg)


# Sparse Matrices and Dense Collections #######################################

@docwrap('immlib.is_sparse')
def is_sparse(obj, /, dtype=None, *,
              shape=None, ndim=None, numel=None,
              quant=None, ureg=None, unit=Ellipsis):
    """Returns ``True`` if an object is a sparse SciPy array or a sparse
    PyTorch tensor.

    ``is_sparse(obj)`` returns ``True`` if the given object `obj` is an
    instance of one of the SciPy sparse array classes, is a sparse PyTorch
    tensor, or is a ``pint.Quantity`` whose magnintude is one of
    theese. Additional constraints may be placed on the object via the optional
    argments.

    Parameters
    ----------
    obj : object
        The object whose quality as a sparse numerical object is to be
        assessed.
    %(immlib.is_numeric.parameters.dtype)s
    %(immlib.is_numeric.parameters.ndim)s
    %(immlib.is_numeric.parameters.shape)s
    %(immlib.is_numeric.parameters.numel)s
    %(immlib.is_numeric.parameters.quant)s
    %(immlib.is_numeric.parameters.ureg)s
    %(immlib.is_numeric.parameters.unit)s
    sparsetype : 'matrix', 'array', or None, optional
        The kind of sparse array to accept: either ``'matrix'`` for the scipy
        sparse matrix types (e.g., ``scipy.sparse.csr_matrix``) or ``'array'``
        for the sparse array types (e.g., ``scipy.sparse.csr_array``). If the
        value is ``None`` (the default) then either is accepted.

    Returns
    -------
    bool
        ``True`` if `obj` is a valid sparse numerical object, otherwise
        ``False``.
    """
    return is_numeric(obj, sparse=True,
                      dtype=dtype, shape=shape, ndim=ndim, numel=numel,
                      quant=quant, ureg=ureg, unit=unit)
@docwrap('immlib.to_sparse')
def to_sparse(obj, /, dtype=None, *, quant=None, ureg=None, unit=Ellipsis):
    """Returns a sparse version of the numerical object `obj`.

    ``to_sparse(obj)`` returns `obj` if it is already a PyTorch sparse tensor
    or a SciPy sparse matrix or a quantity whose magnitude is one of these.
    Otherwise, it converts `obj` into a sparse representation and returns
    this. Additional requirements on the output format of the return value can
    be added using the optional parameters.

    Parameters
    ----------
    obj : object
        The object that is to be converted into a sparse representation.
    %(immlib.to_numeric.parameters.dtype)s
    %(immlib.to_numeric.parameters.quant)s
    %(immlib.to_numeric.parameters.ureg)s
    %(immlib.to_numeric.parameters.unit)s

    Returns
    -------
    sparse tensor, sparse array, or quantity with a sparse magnitude
        A sparse version of the argument `obj`.
    """
    return to_numeric(obj, sparse=True,
                      dtype=dtype, quant=quant,
                      ureg=ureg, unit=unit)
@docwrap('immlib.is_dense')
def is_dense(obj, /, dtype=None, *,
             shape=None, ndim=None, numel=None,
             quant=None, ureg=None, unit=Ellipsis):
    """Returns ``True`` if an object is a dense NumPy array or PyTorch tensor.

    ``is_dense(obj)`` returns ``True`` if the given object `obj` is an instance
    of one of the NumPy ``ndarray`` classes, is a dense PyTorch tensor, or is a
    quantity whose magnintude is one of theese. Additional constraints may be
    placed on the object via the optional argments.

    Parameters
    ----------
    obj : object
        The object whose quality as a dense numerical object is to be assessed.
    %(immlib.is_numeric.parameters.dtype)s
    %(immlib.is_numeric.parameters.ndim)s
    %(immlib.is_numeric.parameters.shape)s
    %(immlib.is_numeric.parameters.numel)s
    %(immlib.is_numeric.parameters.quant)s
    %(immlib.is_numeric.parameters.ureg)s
    %(immlib.is_numeric.parameters.unit)s

    Returns
    -------
    bool
        ``True`` if `obj` is a valid dense numerical object, otherwise
        ``False``.
    """
    return is_numeric(obj, sparse=False,
                      dtype=dtype, shape=shape, ndim=ndim, numel=numel,
                      quant=quant, ureg=ureg, unit=unit)
@docwrap('immlib.to_dense')
def to_dense(obj, /, dtype=None, *, quant=None, ureg=None, unit=Ellipsis):
    """Returns a dense version of the numerical object `obj`.

    ``to_dense(obj)`` returns `obj` if it is already a PyTorch dense tensor or
    a NumPy ``ndarray`` or a quantity whose magnitude is one of these.
    Otherwise, it converts `obj` into a dense representation and returns
    this. Additional requirements on the output format of the return value can
    be added using the optional parameters.

    Parameters
    ----------
    obj : object
        The object that is to be converted into a dense representation.
    %(immlib.to_numeric.parameters.dtype)s
    %(immlib.to_numeric.parameters.quant)s
    %(immlib.to_numeric.parameters.ureg)s
    %(immlib.to_numeric.parameters.unit)s

    Returns
    -------
    dense tensor, dense array, or quantity with a dense magnitude
        A dense version of the argument `obj`.
    """
    return to_numeric(obj, sparse=False,
                      dtype=dtype, quant=quant, ureg=ureg, unit=unit)


# Numeric Decorators ##########################################################

@docwrap('immlib.numapi')
class numapi:
    """An interface for defining functions that expect all arguments to be
    either numpy arrays or pytorch tensors.

    A function decorated with ``@numapi`` is a placeholder for two
    subfunctions: one that is called when any of the arguments are pytorch
    tensors (all of whose arguments, when possible, are converted into pytorch
    tensors), and a version called otherwise, all of whose arguments are
    converted into numpy arrays when possible. The body of the decorated
    function is usually ``pass``, but, if desired, it can return either the
    pytorch or the numpy modules to indicate that a particular version of the
    function should be called (if necessary, tensors are converted into numpy
    arrays for this).

    Once a function has been decorated with ``@numapi``, that function should
    be used to decorate two other functions. If, for example, the function
    ``f`` is decorated with ``@numapi``, then ``@f.array`` should be used to
    decorate the version of the function that accepts numpy arrays and
    ``@f.tensor`` should be used to decorate the version of the function that
    accepts pytorch tensors.

    Examples
    --------
    >>> @numapi
    ... def l2_distance(pt1, pt2):
    ...     "Calculates the L2 distance between two points."
    ...     pass
    
    >>> @l2_distance.array
    ... def _(pt1, pt2):
    ...     return np.sqrt(np.sum((pt1 - pt2)**2, axis=0))
    
    >>> @l2_distance.tensor
    ... def _(pt1, pt2):
    ...     return torch.sqrt(torch.sum((pt1 - pt2)**2, axis=0))
    
    >>> l2_distance(torch.tensor([0,0]), [0,1])
    tensor(1.)
    
    >>> l2_distance([0,0], torch.tensor([0,1]))
    tensor(1.)
    
    >>> l2_distance([0,0], [0,1])
    1.0
    """
    # Static Methods ----------------------------------------------------------
    @staticmethod
    def _as_array(arg):
        argmod = type(arg).__module__
        if argmod == 'numpy' or argmod.startswith('numpy.'):
            return arg
        else:
            return to_array(arg)
    @staticmethod
    def _as_tensor(arg, device=None):
        if is_tensor(arg):
            return to_tensor(arg, device=device)
        argmod = type(arg).__module__
        if argmod == 'torch' or argmod.startswith('torch.'):
            return arg
        # Otherwise, try converting it to a tensor.
        try:
            return to_tensor(arg, device=device)
        except (TypeError, RuntimeError):
            pass
        # If all else fails, convert it to a numpy array.
        return numapi._as_array(arg)
    @staticmethod
    def _find_device(args, kwargs):
        dev = kwargs.get('device', None)
        if dev is not None:
            return torch.device(dev)
        tns = next(filter(is_tensor, args), None)
        if tns is None:
            tns = next(filter(is_tensor, kwargs.values()), None)
            if tns is None:
                return None
        return tns.device
    # Constructor -------------------------------------------------------------
    __slots__ = (
        'base_func', 'wrap_func', 'array_func', 'tensor_func', 'signature')
    def __new__(cls, fn):
        self = object.__new__(cls)
        def wrap_fn(*args, **kwargs):
            return self(*args, **kwargs)
        wrap_fn.numapi = self
        wrap_fn.array = self.array
        wrap_fn.tensor = self.tensor
        self.base_func = fn
        self.array_func = None
        self.tensor_func = None
        self.wrap_func = wrap_fn
        self.signature = inspect.signature(fn)
        return wraps(fn)(wrap_fn)
    # Methods -----------------------------------------------------------------
    def array(self, f):
        self.array_func = wraps(self.base_func)(f)
    def tensor(self, f):
        self.tensor_func = wraps(self.base_func)(f)
    def _bind(self, args, kwargs):
        b = self.signature.bind(*args, **kwargs)
        b.apply_defaults()
        return (b.args, b.kwargs)
    def _call_torch(self, args, kwargs):
        if not self.tensor_func:
            raise RuntimeError(
                f"tensor function for {self.__name__} was not defined")
        dev = numapi._find_device(args, kwargs)
        (args, kwargs) = self._bind(args, kwargs)
        args = map(numapi._as_tensor, args)
        if kwargs:
            kwargs = {k: numapi._as_tensor(v) for (k,v) in kwargs.items()}
        return self.tensor_func(*args, **kwargs)
    def _call_numpy(self, args, kwargs):
        if not self.array_func:
            raise RuntimeError(
                f"array function for {self.__name__} was not defined")
        (args, kwargs) = self._bind(args, kwargs)
        args = map(numapi._as_array, args)
        if kwargs:
            kwargs = {k: numapi._as_array(v) for (k,v) in kwargs.items()}
        return self.array_func(*args, **kwargs)
    def __call__(self, *args, **kwargs):
        # First, call the original function
        rval = self.base_func(*args, **kwargs)
        if rval is not None:
            if isinstance(rval, tuple):
                rvallen = len(rval)
                if rvallen == 3:
                    (rval, args, kwargs) = rval
                elif rvallen == 2:
                    (rval, args) = rval
                elif rvallen == 1:
                    rval = rval[0]
                else:
                    raise ValueError(
                        f"invalid value returned from numapi base_func: tuple"
                        f" must have 1-3 values but got {rvallen}")
            if rval is np:
                return self._call_numpy(args, kwargs)
            elif rval is torch:
                return self._call_torch(args, kwargs)
            else:
                raise ValueError(
                    f"invalid value returned from numapi base_func: {rval}")
        if torch_found:
            any_arg = any(map(is_tensor, args))
            any_inp = any_arg or any(map(is_tensor, kwargs.values()))
            if any_inp:
                return self._call_torch(args, kwargs)
        # Otherwise we use the array form.
        return self._call_numpy(args, kwargs)

# The tensor_args, array_args, and numeric_args decorators are very similar,
# but tensor_args has some extra magic for handling the keep_arrays option, and
# numeric_args also needs some magic for finding the device from the first
# tensor argument.
def _args_find_tensor(argvals, varargs, kwargs):
    try:
        return next(filter(is_tensor, argvals))
    except StopIteration:
        pass
    if varargs is not None:
        try:
            return next(filter(is_tensor, varargs))
        except StopIteration:
            pass
    if kwargs is not None:
        try:
            return next(filter(is_tensor, kwargs.values()))
        except StopIteration:
            pass
    return None
def _args_try_tensor(val, name=None, binding=None, first_tensor=None):
    device = None if first_tensor is None else first_tensor.device
    try:
        tns = to_tensor(val, device=device)
    except (TypeError, RuntimeError):
        return val
    if name is not None and val is not tns:
        binding.arguments[name] = tns
    return tns
def _args_try_array(val, name=None, binding=None, first_tensor=None):
    # (The first_tensor parameter is ignored, but included to be similar to the
    #  _args_try_tensor function.)
    try:
        arr = to_array(val)
    except TypeError:
        return val
    if name is not None and val is not arr:
        binding.arguments[name] = arr
    return arr
def _args_try_numeric(val, name=None, binding=None, first_tensor=None):
    try:
        if first_tensor is None:
            arr = to_array(val)
        else:
            arr = to_tensor(val, device=first_tensor.device)
    except TypeError:
        return val
    if name is not None and val is not arr:
        binding.arguments[name] = arr
    return arr
def _args_dispatch(args_try_fn, fn,
                   sig, sig_args, sig_vargs, sig_kwargs, keep_arrays,
                   *args, **kwargs):
    binding = sig.bind(*args, **kwargs)
    binding.apply_defaults()
    vals = tuple(map(binding.arguments.__getitem__, sig_args))
    if sig_vargs:
        vargs = binding.arguments[sig_vargs]
        nvargs = len(vargs)
    else:
        vargs = None
        nvargs = 0
    if sig_kwargs:
        kwargs = binding.arguments[sig_kwargs]
        nkw = len(kw)
    else:
        kwargs = None
        nkw = 0
    if args_try_fn is _args_try_array:
        first_tensor = None
    else:
        first_tensor = _args_find_tensor(vals, vargs, kwargs)
    # Convert to the appropriate types (and update the values in the arguments
    # list if there is a change in any of the args):
    for (argname,val) in zip(sig_args, vals):
        args_try_fn(val, argname, binding, first_tensor)
    if sig_vargs:
        new_vargs = tuple(
            args_try_fn(val, first_tensor=first_tensor)
            for val in vals[nargs:nargs+nva])
        binding.arguments[sig_varargs] = new_varargs
    if sig_kwargs:
        for (k,val) in kw.items():
            cnv = args_try_fn(val, first_tensor=first_tensor)
            if cnv is not val:
                kw[k] = tns
    rval = fn(*binding.args, **binding.kwargs)
    if keep_arrays and first_tensor is None:
        if is_tuple(rval):
            rval = tuple(
                to_array(u, copy=False) if is_tensor(u) else u
                for u in rval)
        elif is_tensor(rval):
            rval = to_array(rval, copy=False)
    return rval
def _promote_args_decorate(arglist, args_try_fn, keep_arrays, fn):
    "[Private] Dispatcher for tensor_args decorator."
    from ..workflow import calc
    sig = inspect.signature(fn)
    if arglist is None or len(arglist) == 0:
        # We convert all of the args.
        arglist = tuple(sig.parameters.keys())
    sig_args = []
    sig_kwargs = None
    sig_varargs = None
    params = sig.parameters
    for arg in arglist:
        p = params.get(arg)
        if p is None:
            raise ValueError(
                f"'{arg}' requested as tensor but not found in arguments")
        if p.kind is p.VAR_POSITIONAL:
            sig_varargs = p.name
        elif p.kind is p.VAR_KEYWORD:
            sig_kwargs = p.name
        else:
            sig_args.append(p.name)
    nargs = len(sig_args)
    dispatch = partial(
        _args_dispatch,
        args_try_fn, fn,
        sig, sig_args, sig_varargs, sig_kwargs, keep_arrays)
    return wraps(fn)(dispatch)
@docwrap('immlib.tensor_args')
def tensor_args(fn=None, /, *args, keep_arrays=False):
    """Converts arguments of the decorated function into PyTorch tensors.

    The decorator ``@tensor_args``, when applied to a function, will convert
    all of that function's arguments into PyTorch tensors prior to invoking the
    function. ``tensor_args`` considers ``pint.Quantity`` objects whose
    magnitudes are tensors to be tensors and will convert arguments that are
    quantitites into new quantities with tensor magnitudes.

    If a function is decorated with ``@tensor_args('arg1', 'arg2' ...)`` then
    only the arguments whose names are given (``arg1``, ``arg2``, ...) are
    converted into tensors.

    When arguments are converted into PyTorch tensors, the first object in the
    argument list that is already a tensor is found and its device is used as
    the device for all converted objects. If no such object is found, then
    ``None`` is used for the device.
    
    The optional argument `keep_arrays` (default: ``False``) can be set to
    ``True`` to indicate that the function should convert tensor return values
    back into NumPy arrays if none of the arguments to the function were
    originally tensors. This allows a function to be written using one
    numerical interface (PyTorch) but to work for either PyTorch tensors or
    NumPy arrays while returning values whose types match the input types.

    """
    if fn is None:
        # A function is being decorated with `@tensor_args(keep_arrays=value)`
        # but not `@tensor_args` alone.
        return partial(
            _promote_args_decorate,
            args, _args_try_tensor, keep_arrays)
    elif is_str(fn):
        # A function is being decorated with `@tensor_args('arg1' ...)`.
        return partial(
            _promote_args_decorate,
            (fn,) + args, _args_try_tensor, keep_arrays)
    elif not callable(fn):
        # We weren't given a string or a valid function to decorate.
        raise TypeError(
            f"expected string or callable for first argument; got {type(fn)}")
    else:
        # Otherwise, we have a callable, and maybe a list of strings. If we
        # have a list of strings, we may as well use it, thus allowing the
        # tensor_args decorator to be used either as:
        #   @tensor_args('a')
        #   def fn(a, b): ...
        # or as
        #   fn = tensor_args(lambda a,b: ..., 'a').
        return _promote_args_decorate(args, _args_try_tensor, keep_arrays, fn)
@docwrap('immlib.array_args')
def array_args(fn=None, /, *args):
    """Converts arguments of the decorated function into NumPy arrays.

    The decorator ``@array_args``, when applied to a function, will convert all
    of that function's arguments into NumPy arrays prior to invoking the
    function. ``array_args`` considers ``pint.Quantity`` objects whose
    magnitudes are arrays to be arrays and will convert arguments that are
    quantitites whose magnitudes are not arrays into new quantities with array
    magnitudes.

    If a function is decorated with ``@array_args('arg1', 'arg2' ...)`` then
    only the arguments whose names are given (``arg1``, ``arg2``, ...) are
    converted into arrays.
    """
    if fn is None:
        # A function is being decorated with `@array_args()` or `@array_args`
        # alone.
        return partial(
            _promote_args_decorate,
            args, _args_try_array, False)
    elif is_str(fn):
        # A function is being decorated with `@array_args('arg1' ...)`
        return partial(
            _promote_args_decorate,
            (fn,) + args, _args_try_array, False)
    elif not callable(fn):
        # We weren't given a string or a valid function to decorate.
        raise TypeError(
            f"expected string or callable for first argument; got {type(fn)}")
    else:
        # Otherwise, we have a callable, and maybe a list of strings. If we
        # have a list of strings, we may as well use it, thus allowing the
        # tensor_args decorator to be used either as:
        #   @array_args('a')
        #   def fn(a, b): ...
        # or as
        #   fn = array_args(lambda a,b: ..., 'a').
        return _promote_args_decorate(args, _args_try_array, False, fn)
@docwrap('immlib.numeric_args')
def numeric_args(fn=None, /, *args):
    """Converts arguments of the decorated function into either NumPy arrays or
    PyTorch tensors.

    The decorator ``@numeric_args``, when applied to a function, will convert
    all of that function's arguments into numeric collections--either PyTorch
    tensors or NumPy arrays--prior to invoking the function. Either all
    arguments are converted into either NumPy arrays or all arguments are
    converted into PyTorch tensors; the former only occurs when no PyTorch
    tensors occur in the argument list. ``numeric_args`` considers
    ``pint.Quantity`` objects whose magnitudes are numeric collections to be
    numeric collections and will convert arguments that are quantitites into
    new quantities with numeric magnitudes if necessary.

    If a function is decorated with ``@numeric_args('arg1', 'arg2' ...)`` then
    only the arguments whose names are given (``arg1``, ``arg2``, ...) are
    converted into numeric collections.

    When arguments are converted into PyTorch tensors, the first object in the
    argument list that is already a tensor is found and its device is used as
    the device for all converted objects. If no such object is found, then
    ``None`` is used for the device.
    """
    if fn is None:
        # A function is being decorated with `@numeric_args(keep_arrays=value)`
        # or `@numeric_args` alone.
        return partial(
            _promote_args_decorate,
            args, _args_try_numeric, False)
    elif is_str(fn):
        # A function is being decorated with `@array_args('arg1' ...)`
        return partial(
            _promote_args_decorate,
            (fn,) + args, _args_try_numeric, False)
    elif not callable(fn):
        # We weren't given a string or a valid function to decorate.
        raise TypeError(
            f"expected string or callable for first argument; got {type(fn)}")
    else:
        # Otherwise, we have a callable, and maybe a list of strings. If we
        # have a list of strings, we may as well use it, thus allowing the
        # numeric_args decorator to be used either as:
        #   @numeric_args('a')
        #   def fn(a, b): ...
        # or as
        #   fn = numeric_args(lambda a,b: ..., 'a').
        return _promote_args_decorate(args, _args_try_numeric, False, fn)
