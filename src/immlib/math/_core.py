# -*- coding: utf-8 -*-
###############################################################################
# immlib/math/_core.py

"""Implementation of the ``immlib.math`` common numerical namespace.

``immlib.math`` is a small, deliberately common subset of NumPy/PyTorch
functionality that operates directly on ``immlib.Quantity`` objects (as well
as on plain NumPy arrays, PyTorch tensors, and Python numbers, each of which
is treated as a unit-less--``units=None``--quantity). The governing
principles are, in short:

  * the backend (NumPy or PyTorch) is selected per call: any tensor magnitude
    among the arguments selects PyTorch, otherwise NumPy is used;
  * every function returns an ``immlib.Quantity``, *except* a function whose
    natural result is a boolean or an index/count (the comparisons,
    ``any``/``all``, the boolean predicates, ``allclose``/``isclose``, and the
    index and count functions), which instead returns a plain NumPy array or
    PyTorch tensor, matching ordinary NumPy/PyTorch ergonomics for masks and
    indexing -- and except the linear-algebra factorizations (``svd``,
    ``pinv``, ``matrix_rank``, ``lstsq``) and ``einsum``, whose parts have
    different units and which therefore require a unit-less input and return
    plain arrays or tensors;
  * units are computed directly from each function's mathematical meaning
    (unit-preserving, unitless-required, exponentiated, etc.)--not by
    delegating a whole ``Quantity`` to ``np.foo``/``torch.foo`` dispatch,
    since Pint's own NumPy dispatch machinery does not understand immlib's
    ``None`` ("no units", as opposed to Pint's real ``dimensionless``) unit
    convention and fails outright for it;
  * the allocation functions that take an argument as their example
    (``zeros_like``, ``ones_like``, ``full_like``, ``empty_like`` and the
    random allocators) derive the backend, and a tensor's device, from that
    example and keep its units; ``empty_like``'s contents are undefined, and
    the random allocators exist only in PyTorch, so an array example is filled
    with ``numpy.random`` (the two backends are seeded separately);

and, governing every function here and every method of ``immlib.Quantity``,
two rules:

  * **Rule 1 (backend agreement).** A function must produce equal results
    for a NumPy array and a PyTorch tensor that are equal; only the type of
    the result differs, matching the type of the input. Where the two
    libraries disagree--in a default (``torch.std``'s Bessel correction), in
    a result shape (``torch.min``'s ``(values, indices)``), in strictness
    (``numpy.squeeze`` raises for an axis that is not of size 1, where
    ``torch.squeeze`` does nothing), or in the meaning of a name
    (``numpy.transpose`` reverses every axis; ``torch.transpose`` swaps
    two)--immlib picks one behavior and implements it for both backends.
  * **Rule 2 (gradients).** Given a choice of implementations, immlib never
    chooses one that breaks PyTorch's gradient tracking: every magnitude
    computation on a tensor uses ordinary, differentiable PyTorch operations
    directly on the tensor, never a NumPy round-trip, so that gradients flow
    through ``immlib.math`` and through ``Quantity``'s methods exactly as
    they would through the equivalent raw PyTorch code. The exception is a
    function whose operation is not differentiable in any implementation
    (the set operations, for instance), which is documented as such.

  * The behavior chosen under Rule 1 is **PyTorch's**: ``immlib.math``
    follows PyTorch's names, signatures, defaults and semantics, and the
    NumPy backend is made to comply. PyTorch's API is generally the smaller
    of the two, so meeting it with NumPy is a translation rather than a
    reimplementation, and a PyTorch user's expectations hold here. Where
    PyTorch accepts NumPy's spelling of an argument (``axis`` for ``dim``,
    ``keepdims`` for ``keepdim``), so does immlib--uniformly, for every
    function that has the argument, rather than reproducing the gaps in
    PyTorch's own alias coverage. A function that exists only in NumPy is
    provided when it is easy to implement for tensors, and documents what it
    does with them.
  * SciPy sparse magnitudes stay sparse wherever the operation preserves
    sparsity (elementwise functions that map 0 to 0, arithmetic, reshaping,
    and 2-D concatenation); a reduction along an axis (``sum``, ``mean``,
    ``min``, etc.) returns a dense NumPy array (or a NumPy scalar for a full
    reduction), as does a reduction of a sparse PyTorch tensor; and an
    operation whose result would be dense (e.g. ``exp``, ``cos``, or
    ``where``) raises ``TypeError`` rather than allocating a dense array
    implicitly (use ``immlib.to_dense`` first if that is what you want);
  * a name that means different things in the two libraries takes PyTorch's
    meaning, per Rule 1: ``equal`` is ``torch.equal``'s whole-array test,
    not NumPy's elementwise comparison, which is spelled ``eq`` (and
    ``not_equal``, ``less``, and the rest keep their elementwise meaning,
    which both libraries agree on).
"""

from __future__ import annotations

import builtins
import operator
from collections.abc import Sequence
from numbers import Number
from typing import TYPE_CHECKING, Any, Union, cast
from collections import namedtuple

import numpy as np
import pint
import scipy.sparse as sps
from docshare import docwrap

from ..util._numeric import torch, to_tensor
from ..util._core import unitregistry
from ..util._quantity import (Quantity, quant, mag, alike_units,
                               promote, is_quant)

if TYPE_CHECKING:
    import torch as _torch

#: The value an ``immlib.math`` function accepts: a quantity, an array, a
#: tensor, or a number (treated as a unit-less quantity).
QuantityLike = Union[Quantity, np.ndarray, '_torch.Tensor', Number, Sequence]
#: The plain boolean array or tensor a comparison or ``any``/``all`` returns.
BoolArray = Union[np.ndarray, '_torch.Tensor']


# Helpers ######################################################################

def _is_tensor_backed(*mags):
    return builtins.any(torch.is_tensor(m) for m in mags)

#: A sentinel for a `dim` argument that the caller did not give, used by the
#: functions whose own default for it is not ``None``.
_UNSET = object()

def _dimargs(fname, kwargs, *, dim=None, keepdim=False):
    """Returns ``(dim, keepdim)`` after applying the NumPy spellings of those
    arguments, ``axis`` and ``keepdims``.

    PyTorch accepts both spellings for most, but not all, of the functions
    that have these arguments; ``immlib.math`` accepts them for all of them,
    so that there is one rule rather than a table of exceptions. Giving both
    spellings of the same argument is an error, as is any other keyword.
    """
    if 'axis' in kwargs:
        if dim is not None and dim is not _UNSET:
            raise TypeError(
                f"immlib.math.{fname}: 'dim' and 'axis' are the same"
                f" argument; give only one of them")
        dim = kwargs.pop('axis')
    if 'keepdims' in kwargs:
        if keepdim is not False:
            raise TypeError(
                f"immlib.math.{fname}: 'keepdim' and 'keepdims' are the same"
                f" argument; give only one of them")
        keepdim = kwargs.pop('keepdims')
    if kwargs:
        k = next(iter(kwargs))
        raise TypeError(
            f"immlib.math.{fname}: unexpected keyword argument '{k}'")
    return (dim, keepdim)

def _one_dim(fname, dim):
    """Rejects a tuple/list `dim` for a function that reduces a single
    dimension only. PyTorch's ``prod`` and ``cumsum`` take one dimension, so
    (Rule 1) neither backend takes more than one here."""
    if isinstance(dim, (tuple, list)):
        raise TypeError(
            f"immlib.math.{fname}: only one dimension may be reduced at a"
            f" time (PyTorch's own {fname} does not accept several), so a"
            f" tuple or list 'dim' is not supported for either backend;"
            f" reduce one dimension at a time instead")
    return dim

def _require_unitless(q, fname):
    if q.units is not None:
        raise TypeError(
            f"immlib.math.{fname} requires a unit-less (units=None)"
            f" quantity; got units {q.units}")

def _as_real_units(q, units):
    """Returns the magnitude of `q` in `units`; a quantity whose units are
    ``None`` is treated as a bare, dimensionless value (as Pint treats a bare
    value). Raises ``pint.DimensionalityError`` for incompatible units."""
    if q.units is None:
        q = quant(q.m, 'dimensionless', ureg=unitregistry(q))
    elif q.units == units:
        return q.m
    return q.m_as(units)

def _align_units(a, b, fname):
    """Returns ``(mag_a, mag_b, units)`` for combining `a` and `b`.

    If both have units of ``None``, the result has units of ``None``.
    Otherwise, following Pint's treatment of bare values, the result has the
    units of the first argument that has real units, and a unit-less argument
    is treated as a dimensionless value. Raises ``pint.DimensionalityError``
    when the units cannot be combined.
    """
    a = quant(a)
    b = quant(b)
    au = a.units
    bu = b.units
    if au is None and bu is None:
        return (a.m, b.m, None)
    u = bu if au is None else au
    return (_as_real_units(a, u), _as_real_units(b, u), u)

def _reconcile_seq(seq, fname):
    """Returns ``(mags, units)`` for a sequence of quantities/arrays/tensors
    to be combined (``stack``/``concatenate``). If every element has units of
    ``None``, so does the result; otherwise the result has the units of the
    first element with real units, and every element is converted into those
    units (unit-less elements are treated as dimensionless, as in
    ``_align_units``).
    """
    quants = [quant(x) for x in seq]
    if not quants:
        raise ValueError(
            f"immlib.math.{fname}: cannot combine an empty sequence")
    u = next((q.units for q in quants if q.units is not None), None)
    if u is None:
        return ([q.m for q in quants], None)
    return ([_as_real_units(q, u) for q in quants], u)

def _reduce_mag(np_fn, torch_name, m, axis, keepdims, *,
                 np_kwargs=None, torch_kwargs=None):
    """Applies a NumPy-semantics reduction to the raw magnitude `m` (a NumPy
    array or PyTorch tensor), translating the NumPy-style `axis`/`keepdims`
    arguments into PyTorch's `dim`/`keepdim` (and, for a full reduction with
    `keepdims=True`, into an explicit all-axes `dim` tuple, since PyTorch's
    own no-`dim` reductions do not support `keepdim`).

    `torch_name` (a plain attribute-name string, not a resolved function) is
    looked up on `torch` only inside the tensor branch, so that calling this
    on a NumPy-backed quantity never touches `torch` at all--this matters
    because, when PyTorch is not installed, any attribute access on the
    ``torch`` placeholder object raises immediately, and that must not
    happen merely from *passing* a would-be ``torch.foo`` reference as an
    argument, only from actually needing it.
    """
    np_kwargs = {} if np_kwargs is None else np_kwargs
    torch_kwargs = {} if torch_kwargs is None else torch_kwargs
    if torch.is_tensor(m):
        torch_fn = getattr(torch, torch_name)
        if axis is None:
            if keepdims:
                dims = tuple(range(m.dim()))
                r = torch_fn(m, dim=dims, keepdim=True, **torch_kwargs)
            else:
                r = torch_fn(m, **torch_kwargs)
        else:
            r = torch_fn(m, dim=axis, keepdim=keepdims, **torch_kwargs)
        # A reduction of a sparse tensor is returned as a dense tensor.
        if r.layout != torch.strided:
            r = r.to_dense()
        return r
    if sps.issparse(m):
        return _sparse_reduce(torch_name, m, axis, keepdims, **np_kwargs)
    return np_fn(m, axis=axis, keepdims=keepdims, **np_kwargs)

def _sparse_axes(m, axis, fname):
    ndim = m.ndim
    if axis is None:
        return tuple(range(ndim))
    axes = axis if isinstance(axis, (tuple, list)) else (axis,)
    axes = tuple(sorted(set(int(ax) % ndim for ax in axes)))
    if 1 < len(axes) < ndim:
        raise TypeError(
            f"immlib.math.{fname}: reducing a sparse array along more than"
            f" one (but not every) axis is not supported")
    return axes

def _sparse_result(r, shape, axes, keepdims):
    """Converts a SciPy sparse reduction result into a dense NumPy result
    whose shape follows NumPy's rules for `axes` and `keepdims`."""
    if sps.issparse(r):
        r = r.toarray()
    r = np.asarray(r)
    if keepdims:
        newshape = tuple(1 if ii in axes else n for (ii,n) in enumerate(shape))
    else:
        newshape = tuple(n for (ii,n) in enumerate(shape) if ii not in axes)
    if newshape == ():
        return r.reshape(())[()]
    return r.reshape(newshape)

def _sparse_reduce(name, m, axis, keepdims, ddof=0):
    """Reduces the SciPy sparse array/matrix `m` without densifying it;
    `name` is the reduction's PyTorch-style name (``'sum'``, ``'amin'``,
    etc.). The result is dense (see ``_sparse_result``)."""
    axes = _sparse_axes(m, axis, name)
    full = len(axes) == m.ndim
    ax = None if full else axes[0]
    shape = m.shape
    count = 1
    for ii in axes:
        count *= shape[ii]
    if name == 'sum':
        r = m.sum(axis=ax)
    elif name == 'mean':
        r = m.mean(axis=ax)
    elif name == 'amin':
        r = m.min(axis=ax)
    elif name == 'amax':
        r = m.max(axis=ax)
    elif name in ('any', 'all'):
        nnz = (m != 0).sum(axis=ax)
        r = np.asarray(nnz) > 0 if name == 'any' else np.asarray(nnz) == count
    elif name in ('var', 'std'):
        s1 = np.asarray(m.sum(axis=ax))
        s2 = np.asarray(builtins.abs(m).power(2).sum(axis=ax))
        mu = s1 / count
        r = np.maximum((s2 - count * np.abs(mu)**2) / (count - ddof), 0)
        if name == 'std':
            r = np.sqrt(r)
    elif name == 'prod':
        r = _sparse_prod(m, ax)
    else:
        raise TypeError(f"immlib.math.{name}: unsupported for sparse arrays")
    return _sparse_result(r, shape, axes, keepdims)

def _sparse_prod(m, axis):
    if m.ndim != 2:
        raise TypeError(
            "immlib.math.prod: only 2-dimensional sparse arrays are supported")
    (nr, nc) = m.shape
    if axis is None:
        c = m.tocoo(copy=True)
        c.sum_duplicates()
        if c.nnz < nr * nc:
            return np.zeros((), dtype=c.dtype)[()]
        return np.prod(c.data)
    c = m.tocsc(copy=True) if axis == 0 else m.tocsr(copy=True)
    c.sum_duplicates()
    n = nr if axis == 0 else nc
    counts = np.diff(c.indptr)
    starts = c.indptr[:-1]
    r = np.ones(len(counts), dtype=c.dtype)
    nonempty = counts > 0
    if nonempty.any():
        r[nonempty] = np.multiply.reduceat(c.data, starts[nonempty])
    r[counts < n] = 0
    return r

def _sparse_dense_error(fname):
    return TypeError(
        f"immlib.math.{fname} does not support SciPy sparse arrays because"
        f" its result would not be sparse; convert the argument with"
        f" immlib.to_dense first")

def _sparse_elementwise(fname, m, method):
    """Applies the sparsity-preserving SciPy method `method` to `m`, or raises
    ``TypeError`` if SciPy has no such method (i.e., the result would be
    dense)."""
    fn = getattr(m, method, None) if method is not None else None
    if fn is None:
        raise _sparse_dense_error(fname)
    return fn()

def _unitless_elementwise(fname, np_fn, torch_name, a):
    # See _reduce_mag's docstring regarding why `torch_name` is a string,
    # looked up lazily, rather than an already-resolved `torch.foo`.
    a = quant(a)
    _require_unitless(a, fname)
    m = a.m
    if torch.is_tensor(m):
        rmag = getattr(torch, torch_name)(m)
    elif sps.issparse(m):
        rmag = _sparse_elementwise(fname, m, np_fn.__name__)
    else:
        rmag = np_fn(m)
    return quant(rmag, None)


# Shared documentation #########################################################
# The functions in this module take a small, repeated vocabulary of
# arguments, so each argument is documented once, here, and the functions
# inherit the descriptions of the arguments they actually have (docshare
# inherits only those). The prototypes below are documentation and nothing
# else: they are never called.

def _doc_params(a, b, y, x, cond, seq, elements, test_elements, index, mask,
                weights, q, repeats, shifts, dims, dim0, dim1, shape,
                start_dim, end_dim, decimals, keepdim, correction,
                descending, stable, interpolation, sorted, return_inverse,
                return_counts, assume_unique, invert, as_tuple, min, max,
                split_size_or_sections, chunks, pad, mode, value):
    """The arguments of ``immlib.math``'s functions, documented once.

    Parameters
    ----------
    a : quantity or array or tensor or number
        The value to operate on. Anything that is not already an
        ``immlib.Quantity`` is treated as one with no units (see
        ``immlib.quant``), and the backend follows the magnitude: a PyTorch
        tensor selects PyTorch and anything else selects NumPy.
    b : quantity or array or tensor or number
        The second value, treated as `a` is.
    y : quantity or array or tensor or number
        The numerator, treated as `a` is.
    x : quantity or array or tensor or number
        The denominator, treated as `a` is.
    cond : array or tensor of bool
        The condition, elementwise: a plain boolean array or tensor, or a
        unit-less quantity of one.
    seq : sequence of quantity or array or tensor
        The values to combine. They need not have the same units, but they
        must be dimensionally compatible.
    elements : quantity or array or tensor or number
        The values to look up, treated as `a` is.
    test_elements : quantity or array or tensor or number
        The values to look for `elements` among, treated as `a` is.
    index : array or tensor of int
        The indices to take, as a plain array or tensor of integers, or a
        unit-less quantity of them. An index in either backend's spelling
        is accepted and converted.
    mask : array or tensor of bool
        Which elements to take, elementwise.
    weights : array or tensor or number, optional
        The weight of each element, which must be unit-less and must
        broadcast against `a`. The default, ``None``, weights them equally.
    q : number or array or tensor
        The quantile or quantiles to compute.
    repeats : int or array or tensor of int
        How many times to repeat each element.
    shifts : int or tuple of int
        How far to shift the elements.
    dims : int or tuple of int, optional
        The dimensions to operate on.
    dim0 : int
        The first of the two dimensions to exchange.
    dim1 : int
        The second of the two dimensions to exchange.
    shape : int or tuple of int
        The shape of the result, given either as a tuple or as separate
        arguments.
    start_dim : int, optional
        The first dimension to flatten; the default is ``0``.
    end_dim : int, optional
        The last dimension to flatten, inclusive; the default is ``-1``.
    decimals : int, optional
        The number of decimal places to round to; the default is ``0``.
        PyTorch names this argument, and it may also be given positionally,
        as in NumPy.
    keepdim : bool, optional
        Whether the reduced dimensions are kept, with length 1, rather than
        removed. The default is ``False``. ``keepdims`` is accepted as an
        alias, as PyTorch accepts it for some of its own functions.
    correction : int, optional
        The difference between the number of elements and the denominator's
        degrees of freedom. The default is ``1``, a Bessel-corrected sample
        statistic, as in PyTorch; ``numpy``'s ``ddof`` defaults to ``0``
        instead, which is ``correction=0`` here.
    descending : bool, optional
        Whether to sort from largest to smallest rather than smallest to
        largest. The default is ``False``.
    stable : bool, optional
        Whether equal elements keep their original order. The default is
        ``False``.
    interpolation : str, optional
        How to compute a quantile that falls between two elements:
        ``'linear'`` (the default), ``'lower'``, ``'higher'``,
        ``'nearest'``, or ``'midpoint'``.
    sorted : bool, optional
        Accepted for ``torch.unique``'s sake. The distinct values are
        always returned in order, as ``numpy.unique`` returns them.
    return_inverse : bool, optional
        Whether to also return, for each element of the input, its index
        among the distinct values. The default is ``False``.
    return_counts : bool, optional
        Whether to also return how many times each distinct value occurs.
        The default is ``False``.
    assume_unique : bool, optional
        Whether the inputs may be assumed to contain no repeated values,
        which makes the operation faster. The default is ``False``.
    invert : bool, optional
        Whether to return the negation of the usual result. The default is
        ``False``.
    as_tuple : bool, optional
        Whether to return one index per dimension, as ``numpy.nonzero``
        does, rather than a single array of index rows, as
        ``torch.nonzero`` does. The default is ``False``.
    min : quantity or array or tensor or number, optional
        The lower bound, unit-aligned with `a` as in ``maximum``. The
        default, ``None``, applies no lower bound.
    max : quantity or array or tensor or number, optional
        The upper bound, treated as `min` is.
    split_size_or_sections : int or sequence of int
        The size of each piece, or the size of every piece individually.
    chunks : int
        The greatest number of pieces to cut the argument into.
    pad : sequence of int
        The amounts to pad with, as ``(before, after)`` for the last
        dimension, then for the second-to-last, and so on--PyTorch's
        spelling rather than NumPy's.
    mode : str, optional
        How to fill the padding: ``'constant'`` (the default),
        ``'reflect'``, ``'replicate'`` or ``'circular'``.
    value : quantity or number, optional
        The element to pad a constant padding with; the default is zero. A
        bare number is in `a`'s own units.
    """
    raise NotImplementedError("_doc_params is documentation, not a function")

def _doc_dim_reduce(dim):
    """The dimension argument of ``immlib.math``'s reductions.

    Parameters
    ----------
    dim : int or tuple of int, optional
        The dimension or dimensions to reduce. The default, ``None``,
        reduces every dimension, giving a single value. ``axis`` is
        accepted as an alias, as PyTorch accepts it for most of its own
        reductions.
    """
    raise NotImplementedError("_doc_dim_reduce is documentation")

def _doc_dim_along(dim):
    """The dimension argument of ``immlib.math``'s non-reducing functions.

    Parameters
    ----------
    dim : int, optional
        The dimension to operate along. ``axis`` is accepted as an alias,
        as PyTorch accepts it for most of its own functions.
    """
    raise NotImplementedError("_doc_dim_along is documentation")

def _doc_returns_quantity(a):
    """The usual result of an ``immlib.math`` function.

    Returns
    -------
    immlib.Quantity
        The result, whose magnitude is a NumPy array for a NumPy argument
        and a PyTorch tensor for a tensor argument, and whose units are
        described above.
    """
    raise NotImplementedError("_doc_returns_quantity is documentation")

def _doc_returns_bool(a):
    """The result of an ``immlib.math`` function that answers a question.

    Returns
    -------
    array or tensor of bool
        A plain NumPy array or PyTorch tensor of booleans--not an
        ``immlib.Quantity``--so that it can be used directly as a mask.
    """
    raise NotImplementedError("_doc_returns_bool is documentation")

def _doc_returns_indices(a):
    """The result of an ``immlib.math`` function that answers with indices.

    Returns
    -------
    array or tensor of int
        A plain NumPy array or PyTorch tensor of integer indices--not an
        ``immlib.Quantity``--so that it can be used directly for indexing.
    """
    raise NotImplementedError("_doc_returns_indices is documentation")


# Elementwise arithmetic #######################################################

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def abs(a: QuantityLike) -> Quantity:
    """Returns the elementwise absolute value of `a`, preserving units."""
    return builtins.abs(quant(a))  # type: ignore[return-value]

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def add(a: QuantityLike, b: QuantityLike) -> Quantity:
    """Returns ``a + b``; see ``immlib.Quantity``'s unit-aware addition."""
    return quant(a) + quant(b)  # type: ignore[return-value]

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def subtract(a: QuantityLike, b: QuantityLike) -> Quantity:
    """Returns ``a - b``; see ``immlib.Quantity``'s unit-aware subtraction."""
    return quant(a) - quant(b)  # type: ignore[return-value]

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def multiply(a: QuantityLike, b: QuantityLike) -> Quantity:
    """Returns ``a * b``; see ``immlib.Quantity``'s unit-aware multiplication."""
    return quant(a) * quant(b)  # type: ignore[return-value]

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def divide(a: QuantityLike, b: QuantityLike) -> Quantity:
    """Returns ``a / b``; see ``immlib.Quantity``'s unit-aware division."""
    return quant(a) / quant(b)  # type: ignore[return-value]

true_divide = divide

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def pow(a: QuantityLike, b: QuantityLike) -> Quantity:
    """Returns ``a ** b``; see ``immlib.Quantity.__pow__``. `b` must be a
    plain number or a unit-less quantity unless `a` is itself unit-less.

    This is PyTorch's name for the operation; NumPy calls it ``power``,
    which PyTorch does not define and which is therefore not defined here.
    """
    return quant(a) ** b

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def negative(a: QuantityLike) -> Quantity:
    """Returns ``-a``, preserving units."""
    return -quant(a)  # type: ignore[return-value]

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def positive(a: QuantityLike) -> Quantity:
    """Returns ``+a``, preserving units."""
    return +quant(a)  # type: ignore[return-value]


# Comparisons ###################################################################
# Each of these returns a plain NumPy array or PyTorch tensor of bool, not an
# immlib.Quantity--see the module docstring.

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_bool)
def eq(a: QuantityLike, b: QuantityLike) -> BoolArray:
    """Returns the elementwise result of ``a == b`` as a plain bool array or
    tensor.

    The comparison is unit-aware: differing but compatible units are
    converted first, and incompatible units compare unequal (an all-false
    result) rather than raising. A unit-less (``units=None``) value is
    compared as a bare value would be, i.e., as a dimensionless value, so it
    is unequal to a quantity with real dimensions (except that, as in Pint,
    an all-zero or NaN bare value is compared by magnitude alone).

    ``eq`` is PyTorch's name for the elementwise comparison; note that
    ``equal``, in PyTorch and here, is the whole-array test instead.
    """
    return quant(a) == quant(b)

@docwrap(format='numpy', inheritparams=_doc_params)
def equal(a: QuantityLike, b: QuantityLike) -> bool:
    """Returns whether `a` and `b` have the same shape and equal elements,
    as a single ``bool``.

    This is ``torch.equal``'s meaning rather than ``numpy.equal``'s: the
    elementwise comparison is ``eq``. The one departure from
    ``torch.equal`` is that a difference of dtype alone does not make two
    otherwise-equal arguments unequal, since ``numpy`` has no such rule and
    the two backends must agree.

    Units are handled as in ``eq``, so quantities with compatible units are
    converted before comparing and incompatible ones are simply unequal.

    Parameters
    ----------

    Returns
    -------
    bool
        A single Python ``bool``, not an array of them: ``eq`` is the
        elementwise comparison.
    """
    r = eq(a, b)
    if np.shape(quant(a).m) != np.shape(quant(b).m):
        return False
    return builtins.bool(r.all())

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_bool)
def not_equal(a: QuantityLike, b: QuantityLike) -> BoolArray:
    """Returns the elementwise result of ``a != b``; see ``eq``."""
    return quant(a) != quant(b)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_bool)
def less(a: QuantityLike, b: QuantityLike) -> BoolArray:
    """Returns the elementwise result of ``a < b`` (unit-aware; raises for
    dimensionally incompatible real units, per ``immlib.Quantity``)."""
    return quant(a) < quant(b)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_bool)
def less_equal(a: QuantityLike, b: QuantityLike) -> BoolArray:
    """Returns the elementwise result of ``a <= b``; see ``less``."""
    return quant(a) <= quant(b)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_bool)
def greater(a: QuantityLike, b: QuantityLike) -> BoolArray:
    """Returns the elementwise result of ``a > b``; see ``less``."""
    return quant(a) > quant(b)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_bool)
def greater_equal(a: QuantityLike, b: QuantityLike) -> BoolArray:
    """Returns the elementwise result of ``a >= b``; see ``less``."""
    return quant(a) >= quant(b)

#: An alias of ``immlib.math.not_equal``, as in PyTorch.
ne = not_equal
#: An alias of ``immlib.math.less``, as in PyTorch.
lt = less
#: An alias of ``immlib.math.less_equal``, as in PyTorch.
le = less_equal
#: An alias of ``immlib.math.greater``, as in PyTorch.
gt = greater
#: An alias of ``immlib.math.greater_equal``, as in PyTorch.
ge = greater_equal

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def maximum(a: QuantityLike, b: QuantityLike) -> Quantity:
    """Returns the elementwise maximum of `a` and `b`.

    If either argument has real units, the result has the units of the first
    such argument, the other argument is converted into them, and a unit-less
    argument is treated as a dimensionless value (so ``maximum(quant(x),
    quant(y, 'm'))`` raises ``pint.DimensionalityError``, just as
    ``numpy.maximum(x, quant(y, 'm'))`` does).
    """
    (ma, mb, u) = _align_units(a, b, 'maximum')
    if _is_tensor_backed(ma, mb):
        (ma, mb) = promote(ma, mb)
        rmag = torch.maximum(ma, mb)
    elif sps.issparse(ma):
        rmag = ma.maximum(mb)
    elif sps.issparse(mb):
        rmag = mb.maximum(ma)
    else:
        rmag = np.maximum(ma, mb)
    return quant(rmag, u)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def minimum(a: QuantityLike, b: QuantityLike) -> Quantity:
    """Returns the elementwise minimum of `a` and `b`; see ``maximum``."""
    (ma, mb, u) = _align_units(a, b, 'minimum')
    if _is_tensor_backed(ma, mb):
        (ma, mb) = promote(ma, mb)
        rmag = torch.minimum(ma, mb)
    elif sps.issparse(ma):
        rmag = ma.minimum(mb)
    elif sps.issparse(mb):
        rmag = mb.minimum(ma)
    else:
        rmag = np.minimum(ma, mb)
    return quant(rmag, u)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def where(cond: Any, a: QuantityLike, b: QuantityLike) -> Quantity:
    """Returns `a` where `cond` is true and `b` otherwise, elementwise;
    `cond` must be a plain (or unit-less-quantity) boolean array/tensor, and
    `a`/`b` are unit-aligned as in ``maximum``."""
    condm = mag(cond, None) if is_quant(cond) else cond
    (ma, mb, u) = _align_units(a, b, 'where')
    if builtins.any(sps.issparse(x) for x in (condm, ma, mb)):
        raise _sparse_dense_error('where')
    if _is_tensor_backed(condm, ma, mb):
        (condm, ma, mb) = promote(condm, ma, mb)
        if condm.dtype is not torch.bool:
            condm = condm.bool()
        rmag = torch.where(condm, ma, mb)
    else:
        rmag = np.where(condm, ma, mb)
    return quant(rmag, u)


# Elementary functions ##########################################################

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def sqrt(a: QuantityLike) -> Quantity:
    """Returns the elementwise square root of `a`; the result's units are
    `a`'s units raised to the 1/2 power (e.g. ``sqrt(4 m**2) == 2 m``)."""
    a = quant(a)
    m = a.m
    if torch.is_tensor(m):
        rmag = torch.sqrt(m)
    elif sps.issparse(m):
        rmag = _sparse_elementwise('sqrt', m, 'sqrt')
    else:
        rmag = np.sqrt(m)
    u = None if a.units is None else (a.units ** 0.5)
    return quant(rmag, u)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def exp(a: QuantityLike) -> Quantity:
    """Returns the elementwise exponential of `a`; `a` must be unit-less."""
    return _unitless_elementwise('exp', np.exp, 'exp', a)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def log(a: QuantityLike) -> Quantity:
    """Returns the elementwise natural log of `a`; `a` must be unit-less."""
    return _unitless_elementwise('log', np.log, 'log', a)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def log10(a: QuantityLike) -> Quantity:
    """Returns the elementwise base-10 log of `a`; `a` must be unit-less."""
    return _unitless_elementwise('log10', np.log10, 'log10', a)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def sin(a: QuantityLike) -> Quantity:
    """Returns the elementwise sine of `a`; `a` must be unit-less."""
    return _unitless_elementwise('sin', np.sin, 'sin', a)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def cos(a: QuantityLike) -> Quantity:
    """Returns the elementwise cosine of `a`; `a` must be unit-less."""
    return _unitless_elementwise('cos', np.cos, 'cos', a)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def tan(a: QuantityLike) -> Quantity:
    """Returns the elementwise tangent of `a`; `a` must be unit-less."""
    return _unitless_elementwise('tan', np.tan, 'tan', a)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def arcsin(a: QuantityLike) -> Quantity:
    """Returns the elementwise arcsine of `a`; `a` must be unit-less."""
    return _unitless_elementwise('arcsin', np.arcsin, 'asin', a)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def arccos(a: QuantityLike) -> Quantity:
    """Returns the elementwise arccosine of `a`; `a` must be unit-less."""
    return _unitless_elementwise('arccos', np.arccos, 'acos', a)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def arctan(a: QuantityLike) -> Quantity:
    """Returns the elementwise arctangent of `a`; `a` must be unit-less."""
    return _unitless_elementwise('arctan', np.arctan, 'atan', a)

#: An alias of ``immlib.math.arcsin``, as in PyTorch.
asin = arcsin
#: An alias of ``immlib.math.arccos``, as in PyTorch.
acos = arccos
#: An alias of ``immlib.math.arctan``, as in PyTorch.
atan = arctan

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def arctan2(y: QuantityLike, x: QuantityLike) -> Quantity:
    """Returns the elementwise ``arctan2(y, x)``; the units of `y` and `x`
    are aligned as in ``maximum``, and the angle result is always unit-less.
    """
    (my, mx, _u) = _align_units(y, x, 'arctan2')
    if _is_tensor_backed(my, mx):
        (my, mx) = promote(my, mx)
        rmag = torch.atan2(my, mx)
    else:
        rmag = np.arctan2(my, mx)
    return quant(rmag, None)

#: An alias of ``immlib.math.arctan2``, as in PyTorch.
atan2 = arctan2

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def floor(a: QuantityLike) -> Quantity:
    """Returns the elementwise floor of `a`, preserving units."""
    a = quant(a)
    m = a.m
    if torch.is_tensor(m):
        rmag = torch.floor(m)
    elif sps.issparse(m):
        rmag = _sparse_elementwise('floor', m, 'floor')
    else:
        rmag = np.floor(m)
    return quant(rmag, a.units)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def ceil(a: QuantityLike) -> Quantity:
    """Returns the elementwise ceiling of `a`, preserving units."""
    a = quant(a)
    m = a.m
    if torch.is_tensor(m):
        rmag = torch.ceil(m)
    elif sps.issparse(m):
        rmag = _sparse_elementwise('ceil', m, 'ceil')
    else:
        rmag = np.ceil(m)
    return quant(rmag, a.units)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def round(a: QuantityLike, decimals: int=0) -> Quantity:
    """Returns `a` elementwise-rounded to `decimals` decimal places (default
    0), preserving units. The argument is named as ``torch.round`` names it,
    though it may be given positionally here, as in NumPy."""
    a = quant(a)
    m = a.m
    if torch.is_tensor(m):
        rmag = torch.round(m, decimals=decimals)
    elif sps.issparse(m):
        rmag = m.tocsr(copy=True)
        rmag.data = np.round(rmag.data, decimals=decimals)
        rmag.eliminate_zeros()
    else:
        rmag = np.round(m, decimals=decimals)
    return quant(rmag, a.units)


# Reductions #####################################################################

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_reduce), inheritreturns=_doc_returns_quantity)
def sum(a: QuantityLike, dim: Any=None, keepdim: bool=False, **kwargs: Any) -> Quantity:
    """Returns the sum of `a`'s elements (optionally along `dim`),
    preserving units."""
    (dim, keepdim) = _dimargs('sum', kwargs, dim=dim, keepdim=keepdim)
    a = quant(a)
    rmag = _reduce_mag(np.sum, 'sum', a.m, dim, keepdim)
    return quant(rmag, a.units)

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_reduce), inheritreturns=_doc_returns_quantity)
def prod(a: QuantityLike, dim: Any=None, keepdim: bool=False, **kwargs: Any) -> Quantity:
    """Returns the product of `a`'s elements (optionally along `dim`); the
    result's units are `a`'s units raised to the power of the number of
    elements combined into each output value (e.g. the product of 3
    quantities in meters has units of ``m**3``). Only one dimension may be
    reduced at a time, for either backend, since PyTorch's own ``prod``
    accepts only one.
    """
    (dim, keepdim) = _dimargs('prod', kwargs, dim=dim, keepdim=keepdim)
    axis = _one_dim('prod', dim)
    keepdims = keepdim
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        rmag = _sparse_reduce('prod', m, axis, keepdims)
    elif torch.is_tensor(m):
        if axis is None:
            if keepdims:
                dims = tuple(range(m.dim()))
                rmag = m
                for d in sorted(dims, reverse=True):
                    rmag = torch.prod(rmag, dim=d, keepdim=True)
            else:
                rmag = torch.prod(m)
        else:
            rmag = torch.prod(m, dim=axis, keepdim=keepdims)
    else:
        # numpy's stubs type `keepdims` as a Literal, so a run-time `bool`
        # matches no overload; cast it away (the value is what numpy wants).
        rmag = np.prod(m, axis=axis, keepdims=cast(Any, keepdims))
    if a.units is None:
        u = None
    else:
        shape = m.shape
        if axis is None:
            count = 1
            for s in shape:
                count *= s
        elif isinstance(axis, (tuple, list)):
            count = 1
            for ax in axis:
                count *= shape[ax]
        else:
            count = shape[axis]
        u = a.units ** int(count)
    return quant(rmag, u)

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_reduce), inheritreturns=_doc_returns_quantity)
def mean(a: QuantityLike, dim: Any=None, keepdim: bool=False, **kwargs: Any) -> Quantity:
    """Returns the mean of `a`'s elements (optionally along `dim`),
    preserving units."""
    (dim, keepdim) = _dimargs('mean', kwargs, dim=dim, keepdim=keepdim)
    a = quant(a)
    rmag = _reduce_mag(np.mean, 'mean', a.m, dim, keepdim)
    return quant(rmag, a.units)

#: The result of ``immlib.math.min`` when a dimension is given: the minimum
#: values, as an ``immlib.Quantity``, and the index of the first minimum
#: along that dimension, as a plain array or tensor of integers.
min_result = namedtuple('min', ('values', 'indices'))  # type: ignore[name-match]
#: The result of ``immlib.math.max`` when a dimension is given; see
#: ``immlib.math.min_result``.
max_result = namedtuple('max', ('values', 'indices'))  # type: ignore[name-match]

def _minmax(fname, a, dim, keepdim, kwargs):
    (dim, keepdim) = _dimargs(fname, kwargs, dim=dim, keepdim=keepdim)
    amin_amax = 'amin' if fname == 'min' else 'amax'
    argfn = np.argmin if fname == 'min' else np.argmax
    a = quant(a)
    if dim is None:
        rmag = _reduce_mag(getattr(np, amin_amax), amin_amax, a.m, None,
                           keepdim)
        return quant(rmag, a.units)
    dim = _one_dim(fname, dim)
    m = a.m
    if torch.is_tensor(m):
        r = getattr(torch, fname)(m, dim=dim, keepdim=keepdim)
        (vals, idcs) = (r.values, r.indices)
    else:
        if sps.issparse(m):
            raise _sparse_dense_error(fname)
        idcs = argfn(m, axis=dim)
        vals = np.take_along_axis(m, np.expand_dims(idcs, dim), axis=dim)
        if keepdim:
            idcs = np.expand_dims(idcs, dim)
        else:
            vals = np.squeeze(vals, axis=dim)
    cls = min_result if fname == 'min' else max_result
    return cls(quant(vals, a.units), idcs)

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_reduce))
def min(a: QuantityLike, dim: Any=None, keepdim: bool=False, **kwargs: Any) -> Any:
    """Returns the minimum of `a`'s elements, preserving units.

    ``min(a)`` returns the smallest element of `a` as an
    ``immlib.Quantity``. ``min(a, dim)`` returns a ``(values, indices)``
    named tuple, as ``torch.min`` does: `values` is an ``immlib.Quantity``
    of the minima along `dim` and `indices` is a plain array or tensor
    giving, for each of them, the index of the first minimal element along
    `dim`.

    Use ``amin`` for the values alone, and ``minimum`` for the elementwise
    minimum of two arguments (as in PyTorch, whose own two-argument ``min``
    is deprecated in favor of ``torch.minimum``).

    Parameters
    ----------

    Returns
    -------
    immlib.Quantity or min_result
        The smallest element, as a quantity, when `dim` is not given; a
        ``(values, indices)`` named tuple when it is, whose `values` is a
        quantity and whose `indices` is a plain array or tensor of
        integers.
    """
    return _minmax('min', a, dim, keepdim, kwargs)

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_reduce))
def max(a: QuantityLike, dim: Any=None, keepdim: bool=False, **kwargs: Any) -> Any:
    """Returns the maximum of `a`'s elements, preserving units; see
    ``min``, whose behavior this mirrors.

    Parameters
    ----------

    Returns
    -------
    immlib.Quantity or max_result
        The largest element, as a quantity, when `dim` is not given; a
        ``(values, indices)`` named tuple when it is.
    """
    return _minmax('max', a, dim, keepdim, kwargs)

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_reduce), inheritreturns=_doc_returns_quantity)
def amin(a: QuantityLike, dim: Any=None, keepdim: bool=False, **kwargs: Any) -> Quantity:
    """Returns the minimum of `a`'s elements (optionally along `dim`),
    preserving units, as an ``immlib.Quantity``--never the ``(values,
    indices)`` tuple that ``min`` returns for a given `dim`. Unlike ``min``,
    several dimensions may be reduced at once."""
    (dim, keepdim) = _dimargs('amin', kwargs, dim=dim, keepdim=keepdim)
    a = quant(a)
    return quant(_reduce_mag(np.amin, 'amin', a.m, dim, keepdim), a.units)

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_reduce), inheritreturns=_doc_returns_quantity)
def amax(a: QuantityLike, dim: Any=None, keepdim: bool=False, **kwargs: Any) -> Quantity:
    """Returns the maximum of `a`'s elements (optionally along `dim`),
    preserving units; see ``amin``."""
    (dim, keepdim) = _dimargs('amax', kwargs, dim=dim, keepdim=keepdim)
    a = quant(a)
    return quant(_reduce_mag(np.amax, 'amax', a.m, dim, keepdim), a.units)

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_reduce), inheritreturns=_doc_returns_bool)
def any(a: QuantityLike, dim: Any=None, keepdim: bool=False, **kwargs: Any) -> BoolArray:
    """Returns whether any of `a`'s elements are truthy (optionally along
    `dim`), as a plain bool array or tensor (not an ``immlib.Quantity``)."""
    (dim, keepdim) = _dimargs('any', kwargs, dim=dim, keepdim=keepdim)
    a = quant(a)
    return _reduce_mag(np.any, 'any', a.m, dim, keepdim)

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_reduce), inheritreturns=_doc_returns_bool)
def all(a: QuantityLike, dim: Any=None, keepdim: bool=False, **kwargs: Any) -> BoolArray:
    """Returns whether all of `a`'s elements are truthy (optionally along
    `dim`), as a plain bool array or tensor (not an ``immlib.Quantity``)."""
    (dim, keepdim) = _dimargs('all', kwargs, dim=dim, keepdim=keepdim)
    a = quant(a)
    return _reduce_mag(np.all, 'all', a.m, dim, keepdim)

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_reduce), inheritreturns=_doc_returns_quantity)
def std(a: QuantityLike, dim: Any=None, keepdim: bool=False, correction: int=1, **kwargs: Any) -> Quantity:
    """Returns the standard deviation of `a`'s elements (optionally along
    `dim`), preserving units.

    `correction` is the difference between the number of elements and the
    denominator's degrees of freedom, and defaults to ``1``--a
    Bessel-corrected sample standard deviation, as in ``torch.std``. This
    differs from ``numpy.std``, whose ``ddof`` defaults to ``0``; pass
    ``correction=0`` for a population standard deviation. The NumPy backend
    is given the same correction, so both backends agree.
    """
    (dim, keepdim) = _dimargs('std', kwargs, dim=dim, keepdim=keepdim)
    a = quant(a)
    rmag = _reduce_mag(np.std, 'std', a.m, dim, keepdim,
                        np_kwargs={'ddof': correction},
                        torch_kwargs={'correction': correction})
    return quant(rmag, a.units)

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_reduce), inheritreturns=_doc_returns_quantity)
def var(a: QuantityLike, dim: Any=None, keepdim: bool=False, correction: int=1, **kwargs: Any) -> Quantity:
    """Returns the variance of `a`'s elements (optionally along `dim`); the
    result's units are `a`'s units squared. See ``std`` regarding
    `correction`, which defaults to ``1`` here as it does in PyTorch.
    """
    (dim, keepdim) = _dimargs('var', kwargs, dim=dim, keepdim=keepdim)
    a = quant(a)
    rmag = _reduce_mag(np.var, 'var', a.m, dim, keepdim,
                        np_kwargs={'ddof': correction},
                        torch_kwargs={'correction': correction})
    u = None if a.units is None else (a.units ** 2)
    return quant(rmag, u)

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_along), inheritreturns=_doc_returns_quantity)
def cumsum(a: QuantityLike, dim: Any, **kwargs: Any) -> Quantity:
    """Returns the cumulative sum of `a`'s elements along `dim`, preserving
    units. `dim` is required, as it is in ``torch.cumsum``."""
    (dim, _) = _dimargs('cumsum', kwargs, dim=dim)
    dim = _one_dim('cumsum', dim)
    if dim is None:
        raise TypeError("immlib.math.cumsum: 'dim' is required")
    a = quant(a)
    m = a.m
    if torch.is_tensor(m):
        rmag = torch.cumsum(m, dim=dim)
    elif sps.issparse(m):
        raise _sparse_dense_error('cumsum')
    else:
        rmag = np.cumsum(m, axis=dim)
    return quant(rmag, a.units)


# Shape / combination ############################################################

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def reshape(a: QuantityLike, *shape: Any) -> Quantity:
    """Returns `a` reshaped to `shape`, preserving units. The shape may be
    given as a single tuple or as separate arguments."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    a = quant(a)
    m = a.m
    if torch.is_tensor(m) or sps.issparse(m):
        rmag = m.reshape(shape)
    else:
        rmag = np.reshape(m, shape)
    return quant(rmag, a.units)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def transpose(a: QuantityLike, dim0: Any, dim1: Any) -> Quantity:
    """Returns `a` with the dimensions `dim0` and `dim1` exchanged,
    preserving units.

    This is ``torch.transpose``'s meaning, for both backends: it swaps two
    dimensions and takes both of them. ``numpy.transpose``'s meaning--a full
    permutation of the dimensions, reversing all of them by default--is
    ``permute``. ``swapaxes`` and ``swapdims`` are aliases of this function,
    as they are in PyTorch.
    """
    a = quant(a)
    m = a.m
    if torch.is_tensor(m):
        rmag = torch.transpose(m, dim0, dim1)
    elif sps.issparse(m):
        rmag = m.transpose()
    else:
        rmag = np.swapaxes(m, dim0, dim1)
    return quant(rmag, a.units)

#: An alias of ``immlib.math.transpose``, as in PyTorch.
swapaxes = transpose
#: An alias of ``immlib.math.transpose``, as in PyTorch.
swapdims = transpose

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def permute(a: QuantityLike, *dims: Any) -> Quantity:
    """Returns `a` with its dimensions permuted into the order `dims`,
    preserving units. The dimensions may be given as a single tuple or as
    separate arguments, and giving none reverses them, as
    ``numpy.transpose`` does.
    """
    if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
        dims = tuple(dims[0])
    a = quant(a)
    m = a.m
    if not dims:
        dims = tuple(reversed(range(np.ndim(m))))
    if torch.is_tensor(m):
        rmag = m.permute(*dims)
    elif sps.issparse(m):
        rmag = m.transpose(dims)
    else:
        rmag = np.transpose(m, dims)
    return quant(rmag, a.units)

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_along), inheritreturns=_doc_returns_quantity)
def squeeze(a: QuantityLike, dim: Any=None, **kwargs: Any) -> Quantity:
    """Returns `a` with size-1 dimensions removed, preserving units.

    All of them are removed when `dim` is not given; otherwise only the
    given dimension or dimensions are, and one that is not of size 1 is left
    alone. That last part is ``torch.squeeze``'s behavior, and applies to
    both backends: ``numpy.squeeze`` raises a ``ValueError`` instead, which
    would make the same call succeed for a tensor and fail for an array.
    """
    (dim, _) = _dimargs('squeeze', kwargs, dim=dim)
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        raise TypeError(
            "immlib.math.squeeze: SciPy sparse arrays are not supported")
    if dim is None:
        rmag = m.squeeze() if torch.is_tensor(m) else np.squeeze(m)
        return quant(rmag, a.units)
    dims = dim if isinstance(dim, (tuple, list)) else (dim,)
    ndim = np.ndim(m)
    # Only the size-1 dimensions are dropped; the rest are left alone.
    dims = tuple(d for d in (int(x) % ndim for x in dims)
                 if np.shape(m)[d] == 1)
    if torch.is_tensor(m):
        rmag = m
        for d in sorted(dims, reverse=True):
            rmag = rmag.squeeze(d)
    else:
        rmag = np.squeeze(m, axis=dims) if dims else m
    return quant(rmag, a.units)

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_along), inheritreturns=_doc_returns_quantity)
def unsqueeze(a: QuantityLike, dim: Any, **kwargs: Any) -> Quantity:
    """Returns `a` with a new size-1 dimension inserted at `dim`, preserving
    units. This is ``torch.unsqueeze``; ``numpy.expand_dims`` is the same
    operation under another name."""
    (dim, _) = _dimargs('unsqueeze', kwargs, dim=dim)
    a = quant(a)
    m = a.m
    if torch.is_tensor(m):
        rmag = torch.unsqueeze(m, dim)
    elif sps.issparse(m):
        raise _sparse_dense_error('unsqueeze')
    else:
        rmag = np.expand_dims(m, dim)
    return quant(rmag, a.units)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def ravel(a: QuantityLike) -> Quantity:
    """Returns `a` flattened into one dimension, preserving units."""
    a = quant(a)
    m = a.m
    if torch.is_tensor(m):
        rmag = torch.ravel(m)
    elif sps.issparse(m):
        raise _sparse_dense_error('ravel')
    else:
        rmag = np.ravel(m)
    return quant(rmag, a.units)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def flatten(a: QuantityLike, start_dim: int=0, end_dim: int=-1) -> Quantity:
    """Returns `a` with the dimensions from `start_dim` through `end_dim`
    (inclusive) flattened into one, preserving units. This is
    ``torch.flatten``, which flattens every dimension by default; see
    ``ravel`` for the simpler always-everything form."""
    a = quant(a)
    m = a.m
    if torch.is_tensor(m):
        rmag = torch.flatten(m, start_dim, end_dim)
    elif sps.issparse(m):
        raise _sparse_dense_error('flatten')
    else:
        shape = np.shape(m)
        ndim = len(shape)
        s = int(start_dim) % ndim if ndim else 0
        e = int(end_dim) % ndim if ndim else 0
        if e < s:
            raise ValueError(
                "immlib.math.flatten: end_dim must not precede start_dim")
        n = 1
        for k in shape[s:e+1]:
            n *= k
        rmag = np.reshape(m, shape[:s] + (n,) + shape[e+1:])
    return quant(rmag, a.units)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def conj(a: QuantityLike) -> Quantity:
    """Returns the elementwise complex conjugate of `a`, preserving units."""
    a = quant(a)
    m = a.m
    if torch.is_tensor(m):
        rmag = torch.conj(m)
    elif sps.issparse(m):
        rmag = m.conj()
    else:
        rmag = np.conj(m)
    return quant(rmag, a.units)

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_along), inheritreturns=_doc_returns_quantity)
def stack(seq: QuantityLike, dim: Any=_UNSET, **kwargs: Any) -> Quantity:
    """Returns the quantities/arrays/tensors in `seq` stacked along a new
    dimension `dim`. If any element has real units, the result has the units
    of the first such element, and all elements are converted into them
    (unit-less elements are treated as dimensionless values, as in
    ``maximum``)."""
    (dim, _) = _dimargs('stack', kwargs, dim=dim)
    axis = 0 if dim is None or dim is _UNSET else dim
    (mags, u) = _reconcile_seq(seq, 'stack')
    if builtins.any(sps.issparse(x) for x in mags):
        raise TypeError(
            "immlib.math.stack: SciPy sparse arrays are not supported; use"
            " cat for 2-D sparse arrays")
    if _is_tensor_backed(*mags):
        mags = promote(*mags)
        rmag = torch.stack(mags, dim=axis)
    else:
        rmag = np.stack(mags, axis=axis)
    return quant(rmag, u)

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_along), inheritreturns=_doc_returns_quantity)
def cat(seq: QuantityLike, dim: Any=_UNSET, **kwargs: Any) -> Quantity:
    """Returns the quantities/arrays/tensors in `seq` concatenated along an
    existing dimension `dim`; see ``stack`` regarding units.
    ``concatenate`` and ``concat`` are aliases, as they are in PyTorch."""
    (dim, _) = _dimargs('cat', kwargs, dim=dim)
    axis = 0 if dim is None or dim is _UNSET else dim
    (mags, u) = _reconcile_seq(seq, 'cat')
    if builtins.any(sps.issparse(x) for x in mags):
        if _is_tensor_backed(*mags) or axis not in (0, 1, -1, -2) or (
                builtins.any(np.ndim(x) != 2 for x in mags)):
            raise TypeError(
                "immlib.math.cat: SciPy sparse arrays can only be"
                " concatenated with other 2-D NumPy or SciPy arrays along"
                " dimension 0 or 1")
        combine = sps.vstack if axis in (0, -2) else sps.hstack
        rmag = combine(mags)
        return quant(rmag, u)
    if _is_tensor_backed(*mags):
        mags = promote(*mags)
        rmag = torch.cat(mags, dim=axis)
    else:
        rmag = np.concatenate(mags, axis=axis)
    return quant(rmag, u)

#: An alias of ``immlib.math.cat``, as in PyTorch.
concatenate = cat
#: An alias of ``immlib.math.cat``, as in PyTorch.
concat = cat


# Sorting and order statistics ##################################################

#: The result of ``immlib.math.sort``: the sorted values, as an
#: ``immlib.Quantity``, and the index each value came from, as a plain array
#: or tensor of integers.
sort_result = namedtuple('sort', ('values', 'indices'))  # type: ignore[name-match]
#: The result of ``immlib.math.median`` when a dimension is given; see
#: ``immlib.math.sort_result``.
median_result = namedtuple('median', ('values', 'indices'))  # type: ignore[name-match]

def _signed(m):
    """Returns `m` in a dtype that can be negated, for a descending sort."""
    if isinstance(m, np.ndarray) and m.dtype.kind in 'ub':
        return m.astype(np.int64)
    return m

def _sort_indices(m, dim, descending, stable):
    """Returns the NumPy indices that sort `m` along `dim`."""
    kind = 'stable' if stable else None
    if descending:
        # Negating rather than reversing keeps equal elements in their
        # original order, as PyTorch's descending sort does.
        return np.argsort(-_signed(m), axis=dim, kind=kind)
    return np.argsort(m, axis=dim, kind=kind)

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_along))
def sort(a: QuantityLike, dim: Any=-1, descending: bool=False, stable: bool=False, **kwargs: Any) -> Any:
    """Returns `a`'s elements sorted along `dim`, preserving units.

    The result is a ``(values, indices)`` named tuple, as ``torch.sort``
    returns: `values` is an ``immlib.Quantity`` and `indices` is a plain
    array or tensor giving the index each value came from. Sorting is along
    the last dimension by default, and ascending unless `descending`;
    `stable` keeps equal elements in their original order.

    Parameters
    ----------

    Returns
    -------
    sort_result
        A ``(values, indices)`` named tuple, whose `values` is a quantity
        in `a`'s units and whose `indices` is a plain array or tensor of
        integers.
    """
    (dim, _) = _dimargs('sort', kwargs, dim=dim)
    dim = -1 if dim is None else dim
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        raise _sparse_dense_error('sort')
    if torch.is_tensor(m):
        r = torch.sort(m, dim=dim, descending=descending, stable=stable)
        (vals, idcs) = (r.values, r.indices)
    else:
        idcs = _sort_indices(m, dim, descending, stable)
        vals = np.take_along_axis(m, idcs, axis=dim)
    return sort_result(quant(vals, a.units), idcs)

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_along), inheritreturns=_doc_returns_indices)
def argsort(a: QuantityLike, dim: Any=-1, descending: bool=False, stable: bool=False, **kwargs: Any) -> Any:
    """Returns the indices that sort `a` along `dim`, as a plain array or
    tensor of integers; see ``sort``."""
    (dim, _) = _dimargs('argsort', kwargs, dim=dim)
    dim = -1 if dim is None else dim
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        raise _sparse_dense_error('argsort')
    if torch.is_tensor(m):
        return torch.argsort(m, dim=dim, descending=descending,
                             stable=stable)
    return _sort_indices(m, dim, descending, stable)

def _argminmax(fname, a, dim, keepdim, kwargs):
    (dim, keepdim) = _dimargs(fname, kwargs, dim=dim, keepdim=keepdim)
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        raise _sparse_dense_error(fname)
    if torch.is_tensor(m):
        return getattr(torch, fname)(m, dim=dim, keepdim=keepdim)
    r = getattr(np, fname)(m, axis=dim, keepdims=keepdim)
    # NumPy's keepdims is ignored for a full reduction; PyTorch's is too.
    return r

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_reduce), inheritreturns=_doc_returns_indices)
def argmin(a: QuantityLike, dim: Any=None, keepdim: bool=False, **kwargs: Any) -> Any:
    """Returns the index of `a`'s smallest element, as a plain integer array
    or tensor: the index along `dim`, or, when `dim` is not given, the index
    into the flattened input, as in both NumPy and PyTorch."""
    return _argminmax('argmin', a, dim, keepdim, kwargs)

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_reduce), inheritreturns=_doc_returns_indices)
def argmax(a: QuantityLike, dim: Any=None, keepdim: bool=False, **kwargs: Any) -> Any:
    """Returns the index of `a`'s largest element; see ``argmin``."""
    return _argminmax('argmax', a, dim, keepdim, kwargs)

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_reduce))
def median(a: QuantityLike, dim: Any=None, keepdim: bool=False, **kwargs: Any) -> Any:
    """Returns the median of `a`'s elements, preserving units.

    ``median(a)`` returns the median element itself and ``median(a, dim)``
    returns a ``(values, indices)`` named tuple, as ``torch.median`` does.

    For an even number of elements this is the lower of the two middle
    values--an element of `a`, which is why it has an index--rather than
    their mean, which is what ``numpy.median`` returns. The lower value is
    used for both backends, since one behavior must be chosen for both;
    ``mean(sort(a).values[..., k:k+2])`` gives the interpolated median.

    Parameters
    ----------

    Returns
    -------
    immlib.Quantity or median_result
        The median element, as a quantity, when `dim` is not given; a
        ``(values, indices)`` named tuple when it is.
    """
    (dim, keepdim) = _dimargs('median', kwargs, dim=dim, keepdim=keepdim)
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        raise _sparse_dense_error('median')
    if torch.is_tensor(m):
        if dim is None:
            return quant(torch.median(m), a.units)
        r = torch.median(m, dim=dim, keepdim=keepdim)
        return median_result(quant(r.values, a.units), r.indices)
    if dim is None:
        flat = np.sort(m, axis=None)
        return quant(flat[(flat.size - 1) // 2], a.units)
    dim = _one_dim('median', dim)
    order = np.argsort(m, axis=dim, kind='stable')
    k = (m.shape[dim] - 1) // 2
    idcs = np.take(order, k, axis=dim)
    vals = np.take_along_axis(m, np.expand_dims(idcs, dim), axis=dim)
    if keepdim:
        idcs = np.expand_dims(idcs, dim)
    else:
        vals = np.squeeze(vals, axis=dim)
    return median_result(quant(vals, a.units), idcs)

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_reduce), inheritreturns=_doc_returns_quantity)
def quantile(a: QuantityLike, q: Any, dim: Any=None, keepdim: bool=False, interpolation: str='linear',
             **kwargs: Any) -> Quantity:
    """Returns the `q`-th quantile of `a`'s elements, preserving units.

    `q` is a fraction between 0 and 1, or several of them, and
    `interpolation` is the rule for a quantile that falls between two
    elements (``'linear'``, ``'lower'``, ``'higher'``, ``'nearest'`` or
    ``'midpoint'``), as in ``torch.quantile``. See ``percentile`` for the
    same thing on a 0-to-100 scale.
    """
    (dim, keepdim) = _dimargs('quantile', kwargs, dim=dim, keepdim=keepdim)
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        raise _sparse_dense_error('quantile')
    if torch.is_tensor(m):
        qq = q if torch.is_tensor(q) else torch.as_tensor(
            q, dtype=m.dtype, device=m.device)
        if dim is None:
            rmag = torch.quantile(m, qq, keepdim=keepdim,
                                  interpolation=interpolation)
        else:
            rmag = torch.quantile(m, qq, dim=dim, keepdim=keepdim,
                                  interpolation=interpolation)
    else:
        rmag = np.quantile(m, q, axis=dim, keepdims=keepdim,  # type: ignore[call-overload]
                           method=interpolation)
    return quant(rmag, a.units)

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_reduce), inheritreturns=_doc_returns_quantity)
def percentile(a: QuantityLike, q: Any, dim: Any=None, keepdim: bool=False, interpolation: str='linear',
               **kwargs: Any) -> Quantity:
    """Returns the `q`-th percentile of `a`'s elements, preserving units;
    this is ``quantile(a, q / 100)``, the scale ``numpy.percentile`` uses.
    PyTorch has no ``percentile`` of its own."""
    q = np.asarray(q) / 100 if not torch.is_tensor(q) else q / 100
    return quantile(a, q, dim, keepdim, interpolation, **kwargs)

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_reduce), inheritreturns=_doc_returns_quantity)
def ptp(a: QuantityLike, dim: Any=None, keepdim: bool=False, **kwargs: Any) -> Quantity:
    """Returns the range of `a`'s elements--the largest minus the smallest,
    "peak to peak"--preserving units.

    ``numpy.ptp`` is the origin of the name; PyTorch has no equivalent, so
    this is computed as ``amax(a, dim) - amin(a, dim)`` for both backends,
    which keeps a tensor's gradient tracking.
    """
    (dim, keepdim) = _dimargs('ptp', kwargs, dim=dim, keepdim=keepdim)
    return amax(a, dim, keepdim) - amin(a, dim, keepdim)  # type: ignore[return-value]

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_reduce), inheritreturns=_doc_returns_quantity)
def average(a: QuantityLike, dim: Any=None, weights: Any=None, keepdim: bool=False, **kwargs: Any) -> Quantity:
    """Returns the weighted mean of `a`'s elements, preserving units.

    With no `weights` this is ``mean``. With them it is ``sum(a * weights,
    dim) / sum(weights, dim)``, computed that way for both backends, so a
    tensor keeps its gradient tracking. ``numpy.average`` is the origin of
    the name; PyTorch has no equivalent.

    The weights must be unit-less (their units would cancel in any case),
    and must broadcast against `a`.
    """
    (dim, keepdim) = _dimargs('average', kwargs, dim=dim, keepdim=keepdim)
    if weights is None:
        return mean(a, dim, keepdim)
    a = quant(a)
    w = quant(weights)
    _require_unitless(w, 'average')
    # The weights are summed over the same elements as the values are, so
    # weights of a broadcastable shape are broadcast to `a`'s shape first.
    (am, wm) = (a.m, w.m)
    if np.shape(wm) != np.shape(am):
        if torch.is_tensor(am) or torch.is_tensor(wm):
            (am, wm) = promote(am, wm)
            wm = torch.broadcast_to(wm, am.shape)
        else:
            wm = np.broadcast_to(wm, np.shape(am))
        w = quant(wm)
    total = sum(multiply(a, w), dim, keepdim)
    norm = sum(w, dim, keepdim)
    return divide(total, norm)


# Set operations ################################################################
# These are NumPy's; PyTorch has no equivalent for most of them, and none of
# them is differentiable in any implementation (their results are drawn from
# their inputs by comparison, not computed from them), so a tensor magnitude
# is detached and handed to NumPy, and the result is returned as a tensor on
# the same device. See the module docstring's Rule 2.

def _setop_mags(fname, a, b=None):
    """Returns ``(mags, units, like)`` for a set operation: the magnitudes as
    NumPy arrays, the units of the result, and the magnitude whose backend
    the result must be returned in."""
    if b is None:
        a = quant(a)
        (mags, u) = ([a.m], a.units)
    else:
        (ma, mb, u) = _align_units(a, b, fname)
        mags = [ma, mb]
    like = next((m for m in mags if torch.is_tensor(m)), None)
    out = []
    for m in mags:
        if sps.issparse(m):
            raise _sparse_dense_error(fname)
        out.append(m.detach().cpu().numpy() if torch.is_tensor(m)
                   else np.asarray(m))
    return (out, u, like)

def _setop_result(r, u, like):
    """Returns the result `r` of a set operation in `like`'s backend."""
    if like is not None:
        r = torch.as_tensor(r, device=like.device)
    return r if u is Ellipsis else quant(r, u)

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_reduce))
def unique(a: Any, sorted: bool=True, return_inverse: bool=False, return_counts: bool=False,
           dim: Any=None, **kwargs: Any) -> Any:
    """Returns `a`'s distinct elements in order, preserving units.

    ``return_inverse`` and ``return_counts`` add the index of each input
    element among the distinct ones, and the number of times each distinct
    element occurs, as plain arrays or tensors; the result is then a tuple.
    `sorted` is accepted for ``torch.unique``'s sake and is always true, as
    it is in ``numpy.unique``.

    This is not differentiable in either backend: a tensor is detached, and
    the result carries no gradient.

    Parameters
    ----------

    Returns
    -------
    immlib.Quantity or tuple
        The distinct values, as a quantity in `a`'s units; or, when
        `return_inverse` or `return_counts` is given, a tuple of that
        quantity followed by the requested plain integer arrays or tensors.
    """
    (dim, _) = _dimargs('unique', kwargs, dim=dim)
    ((m,), u, like) = _setop_mags('unique', a)
    r = np.unique(m, return_inverse=return_inverse,  # type: ignore[call-overload]
                  return_counts=return_counts, axis=dim)
    if not (return_inverse or return_counts):
        return _setop_result(r, u, like)
    return (_setop_result(r[0], u, like),
            *(_setop_result(x, Ellipsis, like) for x in r[1:]))

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def union1d(a: QuantityLike, b: QuantityLike) -> Quantity:
    """Returns the sorted, distinct elements of `a` and `b` together, in the
    units of the first argument that has them; see ``unique`` regarding
    gradients."""
    (mags, u, like) = _setop_mags('union1d', a, b)
    return _setop_result(np.union1d(*mags), u, like)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def intersect1d(a: Any, b: Any, assume_unique: bool=False) -> Quantity:
    """Returns the sorted, distinct elements common to `a` and `b`, in the
    units of the first argument that has them; see ``unique`` regarding
    gradients."""
    (mags, u, like) = _setop_mags('intersect1d', a, b)
    return _setop_result(
        np.intersect1d(*mags, assume_unique=assume_unique), u, like)  # type: ignore[call-overload]

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def setdiff1d(a: Any, b: Any, assume_unique: bool=False) -> Quantity:
    """Returns the sorted, distinct elements of `a` that are not in `b`, in
    the units of the first argument that has them; see ``unique`` regarding
    gradients."""
    (mags, u, like) = _setop_mags('setdiff1d', a, b)
    return _setop_result(
        np.setdiff1d(*mags, assume_unique=assume_unique), u, like)  # type: ignore[call-overload]

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def setxor1d(a: Any, b: Any, assume_unique: bool=False) -> Quantity:
    """Returns the sorted, distinct elements of exactly one of `a` and `b`,
    in the units of the first argument that has them; see ``unique``
    regarding gradients."""
    (mags, u, like) = _setop_mags('setxor1d', a, b)
    return _setop_result(
        np.setxor1d(*mags, assume_unique=assume_unique), u, like)  # type: ignore[call-overload]

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_bool)
def isin(elements: Any, test_elements: Any, assume_unique: bool=False, invert: bool=False) -> BoolArray:
    """Returns, for each element of `elements`, whether it occurs in
    `test_elements`, as a plain bool array or tensor of `elements`' shape.
    The two are unit-aligned as in ``maximum``; see ``unique`` regarding
    gradients."""
    (mags, _u, like) = _setop_mags('isin', elements, test_elements)
    r = np.isin(*mags, assume_unique=assume_unique, invert=invert)  # type: ignore[misc]
    return _setop_result(r, Ellipsis, like)


# Indexing and rearrangement ####################################################

def _index_mag(m, index, fname):
    """Returns `index` as an integer array or tensor in `m`'s backend."""
    if isinstance(index, pint.Quantity):
        if index.units is not None:
            raise TypeError(
                f"immlib.math.{fname}: an index must be unit-less; got units"
                f" {index.units}")
        index = index.m
    if torch.is_tensor(m):
        if not torch.is_tensor(index):
            index = torch.as_tensor(np.asarray(index), device=m.device)
        return index.long()
    if torch.is_tensor(index):
        index = index.detach().cpu().numpy()
    return np.asarray(index)

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_along), inheritreturns=_doc_returns_quantity)
def gather(a: QuantityLike, dim: Any, index: Any, **kwargs: Any) -> Quantity:
    """Returns the elements of `a` at `index` along `dim`, preserving units:
    the result has `index`'s shape, and its element at position ``(i, j)``
    is ``a[index[i, j], j]`` for ``dim=0``. This is ``torch.gather``;
    ``numpy.take_along_axis`` is the same operation."""
    (dim, _) = _dimargs('gather', kwargs, dim=dim)
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        raise _sparse_dense_error('gather')
    idx = _index_mag(m, index, 'gather')
    if torch.is_tensor(m):
        rmag = torch.gather(m, dim, idx)
    else:
        rmag = np.take_along_axis(m, idx, axis=dim)
    return quant(rmag, a.units)

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_along), inheritreturns=_doc_returns_quantity)
def index_select(a: QuantityLike, dim: Any, index: Any, **kwargs: Any) -> Quantity:
    """Returns the slices of `a` along `dim` at the entries of the 1-D
    `index`, preserving units. This is ``torch.index_select``;
    ``numpy.take`` with an axis is the same operation."""
    (dim, _) = _dimargs('index_select', kwargs, dim=dim)
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        raise _sparse_dense_error('index_select')
    idx = _index_mag(m, index, 'index_select')
    if torch.is_tensor(m):
        rmag = torch.index_select(m, dim, idx)
    else:
        rmag = np.take(m, idx, axis=dim)
    return quant(rmag, a.units)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def take(a: QuantityLike, index: Any) -> Quantity:
    """Returns the elements of `a` at the entries of `index`, which are
    indices into `a` flattened, preserving units. The result has `index`'s
    shape. This is ``torch.take`` and ``numpy.take`` without an axis."""
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        raise _sparse_dense_error('take')
    idx = _index_mag(m, index, 'take')
    rmag = torch.take(m, idx) if torch.is_tensor(m) else np.take(m, idx)
    return quant(rmag, a.units)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def masked_select(a: Any, mask: Any) -> Quantity:
    """Returns the elements of `a` where `mask` is true, as a 1-D quantity
    in `a`'s units. This is ``torch.masked_select``; indexing an array with
    a boolean array of the same shape is the same operation."""
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        raise _sparse_dense_error('masked_select')
    msk = mask.m if isinstance(mask, pint.Quantity) else mask
    if torch.is_tensor(m):
        if not torch.is_tensor(msk):
            msk = torch.as_tensor(np.asarray(msk), device=m.device)
        rmag = torch.masked_select(m, msk.bool())
    else:
        if torch.is_tensor(msk):
            msk = msk.detach().cpu().numpy()
        rmag = m[np.asarray(msk).astype(bool)]
    return quant(rmag, a.units)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def flip(a: QuantityLike, dims: Any=None, **kwargs: Any) -> Quantity:
    """Returns `a` with the order of its elements reversed along `dims` (or
    along every dimension, if `dims` is not given), preserving units."""
    (dims, _) = _dimargs('flip', kwargs, dim=dims)
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        raise _sparse_dense_error('flip')
    if dims is None:
        dims = tuple(range(np.ndim(m)))
    elif not isinstance(dims, (tuple, list)):
        dims = (dims,)
    if torch.is_tensor(m):
        rmag = torch.flip(m, tuple(dims))
    else:
        rmag = np.flip(m, axis=tuple(dims))
    return quant(rmag, a.units)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def roll(a: QuantityLike, shifts: Any, dims: Any=None, **kwargs: Any) -> Quantity:
    """Returns `a` with its elements shifted by `shifts` along `dims`,
    wrapping around, and preserving units. With no `dims`, `a` is flattened,
    shifted and restored to its shape, as in both libraries."""
    (dims, _) = _dimargs('roll', kwargs, dim=dims)
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        raise _sparse_dense_error('roll')
    if torch.is_tensor(m):
        sh = tuple(shifts) if isinstance(shifts, (tuple, list)) else shifts
        if dims is None:
            rmag = torch.roll(m, sh)
        else:
            dd = tuple(dims) if isinstance(dims, (tuple, list)) else dims
            rmag = torch.roll(m, sh, dd)
    else:
        rmag = np.roll(m, shifts, axis=dims)
    return quant(rmag, a.units)

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_along), inheritreturns=_doc_returns_quantity)
def repeat_interleave(a: QuantityLike, repeats: Any, dim: Any=None, **kwargs: Any) -> Quantity:
    """Returns `a` with each of its elements repeated `repeats` times along
    `dim`, preserving units; with no `dim`, `a` is flattened first. This is
    ``torch.repeat_interleave``; ``numpy.repeat`` is the same operation.
    (``numpy.ndarray.repeat``'s meaning, tiling the whole array, is
    ``tile``.)"""
    (dim, _) = _dimargs('repeat_interleave', kwargs, dim=dim)
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        raise _sparse_dense_error('repeat_interleave')
    if torch.is_tensor(m):
        reps = repeats
        if not isinstance(reps, int) and not torch.is_tensor(reps):
            reps = torch.as_tensor(np.asarray(reps), device=m.device)
        rmag = torch.repeat_interleave(m, reps, dim=dim)
    else:
        if torch.is_tensor(repeats):
            repeats = repeats.detach().cpu().numpy()
        rmag = np.repeat(m, repeats, axis=dim)
    return quant(rmag, a.units)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def tile(a: QuantityLike, dims: Any) -> Quantity:
    """Returns `a` tiled `dims` times along each dimension, preserving
    units. This is ``torch.tile`` and ``numpy.tile``."""
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        raise _sparse_dense_error('tile')
    dd = tuple(dims) if isinstance(dims, (tuple, list)) else (dims,)
    rmag = torch.tile(m, dd) if torch.is_tensor(m) else np.tile(m, dd)
    return quant(rmag, a.units)


def _splits(fname, n, sizes, dim, a):
    """Returns the pieces of `a` cut at the cumulative `sizes` along `dim`."""
    m = a.m
    out = []
    start = 0
    for size in sizes:
        stop = start + size
        idx = [slice(None)] * np.ndim(m)
        idx[dim] = slice(start, stop)
        out.append(quant(m[tuple(idx)], a.units))
        start = stop
    return tuple(out)

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_along))
def split(a: QuantityLike, split_size_or_sections: Any, dim: Any=0, **kwargs: Any) -> list:
    """Returns `a` cut into pieces along `dim`, as a tuple of quantities in
    `a`'s units.

    An integer gives the size of each piece, the last being smaller if the
    dimension does not divide evenly; a sequence of integers gives the size
    of each piece individually. This is ``torch.split``'s meaning of the
    argument; ``numpy.split``'s integer is a *number of pieces*, which is
    ``chunk`` here.

    Parameters
    ----------

    Returns
    -------
    tuple of immlib.Quantity
        The pieces, each in `a`'s units.
    """
    (dim, _) = _dimargs('split', kwargs, dim=dim)
    dim = 0 if dim is None else dim
    a = quant(a)
    if sps.issparse(a.m):
        raise _sparse_dense_error('split')
    n = np.shape(a.m)[dim]
    if isinstance(split_size_or_sections, (tuple, list)):
        sizes = [int(s) for s in split_size_or_sections]
        if builtins.sum(sizes) != n:
            raise ValueError(
                f"immlib.math.split: the sections {tuple(sizes)} do not sum"
                f" to the size of dimension {dim} ({n})")
    else:
        size = int(split_size_or_sections)
        if size <= 0:
            raise ValueError(
                "immlib.math.split: the split size must be positive")
        sizes = [size] * (n // size) + ([n % size] if n % size else [])
    return _splits('split', n, sizes, dim, a)

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_along))
def chunk(a: QuantityLike, chunks: Any, dim: Any=0, **kwargs: Any) -> list:
    """Returns `a` cut into at most `chunks` pieces along `dim`, as a tuple
    of quantities in `a`'s units.

    Every piece but the last has the same size, ``ceil(n / chunks)``, which
    is ``torch.chunk``'s rule and can yield fewer than `chunks` pieces--
    ``chunk(a, 4)`` of a dimension of 5 gives three pieces, of 2, 2 and 1.
    ``numpy.array_split`` balances the pieces instead, giving four of 2, 1,
    1 and 1; PyTorch's rule is used for both backends.

    Parameters
    ----------

    Returns
    -------
    tuple of immlib.Quantity
        The pieces, each in `a`'s units.
    """
    (dim, _) = _dimargs('chunk', kwargs, dim=dim)
    dim = 0 if dim is None else dim
    a = quant(a)
    if sps.issparse(a.m):
        raise _sparse_dense_error('chunk')
    chunks = int(chunks)
    if chunks <= 0:
        raise ValueError(
            "immlib.math.chunk: the number of chunks must be positive")
    n = np.shape(a.m)[dim]
    size = -(-n // chunks)          # ceiling division, as PyTorch does it
    if size == 0:
        return _splits('chunk', n, [1] * n, dim, a)
    sizes = [size] * (n // size) + ([n % size] if n % size else [])
    return _splits('chunk', n, sizes, dim, a)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_indices)
def nonzero(a: QuantityLike, as_tuple: bool=False) -> Any:
    """Returns the indices of `a`'s non-zero elements, as plain integer
    arrays or tensors.

    The result is a single ``(n, ndim)`` array or tensor, one row per
    non-zero element, which is ``torch.nonzero``'s form; ``as_tuple=True``
    gives one 1-dimensional index per dimension instead, which is
    ``numpy.nonzero``'s form.
    """
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        raise _sparse_dense_error('nonzero')
    if torch.is_tensor(m):
        return torch.nonzero(m, as_tuple=as_tuple)
    idcs = np.nonzero(m)
    return idcs if as_tuple else np.stack(idcs, axis=-1)

def _predicate(fname, np_fn, torch_name, a):
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        raise _sparse_dense_error(fname)
    if torch.is_tensor(m):
        return getattr(torch, torch_name)(m)
    return np_fn(m)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_bool)
def isnan(a: QuantityLike) -> BoolArray:
    """Returns, elementwise, whether `a` is not a number, as a plain bool
    array or tensor. The units are irrelevant and are not required."""
    return _predicate('isnan', np.isnan, 'isnan', a)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_bool)
def isinf(a: QuantityLike) -> BoolArray:
    """Returns, elementwise, whether `a` is positive or negative infinity,
    as a plain bool array or tensor."""
    return _predicate('isinf', np.isinf, 'isinf', a)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_bool)
def isfinite(a: QuantityLike) -> BoolArray:
    """Returns, elementwise, whether `a` is neither infinite nor a NaN, as a
    plain bool array or tensor."""
    return _predicate('isfinite', np.isfinite, 'isfinite', a)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def clamp(a: QuantityLike, min: QuantityLike | None=None, max: QuantityLike | None=None) -> Quantity:
    """Returns `a` with its elements limited to the range `min` to `max`,
    preserving units.

    At least one bound must be given, as ``torch.clamp`` requires. A bound
    is unit-aligned with `a` as in ``maximum``, so a bound with compatible
    units is converted and one with no units is dimensionless. ``clip`` is
    an alias, as it is in PyTorch.
    """
    a = quant(a)
    if min is None and max is None:
        raise ValueError(
            "immlib.math.clamp: at least one of 'min' or 'max' must be"
            " given")
    m = a.m
    if sps.issparse(m):
        raise _sparse_dense_error('clamp')
    u = a.units
    lo = None if min is None else _align_units(a, min, 'clamp')[1]
    hi = None if max is None else _align_units(a, max, 'clamp')[1]
    if torch.is_tensor(m):
        (lo, hi) = (
            None if lo is None else promote(m, lo)[1],
            None if hi is None else promote(m, hi)[1])
        rmag = torch.clamp(m, lo, hi)
    else:
        rmag = np.clip(m, lo, hi)
    return quant(rmag, u)

#: An alias of ``immlib.math.clamp``, as in PyTorch.
clip = clamp

#: The padding modes that ``immlib.math.pad`` accepts, and the NumPy mode
#: that computes each of them. PyTorch's names are the ones taken.
_PAD_MODES = {'constant': 'constant', 'reflect': 'reflect',
              'replicate': 'edge', 'circular': 'wrap'}

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def pad(a, pad, mode='constant', value=None):
    """Returns `a` with elements added around its edges, preserving units.

    `pad` is a flat sequence of amounts, ``(before, after)`` for the last
    dimension, then for the second-to-last, and so on--which is
    ``torch.nn.functional.pad``'s spelling, not ``numpy.pad``'s per-
    dimension list in the opposite order. `mode` is ``'constant'`` (the
    default), ``'reflect'``, ``'replicate'`` or ``'circular'``, again
    PyTorch's names; NumPy calls the last two ``'edge'`` and ``'wrap'``.

    `value` is the element to pad a constant padding with. It defaults to
    zero, and a bare number is taken to be in `a`'s own units, since a fill
    value replaces an element of `a` rather than combining with one; a
    quantity is converted into `a`'s units.

    Every mode but ``'constant'`` requires that the number of padded
    dimensions be one or two fewer than `a`'s number of dimensions, as
    PyTorch requires; the restriction is enforced for both backends, so
    that the same call behaves the same way.
    """
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        raise _sparse_dense_error('pad')
    if mode not in _PAD_MODES:
        raise ValueError(
            f"immlib.math.pad: unrecognized mode '{mode}'; the modes are"
            f" {tuple(_PAD_MODES)}")
    pads = tuple(int(p) for p in pad)
    if len(pads) % 2:
        raise ValueError(
            "immlib.math.pad: pad must give a pair of amounts per padded"
            " dimension, so its length must be even")
    npad = len(pads) // 2
    ndim = np.ndim(m)
    if npad > ndim:
        raise ValueError(
            f"immlib.math.pad: pad gives amounts for {npad} dimensions, but"
            f" the argument has {ndim}")
    if mode != 'constant' and ndim not in (npad + 1, npad + 2):
        raise ValueError(
            f"immlib.math.pad: mode '{mode}' pads {npad} dimension(s) of an"
            f" argument with {npad + 1} or {npad + 2} dimensions, but this"
            f" one has {ndim} (PyTorch's restriction, applied to both"
            f" backends)")
    # A bare fill value is in a's own units.
    if value is None:
        fill = 0
    elif isinstance(value, pint.Quantity) and value.units is not None:
        fill = _as_real_units(value, a.units) if a.units is not None else \
            _align_units(a, value, 'pad')[1]
    else:
        fill = value.m if isinstance(value, pint.Quantity) else value
    if torch.is_tensor(m):
        import torch.nn.functional as _F
        kw = {} if mode != 'constant' else {'value': builtins.float(fill)}
        rmag = _F.pad(m, pads, mode=mode, **kw)
    else:
        pairs = [(pads[2*i], pads[2*i + 1]) for i in range(npad)]
        width = [(0, 0)] * (ndim - npad) + list(reversed(pairs))
        if mode == 'constant':
            rmag = np.pad(m, width, mode='constant', constant_values=fill)
        else:
            rmag = np.pad(m, width, mode=_PAD_MODES[mode])
    return quant(rmag, a.units)


# Linear algebra ##################################################################

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def matmul(a: QuantityLike, b: QuantityLike) -> Quantity:
    """Returns ``a @ b``; see ``immlib.Quantity.__matmul__``.

    See ``dot`` for the 1-D inner product.
    """
    return quant(a) @ quant(b)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def dot(a: QuantityLike, b: QuantityLike) -> Quantity:
    """Returns the inner product of the 1-dimensional `a` and `b`; the
    result's units are the product of theirs.

    This is ``torch.dot``'s meaning, for both backends: both arguments must
    be 1-dimensional, and anything else is an error. ``numpy.dot`` instead
    behaves like matrix multiplication with broadcasting for 2-dimensional
    and higher input, which is ``matmul`` (or the ``@`` operator) here. The
    two libraries would otherwise disagree about the same call, so immlib
    takes the narrower meaning and leaves the wider one to ``matmul``.
    """
    a = quant(a)
    b = quant(b)
    (ma, mb) = (a.m, b.m)
    if np.ndim(ma) != 1 or np.ndim(mb) != 1:
        raise ValueError(
            f"immlib.math.dot: both arguments must be 1-dimensional (got"
            f" {np.ndim(ma)} and {np.ndim(mb)} dimensions); use"
            f" immlib.math.matmul, or the @ operator, for matrix"
            f" multiplication")
    # The lengths are checked here rather than by the backend, which would
    # raise a ValueError for an array and a RuntimeError for a tensor.
    if np.shape(ma) != np.shape(mb):
        raise ValueError(
            f"immlib.math.dot: both arguments must have the same length"
            f" (got {np.shape(ma)[0]} and {np.shape(mb)[0]})")
    return a @ b


# Linear algebra ###############################################################

def _linalg_mag(fname, a):
    """Returns the unit-less magnitude of `a` for a linear-algebra function.

    Like ``exp`` and ``log``, these functions require an argument with no units
    (``units=None``), because a decomposition or a fit produces several values
    whose units differ and are not tracked here. Sparse arguments and arguments
    with fewer than two dimensions are rejected.
    """
    a = quant(a)
    _require_unitless(a, fname)
    m = a.m
    if sps.issparse(m):
        raise _sparse_dense_error(fname)
    if np.ndim(m) < 2:
        raise ValueError(
            f"immlib.math.{fname}: the argument must have at least 2"
            f" dimensions (got {np.ndim(m)})")
    return m

def _linalg_tol(m, atol, rtol):
    """Returns ``(atol, rtol)`` with PyTorch's defaults filled in.

    PyTorch's ``pinv`` and ``matrix_rank`` default to ``atol=0`` and
    ``rtol = eps * max(m, n)`` (``eps`` being the dtype's machine epsilon).
    Filling them in here and applying the same rule to both backends is what
    keeps an array and a tensor in agreement.
    """
    (rows, cols) = (m.shape[-2], m.shape[-1])
    if torch.is_tensor(m):
        eps = float(torch.finfo(m.dtype).eps)
    else:
        eps = float(np.finfo(m.dtype).eps)
    if atol is None:
        atol = 0.0
    if rtol is None:
        rtol = eps * builtins.max(rows, cols)
    return (atol, rtol)


@docwrap(format='numpy', inheritparams=_doc_params)
def movedim(a, source, destination):
    """Returns `a` with the dimensions `source` moved to `destination`, keeping
    its units.

    This is ``torch.movedim``; ``numpy.moveaxis`` is the same operation under
    another name, so ``moveaxis`` is an alias of this function.

    Parameters
    ----------
    source : int or sequence of int
        The dimension or dimensions to move. Negative dimensions count from
        the end.
    destination : int or sequence of int
        The position or positions to move them to; it must have the same
        number of elements as `source`.

    Returns
    -------
    immlib.Quantity
        `a` with the given dimensions moved to the given positions, with the
        same units as `a`.
    """
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        raise _sparse_dense_error('movedim')
    if torch.is_tensor(m):
        rmag = torch.movedim(m, source, destination)
    else:
        rmag = np.moveaxis(m, source, destination)
    return quant(rmag, a.units)
moveaxis = movedim


@docwrap(format='numpy', inheritparams=_doc_params)
def svd(a, full_matrices=True, *, driver=None):
    """Returns the singular value decomposition of `a`.

    The result is the triple ``(U, S, Vh)`` that ``torch.linalg.svd`` returns
    (and that ``numpy.linalg.svd`` returns as well), so that ``a`` is ``U @
    diag(S) @ Vh`` up to the trailing dimensions when `full_matrices` is
    ``False``. The argument must be unit-less, and the result is returned as
    plain arrays or tensors rather than quantities, since its parts have
    different units.

    Parameters
    ----------
    full_matrices : bool, optional
        Whether to return the full ``(m, m)`` and ``(n, n)`` unitary matrices
        (``True``, the default) or only the leading ``min(m, n)`` columns of
        each (``False``).
    driver : str or None, optional
        The LAPACK driver to use, as ``torch.linalg.svd`` accepts it. NumPy's
        SVD has no driver, so giving one with an array argument is an error.

    Returns
    -------
    U : array or tensor
        The left singular vectors.
    S : array or tensor
        The singular values, in descending order.
    Vh : array or tensor
        The right singular vectors, already transposed (and conjugated).
    """
    m = _linalg_mag('svd', a)
    if torch.is_tensor(m):
        return tuple(torch.linalg.svd(m, full_matrices=full_matrices,
                                      driver=driver))
    if driver is not None:
        raise TypeError(
            "immlib.math.svd: the 'driver' option is only supported for"
            " tensors; numpy.linalg.svd has no driver")
    return tuple(np.linalg.svd(m, full_matrices=full_matrices))


@docwrap(format='numpy', inheritparams=_doc_params)
def pinv(a, *, atol=None, rtol=None, hermitian=False):
    """Returns the pseudo-inverse of `a`.

    This is ``torch.linalg.pinv``, with its default tolerance rule
    (``atol=0`` and ``rtol = eps * max(m, n)``) reproduced for both backends,
    so that an array and a tensor give the same result. A singular value (or,
    for a Hermitian argument, an eigenvalue) at or below the cutoff ``atol +
    rtol * S.max()`` is treated as zero. The argument must be unit-less, and
    the result is a plain array or tensor.

    Parameters
    ----------
    atol : float or None, optional
        The absolute tolerance. The default, ``None``, is ``0``.
    rtol : float or None, optional
        The relative tolerance. The default, ``None``, is ``eps * max(m, n)``,
        where ``eps`` is the dtype's machine epsilon.
    hermitian : bool, optional
        Whether `a` is assumed Hermitian (or symmetric), which allows a
        faster eigendecomposition-based path. The default is ``False``.

    Returns
    -------
    array or tensor
        The pseudo-inverse of `a`.
    """
    m = _linalg_mag('pinv', a)
    (atol, rtol) = _linalg_tol(m, atol, rtol)
    if torch.is_tensor(m):
        return torch.linalg.pinv(m, atol=atol, rtol=rtol, hermitian=hermitian)
    if hermitian:
        (w, V) = np.linalg.eigh(m)
        aw = np.abs(w)
        keep = aw > (atol + rtol * aw[..., -1:])
        winv = np.where(keep, 1.0 / np.where(keep, w, 1.0), 0.0)
        Vh = np.swapaxes(V, -1, -2).conj()
        return V @ (winv[..., :, None] * Vh)
    (U, S, Vh) = np.linalg.svd(m, full_matrices=False)
    keep = S > (atol + rtol * S[..., :1])
    Sinv = np.where(keep, 1.0 / np.where(keep, S, 1.0), 0.0)
    Uh = np.swapaxes(U, -1, -2).conj()
    return np.swapaxes(Vh, -1, -2).conj() @ (Sinv[..., :, None] * Uh)


@docwrap(format='numpy', inheritparams=_doc_params)
def matrix_rank(a, *, atol=None, rtol=None, hermitian=False):
    """Returns the rank of `a`.

    This is ``torch.linalg.matrix_rank``, with its default tolerance rule
    (``atol=0`` and ``rtol = eps * max(m, n)``) reproduced for both backends,
    so that an array and a tensor give the same result: the rank is the number
    of singular values (or, for a Hermitian argument, eigenvalues) larger than
    ``atol + rtol * S.max()``. The argument must be unit-less, and the result
    is a plain integer array or tensor.

    Parameters
    ----------
    atol : float or None, optional
        The absolute tolerance. The default, ``None``, is ``0``.
    rtol : float or None, optional
        The relative tolerance. The default, ``None``, is ``eps * max(m, n)``,
        where ``eps`` is the dtype's machine epsilon.
    hermitian : bool, optional
        Whether `a` is assumed Hermitian (or symmetric), which allows a
        faster eigendecomposition-based path. The default is ``False``.

    Returns
    -------
    array or tensor of int
        The rank of `a`.
    """
    m = _linalg_mag('matrix_rank', a)
    (atol, rtol) = _linalg_tol(m, atol, rtol)
    if torch.is_tensor(m):
        return torch.linalg.matrix_rank(m, atol=atol, rtol=rtol,
                                        hermitian=hermitian)
    if hermitian:
        s = np.abs(np.linalg.eigvalsh(m))
        smax = s[..., -1:]
    else:
        s = np.linalg.svd(m, compute_uv=False)
        smax = s[..., :1]
    return (s > (atol + rtol * smax)).sum(axis=-1)


@docwrap(format='numpy')
def einsum(equation, *operands):
    """Evaluates an Einstein-summation expression over unit-less operands.

    This is ``torch.einsum`` (and ``numpy.einsum`` for array operands),
    following ``immlib.math``'s backend rule: if any operand is a tensor then
    all of them are treated as tensors, and otherwise they are treated as
    arrays. Every operand must be unit-less -- the units of an arbitrary
    contraction are not computed here -- and the result is a plain array or
    tensor.

    Parameters
    ----------
    equation : str
        The summation expression, in either library's notation.
    operands
        The operands of the expression.

    Returns
    -------
    array or tensor
        The result of the contraction.
    """
    mags = []
    for op in operands:
        q = quant(op)
        _require_unitless(q, 'einsum')
        mags.append(q.m)
    if builtins.any(sps.issparse(x) for x in mags):
        raise _sparse_dense_error('einsum')
    if builtins.any(torch.is_tensor(x) for x in mags):
        # As in matmul, a non-tensor operand is promoted onto the first
        # tensor's device *and* dtype: torch.einsum, unlike NumPy, requires
        # its operands to share an exact dtype.
        first = next(x for x in mags if torch.is_tensor(x))
        ops = [
            x if torch.is_tensor(x)
            else to_tensor(x, dtype=first.dtype, device=first.device)
            for x in mags]
        return torch.einsum(equation, *ops)
    return np.einsum(equation, *mags)


def _np_lstsq(m, mb, rcond):
    """Returns ``(solution, rank, singular_values)`` for a possibly batched
    two-dimensional least-squares system.

    ``numpy.linalg.lstsq`` handles only 2-D systems, so batched inputs are
    solved one slice at a time and the results stacked.
    """
    if m.ndim == 2:
        (sol, _res, rank, sv) = np.linalg.lstsq(m, mb, rcond=rcond)
        return (sol, np.array(rank), sv)
    batch = np.broadcast_shapes(m.shape[:-2], mb.shape[:-2])
    ms = np.broadcast_to(m, batch + m.shape[-2:])
    bs = np.broadcast_to(mb, batch + mb.shape[-2:])
    sols, ranks, svs = [], [], []
    for idx in np.ndindex(batch):
        (sol, _res, rank, sv) = np.linalg.lstsq(ms[idx], bs[idx], rcond=rcond)
        sols.append(sol)
        ranks.append(rank)
        svs.append(sv)
    return (np.stack(sols), np.array(ranks), np.stack(svs))


@docwrap(format='numpy', inheritparams=_doc_params)
def lstsq(a, b, rcond=None, *, driver=None):
    """Returns the least-squares solution to ``a @ x = b``.

    This is ``torch.linalg.lstsq``, and the result is its 4-tuple
    ``(solution, residuals, rank, singular_values)``. Which of the parts are
    computed follows PyTorch's conventions for both backends: `residuals` is
    empty unless a driver that computes it is given (``'gels'``, ``'gelsd'`` or
    ``'gelss'``) *and* the system is overdetermined; `rank` is empty for
    ``'gels'`` (which does not compute it); and `singular_values` is present
    only for ``'gelsd'`` and ``'gelss'``. An empty part is a 1-dimensional empty
    array or tensor, as it is in PyTorch. Both arguments must be unit-less, and
    the results are plain arrays or tensors.

    Parameters
    ----------
    rcond : float or None, optional
        The cutoff for small singular values, as both libraries accept it. The
        default, ``None``, uses the dtype's machine epsilon times ``max(m, n)``.
    driver : str or None, optional
        The LAPACK driver to use: ``'gels'``, ``'gelsy'``, ``'gelsd'`` or
        ``'gelss'``. The default, ``None``, leaves the choice to the backend
        (``'gelsy'`` on CPU PyTorch). NumPy has no driver, so for an array this
        option selects which parts of the result are returned rather than how
        the solve is done.

    Returns
    -------
    solution : array or tensor
        The least-squares solution ``x``, of shape ``(..., n, k)``.
    residuals : array or tensor
        The sum of squared residuals for each right-hand side, of shape
        ``(..., k)``, or an empty array when they are not computed.
    rank : array or tensor of int
        The rank of `a`, or an empty array when it is not computed.
    singular_values : array or tensor
        The singular values of `a`, or an empty array when they are not
        computed.
    """
    ma = _linalg_mag('lstsq', a)
    mb = _linalg_mag('lstsq', b)
    drivers = (None, 'gels', 'gelsy', 'gelsd', 'gelss')
    if driver not in drivers:
        raise ValueError(
            f"immlib.math.lstsq: unrecognized driver {driver!r}; expected one"
            f" of {tuple(d for d in drivers if d is not None)}")
    if torch.is_tensor(ma) or torch.is_tensor(mb):
        if torch.is_tensor(ma) and torch.is_tensor(mb):
            pass
        else:
            first = ma if torch.is_tensor(ma) else mb
            ma = (ma if torch.is_tensor(ma)
                  else to_tensor(ma, dtype=first.dtype, device=first.device))
            mb = (mb if torch.is_tensor(mb)
                  else to_tensor(mb, dtype=first.dtype, device=first.device))
        r = torch.linalg.lstsq(ma, mb, rcond=rcond, driver=driver)
        return (r.solution, r.residuals, r.rank, r.singular_values)
    # NumPy: solve (2-D only, batched by _np_lstsq) and then reproduce which
    # parts torch's driver computes exactly, with the same empty shapes.
    (rows, cols) = (ma.shape[-2], ma.shape[-1])
    (sol, rank, sv) = _np_lstsq(ma, mb, rcond)
    if driver in ('gels', 'gelsd', 'gelss') and rows > cols:
        residuals = np.sum((ma @ sol - mb) ** 2, axis=-2)
    else:
        residuals = np.zeros(0)
    if driver == 'gels':
        rank = np.zeros(0)
    if driver not in ('gelsd', 'gelss'):
        sv = np.zeros(0)
    return (sol, residuals, rank, sv)


# Array creation ###############################################################

def _like_mag(fname, a):
    """Returns ``(a, magnitude)`` for a ``*_like`` function.

    These functions take their backend, shape, and device from an argument, so
    no backend has to be named; a sparse argument is rejected, since the
    result of allocating ``*_like`` is dense.
    """
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        raise _sparse_dense_error(fname)
    return (a, m)

def _fill_mag(fname, fill_value, units):
    """Returns the magnitude to use for `fill_value` in `units`.

    A bare number is taken in `units`; a quantity is converted into `units`
    (and must be unit-less if `units` is ``None``).
    """
    if is_quant(fill_value):
        if fill_value.units is None:
            return fill_value.m
        if units is None:
            raise pint.DimensionalityError(
                fill_value.units, 'dimensionless',
                extra_msg=(f" immlib.math.{fname}: the argument has no units,"
                           f" so a fill value that has units cannot be used"))
        return fill_value.m_as(units)
    return fill_value


@docwrap(format='numpy', inheritparams=_doc_params)
def zeros_like(a, dtype=None):
    """Returns zeros with `a`'s shape and units.

    This is ``numpy.zeros_like`` / ``torch.zeros_like``; because it takes `a`
    as an example, the backend (and the device, for a tensor) follows `a`, so
    no backend has to be named. The result is a quantity with `a`'s units:
    zeros in those units, not a bare array.

    Parameters
    ----------
    dtype : dtype-like or None, optional
        The dtype of the result. The default, ``None``, follows `a`.

    Returns
    -------
    immlib.Quantity
        Zeros with `a`'s shape, backend, and units.
    """
    (a, m) = _like_mag('zeros_like', a)
    if torch.is_tensor(m):
        rmag = torch.zeros_like(m, dtype=dtype)
    else:
        rmag = np.zeros_like(m, dtype=dtype)
    return quant(rmag, a.units)


@docwrap(format='numpy', inheritparams=_doc_params)
def ones_like(a, dtype=None):
    """Returns ones with `a`'s shape and units.

    This is ``numpy.ones_like`` / ``torch.ones_like``; as with ``zeros_like``,
    the backend follows `a`, and the result is a quantity with `a`'s units:
    ones in those units.

    Parameters
    ----------
    dtype : dtype-like or None, optional
        The dtype of the result. The default, ``None``, follows `a`.

    Returns
    -------
    immlib.Quantity
        Ones with `a`'s shape, backend, and units.
    """
    (a, m) = _like_mag('ones_like', a)
    if torch.is_tensor(m):
        rmag = torch.ones_like(m, dtype=dtype)
    else:
        rmag = np.ones_like(m, dtype=dtype)
    return quant(rmag, a.units)


@docwrap(format='numpy', inheritparams=_doc_params)
def full_like(a, fill_value, dtype=None):
    """Returns `fill_value` with `a`'s shape and units.

    This is ``numpy.full_like`` / ``torch.full_like``; the backend follows `a`.
    A bare number is taken in `a`'s units; a quantity is converted into `a`'s
    units, and a quantity with units cannot be used when `a` has none.

    Parameters
    ----------
    fill_value : number or quantity
        The value to fill the result with, in `a`'s units (or a quantity that
        is convertible into them).
    dtype : dtype-like or None, optional
        The dtype of the result. The default, ``None``, follows `a`.

    Returns
    -------
    immlib.Quantity
        `fill_value` with `a`'s shape, backend, and units.
    """
    (a, m) = _like_mag('full_like', a)
    fv = _fill_mag('full_like', fill_value, a.units)
    if torch.is_tensor(m):
        rmag = torch.full_like(m, fv, dtype=dtype)
    else:
        rmag = np.full_like(m, fv, dtype=dtype)
    return quant(rmag, a.units)


def _promote_tensors(mags):
    """Promotes the magnitudes to tensors on the first tensor's device and
    dtype (as matmul and einsum do), for a call whose backend is PyTorch."""
    first = next(x for x in mags if torch.is_tensor(x))
    return tuple(
        x if torch.is_tensor(x)
        else to_tensor(x, dtype=first.dtype, device=first.device)
        for x in mags)


@docwrap(format='numpy', inheritparams=_doc_params)
def empty_like(a, dtype=None):
    """Returns an uninitialized array or tensor with `a`'s shape and units.

    ``numpy.empty_like`` / ``torch.empty_like``: the backend follows `a`, and
    the result is a quantity with `a`'s units. The contents are *not* defined --
    they are whatever the backend's allocator happens to produce, and so differ
    between the backends and between calls -- so use ``zeros_like``,
    ``ones_like`` or ``full_like`` unless every element will be overwritten.

    Parameters
    ----------
    dtype : dtype-like or None, optional
        The dtype of the result. The default, ``None``, follows `a`.

    Returns
    -------
    immlib.Quantity
        An uninitialized array or tensor with `a`'s shape and units.
    """
    (a, m) = _like_mag('empty_like', a)
    if torch.is_tensor(m):
        rmag = torch.empty_like(m, dtype=dtype)
    else:
        rmag = np.empty_like(m, dtype=dtype)
    return quant(rmag, a.units)


@docwrap(format='numpy', inheritparams=_doc_params)
def rand_like(a, dtype=None):
    """Returns uniform random values in ``[0, 1)`` with `a`'s shape and units.

    ``torch.rand_like`` has no NumPy counterpart (NumPy has no ``*_like``
    random function), so an array argument is filled with ``numpy.random``; the
    two backends are seeded separately (``torch.manual_seed`` and
    ``numpy.random.seed``). The result is a quantity with `a`'s units.

    Parameters
    ----------
    dtype : dtype-like or None, optional
        The dtype of the result. The default, ``None``, follows `a` for a
        tensor and is ``float64`` for an array.

    Returns
    -------
    immlib.Quantity
        Uniform random values with `a`'s shape and units.
    """
    (a, m) = _like_mag('rand_like', a)
    if torch.is_tensor(m):
        rmag = torch.rand_like(m, dtype=dtype)
    else:
        if dtype is None:
            dtype = (m.dtype if np.issubdtype(m.dtype, np.floating)
                     else np.float64)
        rmag = np.random.random(m.shape).astype(dtype)
    return quant(rmag, a.units)


@docwrap(format='numpy', inheritparams=_doc_params)
def randn_like(a, dtype=None):
    """Returns standard-normal random values with `a`'s shape and units.

    ``torch.randn_like``; as with ``rand_like``, an array argument is filled
    with ``numpy.random``, and the two backends are seeded separately. The
    result is a quantity with `a`'s units.

    Parameters
    ----------
    dtype : dtype-like or None, optional
        The dtype of the result. The default, ``None``, follows `a` for a
        tensor and is ``float64`` for an array.

    Returns
    -------
    immlib.Quantity
        Standard-normal random values with `a`'s shape and units.
    """
    (a, m) = _like_mag('randn_like', a)
    if torch.is_tensor(m):
        rmag = torch.randn_like(m, dtype=dtype)
    else:
        if dtype is None:
            dtype = (m.dtype if np.issubdtype(m.dtype, np.floating)
                     else np.float64)
        rmag = np.random.randn(*m.shape).astype(dtype)
    return quant(rmag, a.units)


@docwrap(format='numpy', inheritparams=_doc_params)
def randint_like(a, low, high, dtype=None):
    """Returns random integers in ``[low, high)`` with `a`'s shape and units.

    ``torch.randint_like``; as with ``rand_like``, an array argument is filled
    with ``numpy.random``. The result is a quantity with `a`'s units.

    Parameters
    ----------
    low : int
        The lowest value to draw (inclusive).
    high : int
        The highest value to draw (exclusive).
    dtype : dtype-like or None, optional
        The dtype of the result. The default, ``None``, follows `a` when its
        dtype is an integer and is ``int64`` otherwise (as
        ``torch.randint_like`` does).

    Returns
    -------
    immlib.Quantity
        Random integers with `a`'s shape and units.
    """
    (a, m) = _like_mag('randint_like', a)
    if torch.is_tensor(m):
        rmag = torch.randint_like(m, low, high, dtype=dtype)
    else:
        if dtype is None:
            dtype = (m.dtype if np.issubdtype(m.dtype, np.integer)
                     else np.int64)
        rmag = np.random.randint(low, high, size=m.shape, dtype=dtype)
    return quant(rmag, a.units)


# Shape ########################################################################

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def atleast_1d(a):
    """Returns `a` with at least one dimension, keeping its units.

    ``numpy.atleast_1d`` / ``torch.atleast_1d``: a scalar becomes a 1-element
    array. Only a single argument is accepted (the libraries' multiple-argument
    forms return sequences, which do not fit this namespace).
    """
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        raise _sparse_dense_error('atleast_1d')
    rmag = torch.atleast_1d(m) if torch.is_tensor(m) else np.atleast_1d(m)
    return quant(rmag, a.units)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def atleast_2d(a):
    """Returns `a` with at least two dimensions, keeping its units.

    ``numpy.atleast_2d`` / ``torch.atleast_2d``; see ``atleast_1d``."""
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        raise _sparse_dense_error('atleast_2d')
    rmag = torch.atleast_2d(m) if torch.is_tensor(m) else np.atleast_2d(m)
    return quant(rmag, a.units)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def atleast_3d(a):
    """Returns `a` with at least three dimensions, keeping its units.

    ``numpy.atleast_3d`` / ``torch.atleast_3d``; see ``atleast_1d``."""
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        raise _sparse_dense_error('atleast_3d')
    rmag = torch.atleast_3d(m) if torch.is_tensor(m) else np.atleast_3d(m)
    return quant(rmag, a.units)


@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def broadcast_to(a, shape):
    """Returns `a` broadcast to `shape`, keeping its units.

    ``numpy.broadcast_to`` / ``torch.broadcast_to``, which return a read-only
    view rather than a copy.

    Parameters
    ----------
    shape : tuple of int
        The shape of the result. Leading dimensions may be added, and each
        existing dimension must be 1 or already equal to the requested size.
    """
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        raise _sparse_dense_error('broadcast_to')
    rmag = (torch.broadcast_to(m, tuple(shape)) if torch.is_tensor(m)
            else np.broadcast_to(m, tuple(shape)))
    return quant(rmag, a.units)


def _np_expand(m, sizes):
    """Emulates ``torch.Tensor.expand`` with NumPy: prepend the requested
    leading dimensions, resolve ``-1`` to the existing size, and broadcast."""
    sizes = tuple(sizes)
    if len(sizes) < m.ndim:
        raise ValueError(
            f"immlib.math.expand: the number of sizes ({len(sizes)}) cannot be"
            f" fewer than the number of dimensions ({m.ndim})")
    pad = (1,) * (len(sizes) - m.ndim)
    shape = pad + tuple(m.shape)
    resolved = tuple(s if s != -1 else shape[ii]
                     for (ii, s) in enumerate(sizes))
    return np.broadcast_to(m.reshape(shape), resolved)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def expand(a, *sizes):
    """Returns `a` with the dimensions expanded to `sizes`, keeping its units.

    This is ``torch.Tensor.expand`` (NumPy has no function of this name; its
    ``broadcast_to`` is the closest). The sizes may be given as a single tuple
    or as separate arguments; a ``-1`` keeps an existing dimension's size, and
    leading dimensions may be added.

    Parameters
    ----------
    sizes : ints
        The sizes of the result's dimensions, as a tuple or as separate
        arguments. A ``-1`` keeps the dimension's size.
    """
    if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
        sizes = tuple(sizes[0])
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        raise _sparse_dense_error('expand')
    rmag = m.expand(*sizes) if torch.is_tensor(m) else _np_expand(m, sizes)
    return quant(rmag, a.units)


# More sequence functions ######################################################

@docwrap(format='numpy', inheritparams=(_doc_params, _doc_dim_along),
          inheritreturns=_doc_returns_quantity)
def cumprod(a, dim, **kwargs):
    """Returns the cumulative product of `a`'s elements along `dim`, preserving
    units. `dim` is required, as it is in ``torch.cumprod``."""
    (dim, _) = _dimargs('cumprod', kwargs, dim=dim)
    dim = _one_dim('cumprod', dim)
    if dim is None:
        raise TypeError("immlib.math.cumprod: 'dim' is required")
    a = quant(a)
    m = a.m
    if torch.is_tensor(m):
        rmag = torch.cumprod(m, dim=dim)
    elif sps.issparse(m):
        raise _sparse_dense_error('cumprod')
    else:
        rmag = np.cumprod(m, axis=dim)
    return quant(rmag, a.units)

@docwrap(format='numpy', inheritparams=(_doc_params,),
          inheritreturns=_doc_returns_quantity)
def diff(a, n=1, dim=-1, **kwargs):
    """Returns the `n`-th discrete difference along `dim`, keeping `a`'s units.

    ``numpy.diff`` / ``torch.diff``.

    Parameters
    ----------
    n : int, optional
        The number of times the difference is taken. The default is ``1``.
    dim : int, optional
        The dimension along which the difference is taken. The default is
        ``-1``.
    """
    (dim, _) = _dimargs('diff', kwargs, dim=dim)
    a = quant(a)
    m = a.m
    if torch.is_tensor(m):
        rmag = torch.diff(m, n=n, dim=dim)
    elif sps.issparse(m):
        raise _sparse_dense_error('diff')
    else:
        rmag = np.diff(m, n=n, axis=dim)
    return quant(rmag, a.units)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def flipud(a):
    """Returns `a` with the order of the elements along axis 0 reversed,
    keeping its units (``numpy.flipud`` / ``torch.flipud``)."""
    a = quant(a)
    m = a.m
    rmag = torch.flipud(m) if torch.is_tensor(m) else np.flipud(m)
    return quant(rmag, a.units)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def fliplr(a):
    """Returns `a` with the order of the elements along axis 1 reversed,
    keeping its units (``numpy.fliplr`` / ``torch.fliplr``); `a` must be at
    least 2-dimensional."""
    a = quant(a)
    m = a.m
    rmag = torch.fliplr(m) if torch.is_tensor(m) else np.fliplr(m)
    return quant(rmag, a.units)


# Counting and tolerance predicates ###########################################

@docwrap(format='numpy', inheritparams=_doc_params)
def count_nonzero(a, dim=None, **kwargs):
    """Returns the number of non-zero elements, as a plain integer or array.

    ``numpy.count_nonzero`` / ``torch.count_nonzero``. The units are irrelevant
    to a count, so a quantity of any units is accepted and the result is a
    plain integer array or tensor.

    Parameters
    ----------
    dim : int, optional
        The dimension along which to count. The default, ``None``, counts every
        element. ``axis`` is accepted as an alias.

    Returns
    -------
    int or array or tensor of int
        The number of non-zero elements, or, when `dim` is given, the
        counts along `dim`.
    """
    (dim, _) = _dimargs('count_nonzero', kwargs, dim=dim)
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        raise _sparse_dense_error('count_nonzero')
    if torch.is_tensor(m):
        return torch.count_nonzero(m, dim=dim)
    return np.count_nonzero(m, axis=dim)

@docwrap(format='numpy')
def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """Returns whether `a` and `b` are equal to within a tolerance.

    ``numpy.allclose`` / ``torch.allclose``, with units handled as ``isclose``
    does. The result is a plain ``bool``.

    Parameters
    ----------
    a : quantity or array or tensor or number
        The first value, in whose units the comparison is performed.
    b : quantity or array or tensor or number
        The value to compare `a` with; a unit-less (or bare) operand is treated
        as dimensionless, so comparing it with a dimensional one raises
        ``pint.DimensionalityError``, as it does in Pint.
    rtol : float, optional
        The relative tolerance. The default is ``1e-05``.
    atol : float, optional
        The absolute tolerance. The default is ``1e-08``.
    equal_nan : bool, optional
        Whether two NaNs are considered equal. The default is ``False``.

    Returns
    -------
    bool
        ``True`` if the two values agree within the tolerance.
    """
    (ma, mb, _u) = _align_units(a, b, 'allclose')
    if _is_tensor_backed(ma, mb):
        (ma, mb) = _promote_tensors((ma, mb))
        return torch.allclose(ma, mb, rtol=rtol, atol=atol,
                              equal_nan=equal_nan)
    return np.allclose(ma, mb, rtol=rtol, atol=atol, equal_nan=equal_nan)

@docwrap(format='numpy')
def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """Returns, elementwise, whether `a` and `b` are equal within a tolerance.

    ``numpy.isclose`` / ``torch.isclose``. `b` is converted into `a`'s units; a
    unit-less (or bare) operand is treated as dimensionless, so comparing it
    with a dimensional one raises ``pint.DimensionalityError``, exactly as Pint
    does. The result is a plain boolean array or tensor.

    Parameters
    ----------
    a : quantity or array or tensor or number
        The first value, in whose units the comparison is performed.
    b : quantity or array or tensor or number
        The value to compare `a` with.
    rtol : float, optional
        The relative tolerance. The default is ``1e-05``.
    atol : float, optional
        The absolute tolerance. The default is ``1e-08``.
    equal_nan : bool, optional
        Whether two NaNs are considered equal. The default is ``False``.

    Returns
    -------
    array or tensor of bool
        ``True`` where the two values agree within the tolerance.
    """
    (ma, mb, _u) = _align_units(a, b, 'isclose')
    if _is_tensor_backed(ma, mb):
        (ma, mb) = _promote_tensors((ma, mb))
        return torch.isclose(ma, mb, rtol=rtol, atol=atol,
                             equal_nan=equal_nan)
    return np.isclose(ma, mb, rtol=rtol, atol=atol, equal_nan=equal_nan)


# More linear algebra ##########################################################

def _unit_product(a, b):
    """The units of a product of two quantities (a ``None`` operand
    contributes no units)."""
    if a.units is None:
        return b.units
    if b.units is None:
        return a.units
    return a.units * b.units

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def norm(a, ord=None, dim=None, keepdim=False, **kwargs):
    """Returns the norm of `a`, keeping its units.

    ``numpy.linalg.norm`` / ``torch.linalg.norm``. The units of a norm are the
    units of its argument, so a quantity of any units is accepted.

    Parameters
    ----------
    ord : number or str or None, optional
        The order of the norm, as the two libraries accept it. The default,
        ``None``, is the 2-norm of a vector and the Frobenius norm of a matrix.
    dim : int or tuple of int, optional
        The dimension or dimensions along which to compute the norm. ``axis``
        is accepted as an alias.
    keepdim : bool, optional
        Whether the reduced dimensions are kept (with length 1). The default is
        ``False``.
    """
    (dim, keepdim) = _dimargs('norm', kwargs, dim=dim, keepdim=keepdim)
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        raise _sparse_dense_error('norm')
    if torch.is_tensor(m):
        rmag = torch.linalg.norm(m, ord=ord, dim=dim, keepdim=keepdim)
    else:
        rmag = np.linalg.norm(m, ord=ord, axis=dim)
        if keepdim:
            if dim is None:
                rmag = np.reshape(rmag, (1,) * m.ndim)
            else:
                axes = dim if isinstance(dim, (tuple, list)) else (dim,)
                rmag = np.expand_dims(
                    rmag, tuple(ax % m.ndim for ax in axes))
    return quant(rmag, a.units)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def diag(a, offset=0):
    """Returns the diagonal of a 2-dimensional `a`, or a matrix with `a` on its
    diagonal if `a` is 1-dimensional, keeping the units.

    ``numpy.diag`` / ``torch.diag``.

    Parameters
    ----------
    offset : int, optional
        Which diagonal: ``0`` (the default) is the main one, positive is above
        it, and negative is below it.
    """
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        raise _sparse_dense_error('diag')
    rmag = (torch.diag(m, diagonal=offset) if torch.is_tensor(m)
            else np.diag(m, k=offset))
    return quant(rmag, a.units)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def diagonal(a, offset=0, dim1=0, dim2=1):
    """Returns the diagonals of `a` along the dimensions `dim1` and `dim2`,
    keeping the units (``numpy.diagonal`` / ``torch.diagonal``).

    Parameters
    ----------
    offset : int, optional
        Which diagonal to take; the default is ``0``, the main one.
    dim1 : int, optional
        The first dimension to take the diagonal of. The default is ``0``.
    dim2 : int, optional
        The second dimension to take the diagonal of. The default is ``1``.
    """
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        raise _sparse_dense_error('diagonal')
    rmag = (torch.diagonal(m, offset=offset, dim1=dim1, dim2=dim2)
            if torch.is_tensor(m)
            else np.diagonal(m, offset=offset, axis1=dim1, axis2=dim2))
    return quant(rmag, a.units)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def tril(a, diagonal=0):
    """Returns the lower triangle of `a`, keeping the units
    (``numpy.tril`` / ``torch.tril``).

    Parameters
    ----------
    diagonal : int, optional
        The diagonal above which to zero out the elements; the default is
        ``0``, the main diagonal.
    """
    a = quant(a)
    m = a.m
    rmag = (torch.tril(m, diagonal=diagonal) if torch.is_tensor(m)
            else np.tril(m, k=diagonal))
    return quant(rmag, a.units)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def triu(a, diagonal=0):
    """Returns the upper triangle of `a`, keeping the units
    (``numpy.triu`` / ``torch.triu``).

    Parameters
    ----------
    diagonal : int, optional
        The diagonal below which to zero out the elements; the default is
        ``0``, the main diagonal.
    """
    a = quant(a)
    m = a.m
    rmag = (torch.triu(m, diagonal=diagonal) if torch.is_tensor(m)
            else np.triu(m, k=diagonal))
    return quant(rmag, a.units)

@docwrap(format='numpy', inheritparams=_doc_params, inheritreturns=_doc_returns_quantity)
def trace(a):
    """Returns the sum of the diagonal of the 2-dimensional `a`, keeping the
    units (``numpy.trace`` / ``torch.trace``)."""
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        raise _sparse_dense_error('trace')
    rmag = torch.trace(m) if torch.is_tensor(m) else np.trace(m)
    return quant(rmag, a.units)

@docwrap(format='numpy', inheritparams=_doc_params)
def outer(a, b):
    """Returns the outer product of two 1-dimensional arguments, with the
    product of their units.

    ``numpy.outer`` / ``torch.outer``. Both arguments must be 1-dimensional
    (unlike ``numpy.outer``, which flattens its arguments, this raises for a
    higher-dimensional argument so that the two backends agree).

    Parameters
    ----------

    Returns
    -------
    immlib.Quantity
        The outer product, in the product of `a`'s and `b`'s units.
    """
    a = quant(a)
    b = quant(b)
    (ma, mb) = (a.m, b.m)
    if np.ndim(ma) != 1 or np.ndim(mb) != 1:
        raise ValueError(
            f"immlib.math.outer: both arguments must be 1-dimensional (got"
            f" {np.ndim(ma)} and {np.ndim(mb)} dimensions)")
    if torch.is_tensor(ma) or torch.is_tensor(mb):
        (ma, mb) = _promote_tensors((ma, mb))
        rmag = torch.outer(ma, mb)
    else:
        rmag = np.outer(ma, mb)
    return quant(rmag, _unit_product(a, b))

@docwrap(format='numpy', inheritparams=_doc_params)
def inner(a, b):
    """Returns the inner product of `a` and `b`, with the product of their
    units (``numpy.inner`` / ``torch.inner``).

    Parameters
    ----------

    Returns
    -------
    immlib.Quantity
        The inner product, in the product of `a`'s and `b`'s units.
    """
    a = quant(a)
    b = quant(b)
    (ma, mb) = (a.m, b.m)
    if torch.is_tensor(ma) or torch.is_tensor(mb):
        (ma, mb) = _promote_tensors((ma, mb))
        rmag = torch.inner(ma, mb)
    else:
        rmag = np.inner(ma, mb)
    return quant(rmag, _unit_product(a, b))

@docwrap(format='numpy', inheritparams=_doc_params)
def cross(a, b, dim=-1):
    """Returns the cross product of `a` and `b`, with the product of their
    units.

    ``numpy.cross`` / ``torch.linalg.cross``.

    Parameters
    ----------
    dim : int, optional
        The dimension along which the vectors lie; it must have length 3. The
        default is ``-1``.

    Returns
    -------
    immlib.Quantity
        The cross product, in the product of `a`'s and `b'`s units.
    """
    a = quant(a)
    b = quant(b)
    (ma, mb) = (a.m, b.m)
    if torch.is_tensor(ma) or torch.is_tensor(mb):
        (ma, mb) = _promote_tensors((ma, mb))
        rmag = torch.linalg.cross(ma, mb, dim=dim)
    else:
        rmag = np.cross(ma, mb, axisa=dim, axisb=dim, axisc=dim)
    return quant(rmag, _unit_product(a, b))

@docwrap(format='numpy', inheritparams=_doc_params)
def tensordot(a, b, dims=2):
    """Returns the tensor contraction of `a` and `b` over `dims` axes, with the
    product of their units (``numpy.tensordot`` / ``torch.tensordot``).

    Parameters
    ----------
    dims : int or sequence, optional
        The number of axes (or the specific axes) to contract. The default is
        ``2``.

    Returns
    -------
    immlib.Quantity
        The contraction, in the product of `a`'s and `b`'s units.
    """
    a = quant(a)
    b = quant(b)
    (ma, mb) = (a.m, b.m)
    if torch.is_tensor(ma) or torch.is_tensor(mb):
        (ma, mb) = _promote_tensors((ma, mb))
        rmag = torch.tensordot(ma, mb, dims=dims)
    else:
        rmag = np.tensordot(ma, mb, axes=dims)
    return quant(rmag, _unit_product(a, b))


@docwrap(format='numpy')
def searchsorted(a, v, side='left', *, sorter=None, right=None):
    """Returns the indices at which `v` would be inserted into the sorted `a`.

    ``numpy.searchsorted`` / ``torch.searchsorted``. `v` is converted into `a`'s
    units, and the result is plain integer indices (not a quantity). `a` must be
    1-dimensional, so that the two backends agree (``torch.searchsorted`` alone
    also accepts a batch of sorted sequences).

    Parameters
    ----------
    a : quantity or array or tensor of numbers
        The sorted sequence to search; it must be 1-dimensional.
    v : quantity or array or tensor or number
        The value or values to insert. A unit-less (or bare) value compared
        with a dimensional one raises ``pint.DimensionalityError``.
    side : {'left', 'right'}, optional
        Whether to use the first suitable location (``'left'``, the default) or
        the last (``'right'``).
    sorter : array or tensor of int or None, optional
        The indices that would sort `a`, when `a` is not already sorted.
    right : bool or None, optional
        An alias for `side`: ``right=True`` is ``side='right'`` and
        ``right=False`` is ``side='left'`` (``torch.searchsorted``'s spelling).

    Returns
    -------
    array or tensor of int
        The insertion indices.
    """
    if right is not None:
        if side != 'left':
            raise TypeError(
                "immlib.math.searchsorted: give either 'side' or 'right', not"
                " both")
        side = 'right' if right else 'left'
    if side not in ('left', 'right'):
        raise ValueError(
            f"immlib.math.searchsorted: side must be 'left' or 'right', not"
            f" {side!r}")
    (ma, mv, _u) = _align_units(a, v, 'searchsorted')
    if np.ndim(ma) != 1:
        raise ValueError(
            f"immlib.math.searchsorted: the sorted sequence must be"
            f" 1-dimensional (got {np.ndim(ma)} dimensions)")
    if _is_tensor_backed(ma, mv):
        (ma, mv) = _promote_tensors((ma, mv))
        return torch.searchsorted(ma, mv, right=(side == 'right'),
                                  sorter=sorter)
    return np.searchsorted(ma, mv, side=side, sorter=sorter)
