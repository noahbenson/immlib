# -*- coding: utf-8 -*-
###############################################################################
# immlib/math/_core.py

"""Implementation of the ``immlib.math`` common numerical namespace.

``immlib.math`` is a small, deliberately common subset of NumPy/PyTorch
functionality that operates directly on ``immlib.Quantity`` objects (as well
as on plain NumPy arrays, PyTorch tensors, and Python numbers, each of which
is treated as a unit-less--``units=None``--quantity). See the design spec,
section 9, for the governing principles; in short:

  * the backend (NumPy or PyTorch) is selected per call: any tensor magnitude
    among the arguments selects PyTorch, otherwise NumPy is used;
  * every function returns an ``immlib.Quantity``, *except* a function whose
    natural result is a boolean or index array/tensor (the comparisons,
    ``any``, and ``all``), which instead returns a plain NumPy array or
    PyTorch tensor, matching ordinary NumPy/PyTorch ergonomics for masks and
    indexing;
  * units are computed directly from each function's mathematical meaning
    (unit-preserving, unitless-required, exponentiated, etc.)--not by
    delegating a whole ``Quantity`` to ``np.foo``/``torch.foo`` dispatch,
    since Pint's own NumPy dispatch machinery does not understand immlib's
    ``None`` ("no units", as opposed to Pint's real ``dimensionless``) unit
    convention and fails outright for it;
  * NumPy's public semantics and argument conventions (``axis``,
    ``keepdims``, the behavior of functions like ``transpose``) are the
    reference; where PyTorch's own same-named function differs (argument
    names such as ``dim``/``keepdim``, differing defaults such as
    ``std``/``var``'s degrees-of-freedom, or a differing result shape/type
    such as ``torch.min``'s ``(values, indices)`` tuple), the PyTorch side is
    implemented, per function, to match NumPy's behavior instead--using
    whichever underlying PyTorch call achieves that (e.g. ``Tensor.permute``
    for a NumPy-style ``transpose``, ``torch.amin``/``torch.amax`` rather
    than ``torch.min``/``torch.max`` for ``min``/``max``);
  * every magnitude computation on a tensor uses ordinary (differentiable)
    PyTorch tensor operations directly on the tensor--never a NumPy
    round-trip--so that gradients flow through ``immlib.math`` calls exactly
    as they would through the equivalent raw PyTorch code, for any function
    whose underlying operation is itself differentiable;
  * SciPy sparse magnitudes stay sparse wherever the operation preserves
    sparsity (elementwise functions that map 0 to 0, arithmetic, reshaping,
    and 2-D concatenation); a reduction along an axis (``sum``, ``mean``,
    ``min``, etc.) returns a dense NumPy array (or a NumPy scalar for a full
    reduction), as does a reduction of a sparse PyTorch tensor; and an
    operation whose result would be dense (e.g. ``exp``, ``cos``, or
    ``where``) raises ``TypeError`` rather than allocating a dense array
    implicitly (use ``immlib.to_dense`` first if that is what you want);
  * a function whose NumPy and PyTorch semantics differ too materially to
    unify (e.g. ``numpy.dot``, which behaves like matrix multiplication with
    broadcasting for 2-D-and-higher input, versus ``torch.dot``, which is
    restricted to a 1-D inner product) is simply not exposed here; use
    ``immlib.math.matmul`` (backed by ``Quantity.__matmul__``) or the
    backend's own function directly instead.
"""

import builtins
import operator

import numpy as np
import pint
import scipy.sparse as sps

from ..util._numeric import torch, torch_found
from ..util._core import unitregistry
from ..util._quantity import (Quantity, quant, mag, alike_units,
                               promote, is_quant)


# Helpers ######################################################################

def _is_tensor_backed(*mags):
    return builtins.any(torch.is_tensor(m) for m in mags)

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


# Elementwise arithmetic #######################################################

def abs(a):
    """Returns the elementwise absolute value of `a`, preserving units."""
    return builtins.abs(quant(a))

def add(a, b):
    """Returns ``a + b``; see ``immlib.Quantity``'s unit-aware addition."""
    return quant(a) + quant(b)

def subtract(a, b):
    """Returns ``a - b``; see ``immlib.Quantity``'s unit-aware subtraction."""
    return quant(a) - quant(b)

def multiply(a, b):
    """Returns ``a * b``; see ``immlib.Quantity``'s unit-aware multiplication."""
    return quant(a) * quant(b)

def divide(a, b):
    """Returns ``a / b``; see ``immlib.Quantity``'s unit-aware division."""
    return quant(a) / quant(b)

true_divide = divide

def power(a, b):
    """Returns ``a ** b``; see ``immlib.Quantity.__pow__``. `b` must be a
    plain number or a unit-less quantity unless `a` is itself unit-less.
    """
    return quant(a) ** b

def negative(a):
    """Returns ``-a``, preserving units."""
    return -quant(a)

def positive(a):
    """Returns ``+a``, preserving units."""
    return +quant(a)


# Comparisons ###################################################################
# Each of these returns a plain NumPy array or PyTorch tensor of bool, not an
# immlib.Quantity--see the module docstring.

def equal(a, b):
    """Returns the elementwise result of ``a == b`` as a plain bool array or
    tensor.

    The comparison is unit-aware: differing but compatible units are
    converted first, and incompatible units compare unequal (an all-false
    result) rather than raising. A unit-less (``units=None``) value is
    compared as a bare value would be, i.e., as a dimensionless value, so it
    is unequal to a quantity with real dimensions (except that, as in Pint,
    an all-zero or NaN bare value is compared by magnitude alone).
    """
    return quant(a) == quant(b)

def not_equal(a, b):
    """Returns the elementwise result of ``a != b``; see ``equal``."""
    return quant(a) != quant(b)

def less(a, b):
    """Returns the elementwise result of ``a < b`` (unit-aware; raises for
    dimensionally incompatible real units, per ``immlib.Quantity``)."""
    return quant(a) < quant(b)

def less_equal(a, b):
    """Returns the elementwise result of ``a <= b``; see ``less``."""
    return quant(a) <= quant(b)

def greater(a, b):
    """Returns the elementwise result of ``a > b``; see ``less``."""
    return quant(a) > quant(b)

def greater_equal(a, b):
    """Returns the elementwise result of ``a >= b``; see ``less``."""
    return quant(a) >= quant(b)

def maximum(a, b):
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

def minimum(a, b):
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

def where(cond, a, b):
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

def sqrt(a):
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

def exp(a):
    """Returns the elementwise exponential of `a`; `a` must be unit-less."""
    return _unitless_elementwise('exp', np.exp, 'exp', a)

def log(a):
    """Returns the elementwise natural log of `a`; `a` must be unit-less."""
    return _unitless_elementwise('log', np.log, 'log', a)

def log10(a):
    """Returns the elementwise base-10 log of `a`; `a` must be unit-less."""
    return _unitless_elementwise('log10', np.log10, 'log10', a)

def sin(a):
    """Returns the elementwise sine of `a`; `a` must be unit-less."""
    return _unitless_elementwise('sin', np.sin, 'sin', a)

def cos(a):
    """Returns the elementwise cosine of `a`; `a` must be unit-less."""
    return _unitless_elementwise('cos', np.cos, 'cos', a)

def tan(a):
    """Returns the elementwise tangent of `a`; `a` must be unit-less."""
    return _unitless_elementwise('tan', np.tan, 'tan', a)

def arcsin(a):
    """Returns the elementwise arcsine of `a`; `a` must be unit-less."""
    return _unitless_elementwise('arcsin', np.arcsin, 'asin', a)

def arccos(a):
    """Returns the elementwise arccosine of `a`; `a` must be unit-less."""
    return _unitless_elementwise('arccos', np.arccos, 'acos', a)

def arctan(a):
    """Returns the elementwise arctangent of `a`; `a` must be unit-less."""
    return _unitless_elementwise('arctan', np.arctan, 'atan', a)

def arctan2(y, x):
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

def floor(a):
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

def ceil(a):
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

def round(a, ndigits=0):
    """Returns `a` elementwise-rounded to `ndigits` decimal places (default
    0), preserving units."""
    a = quant(a)
    m = a.m
    if torch.is_tensor(m):
        rmag = torch.round(m, decimals=ndigits)
    elif sps.issparse(m):
        rmag = m.tocsr(copy=True)
        rmag.data = np.round(rmag.data, decimals=ndigits)
        rmag.eliminate_zeros()
    else:
        rmag = np.round(m, decimals=ndigits)
    return quant(rmag, a.units)


# Reductions #####################################################################

def sum(a, axis=None, keepdims=False):
    """Returns the sum of `a`'s elements (optionally along `axis`),
    preserving units."""
    a = quant(a)
    rmag = _reduce_mag(np.sum, 'sum', a.m, axis, keepdims)
    return quant(rmag, a.units)

def prod(a, axis=None, keepdims=False):
    """Returns the product of `a`'s elements (optionally along `axis`); the
    result's units are `a`'s units raised to the power of the number of
    elements combined into each output value (e.g. the product of 3
    quantities in meters has units of ``m**3``). A tuple/list `axis` is
    supported for a NumPy-backed quantity but not for a PyTorch-backed one
    (PyTorch's own ``prod`` only reduces a single axis at a time); reduce
    one axis at a time for a tensor-backed quantity instead.
    """
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        rmag = _sparse_reduce('prod', m, axis, keepdims)
    elif torch.is_tensor(m):
        if isinstance(axis, (tuple, list)):
            raise TypeError(
                "immlib.math.prod: multi-axis reduction (axis as a tuple or"
                " list) is not supported for tensor-backed quantities;"
                " reduce one axis at a time instead")
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
        rmag = np.prod(m, axis=axis, keepdims=keepdims)
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

def mean(a, axis=None, keepdims=False):
    """Returns the mean of `a`'s elements (optionally along `axis`),
    preserving units."""
    a = quant(a)
    rmag = _reduce_mag(np.mean, 'mean', a.m, axis, keepdims)
    return quant(rmag, a.units)

def min(a, axis=None, keepdims=False):
    """Returns the minimum of `a`'s elements (optionally along `axis`),
    preserving units. Unlike ``torch.min``, this always returns only the
    value(s)--matching ``numpy.min``--never a ``(values, indices)`` tuple.
    """
    a = quant(a)
    rmag = _reduce_mag(np.amin, 'amin', a.m, axis, keepdims)
    return quant(rmag, a.units)

def max(a, axis=None, keepdims=False):
    """Returns the maximum of `a`'s elements (optionally along `axis`),
    preserving units. See ``min``."""
    a = quant(a)
    rmag = _reduce_mag(np.amax, 'amax', a.m, axis, keepdims)
    return quant(rmag, a.units)

def any(a, axis=None, keepdims=False):
    """Returns whether any of `a`'s elements are truthy (optionally along
    `axis`), as a plain bool array or tensor (not an ``immlib.Quantity``)."""
    a = quant(a)
    return _reduce_mag(np.any, 'any', a.m, axis, keepdims)

def all(a, axis=None, keepdims=False):
    """Returns whether all of `a`'s elements are truthy (optionally along
    `axis`), as a plain bool array or tensor (not an ``immlib.Quantity``)."""
    a = quant(a)
    return _reduce_mag(np.all, 'all', a.m, axis, keepdims)

def std(a, axis=None, ddof=0, keepdims=False):
    """Returns the standard deviation of `a`'s elements (optionally along
    `axis`), preserving units. `ddof` (delta degrees of freedom; default 0,
    matching ``numpy.std``'s population-std default--note this differs from
    ``torch.std``'s own default of a Bessel-corrected sample std) is passed
    to PyTorch as ``correction``.
    """
    a = quant(a)
    rmag = _reduce_mag(np.std, 'std', a.m, axis, keepdims,
                        np_kwargs={'ddof': ddof},
                        torch_kwargs={'correction': ddof})
    return quant(rmag, a.units)

def var(a, axis=None, ddof=0, keepdims=False):
    """Returns the variance of `a`'s elements (optionally along `axis`); the
    result's units are `a`'s units squared. See ``std`` regarding `ddof`.
    """
    a = quant(a)
    rmag = _reduce_mag(np.var, 'var', a.m, axis, keepdims,
                        np_kwargs={'ddof': ddof},
                        torch_kwargs={'correction': ddof})
    u = None if a.units is None else (a.units ** 2)
    return quant(rmag, u)


# Shape / combination ############################################################

def reshape(a, shape):
    """Returns `a` reshaped to `shape`, preserving units."""
    a = quant(a)
    m = a.m
    if torch.is_tensor(m) or sps.issparse(m):
        rmag = m.reshape(shape)
    else:
        rmag = np.reshape(m, shape)
    return quant(rmag, a.units)

def transpose(a, axes=None):
    """Returns `a` with its axes permuted according to `axes` (or fully
    reversed, if `axes` is not given), matching ``numpy.transpose``'s
    semantics for both backends (a PyTorch tensor is permuted via
    ``Tensor.permute``, since ``torch.transpose`` itself only swaps a single
    pair of axes and has no ``numpy.transpose``-style default behavior).
    Units are preserved.
    """
    a = quant(a)
    m = a.m
    if torch.is_tensor(m):
        if axes is None:
            axes = tuple(reversed(range(m.dim())))
        rmag = m.permute(*axes)
    elif sps.issparse(m):
        rmag = m.transpose(axes)
    else:
        rmag = np.transpose(m, axes)
    return quant(rmag, a.units)

def squeeze(a, axis=None):
    """Returns `a` with size-1 axes removed (all of them, or only `axis` if
    given), preserving units."""
    a = quant(a)
    m = a.m
    if torch.is_tensor(m):
        rmag = m.squeeze() if axis is None else m.squeeze(axis)
    elif sps.issparse(m):
        raise TypeError(
            "immlib.math.squeeze: SciPy sparse arrays are not supported")
    else:
        rmag = np.squeeze(m, axis=axis)
    return quant(rmag, a.units)

def stack(seq, axis=0):
    """Returns the quantities/arrays/tensors in `seq` stacked along a new
    `axis`. If any element has real units, the result has the units of the
    first such element, and all elements are converted into them (unit-less
    elements are treated as dimensionless values, as in ``maximum``)."""
    (mags, u) = _reconcile_seq(seq, 'stack')
    if builtins.any(sps.issparse(x) for x in mags):
        raise TypeError(
            "immlib.math.stack: SciPy sparse arrays are not supported; use"
            " concatenate for 2-D sparse arrays")
    if _is_tensor_backed(*mags):
        mags = promote(*mags)
        rmag = torch.stack(mags, dim=axis)
    else:
        rmag = np.stack(mags, axis=axis)
    return quant(rmag, u)

def concatenate(seq, axis=0):
    """Returns the quantities/arrays/tensors in `seq` concatenated along an
    existing `axis`; see ``stack`` regarding units."""
    (mags, u) = _reconcile_seq(seq, 'concatenate')
    if builtins.any(sps.issparse(x) for x in mags):
        if _is_tensor_backed(*mags) or axis not in (0, 1, -1, -2) or (
                builtins.any(np.ndim(x) != 2 for x in mags)):
            raise TypeError(
                "immlib.math.concatenate: SciPy sparse arrays can only be"
                " concatenated with other 2-D NumPy or SciPy arrays along"
                " axis 0 or 1")
        combine = sps.vstack if axis in (0, -2) else sps.hstack
        rmag = combine(mags)
        return quant(rmag, u)
    if _is_tensor_backed(*mags):
        mags = promote(*mags)
        rmag = torch.cat(mags, dim=axis)
    else:
        rmag = np.concatenate(mags, axis=axis)
    return quant(rmag, u)


# Linear algebra ##################################################################

def matmul(a, b):
    """Returns ``a @ b``; see ``immlib.Quantity.__matmul__``.

    ``immlib.math.dot`` is intentionally not provided: ``numpy.dot`` and
    ``torch.dot`` have materially different semantics for anything beyond a
    1-D inner product (``numpy.dot`` behaves like broadcasting matrix
    multiplication for 2-D-and-higher input; ``torch.dot`` is restricted to
    a 1-D inner product and raises for anything else), so unifying them
    behind one name would mean silently different behavior by backend. Use
    ``immlib.math.matmul`` (or the ``@`` operator) instead.
    """
    return quant(a) @ quant(b)
