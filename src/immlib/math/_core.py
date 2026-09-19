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

import builtins
import operator
from collections import namedtuple

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

def pow(a, b):
    """Returns ``a ** b``; see ``immlib.Quantity.__pow__``. `b` must be a
    plain number or a unit-less quantity unless `a` is itself unit-less.

    This is PyTorch's name for the operation; NumPy calls it ``power``,
    which PyTorch does not define and which is therefore not defined here.
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

def eq(a, b):
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

def equal(a, b):
    """Returns whether `a` and `b` have the same shape and equal elements,
    as a single ``bool``.

    This is ``torch.equal``'s meaning rather than ``numpy.equal``'s: the
    elementwise comparison is ``eq``. The one departure from
    ``torch.equal`` is that a difference of dtype alone does not make two
    otherwise-equal arguments unequal, since ``numpy`` has no such rule and
    the two backends must agree.

    Units are handled as in ``eq``, so quantities with compatible units are
    converted before comparing and incompatible ones are simply unequal.
    """
    r = eq(a, b)
    if np.shape(quant(a).m) != np.shape(quant(b).m):
        return False
    return builtins.bool(r.all())

def not_equal(a, b):
    """Returns the elementwise result of ``a != b``; see ``eq``."""
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

#: An alias of ``immlib.math.arcsin``, as in PyTorch.
asin = arcsin
#: An alias of ``immlib.math.arccos``, as in PyTorch.
acos = arccos
#: An alias of ``immlib.math.arctan``, as in PyTorch.
atan = arctan

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

#: An alias of ``immlib.math.arctan2``, as in PyTorch.
atan2 = arctan2

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

def round(a, decimals=0):
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

def sum(a, dim=None, keepdim=False, **kwargs):
    """Returns the sum of `a`'s elements (optionally along `dim`),
    preserving units."""
    (dim, keepdim) = _dimargs('sum', kwargs, dim=dim, keepdim=keepdim)
    a = quant(a)
    rmag = _reduce_mag(np.sum, 'sum', a.m, dim, keepdim)
    return quant(rmag, a.units)

def prod(a, dim=None, keepdim=False, **kwargs):
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

def mean(a, dim=None, keepdim=False, **kwargs):
    """Returns the mean of `a`'s elements (optionally along `dim`),
    preserving units."""
    (dim, keepdim) = _dimargs('mean', kwargs, dim=dim, keepdim=keepdim)
    a = quant(a)
    rmag = _reduce_mag(np.mean, 'mean', a.m, dim, keepdim)
    return quant(rmag, a.units)

#: The result of ``immlib.math.min`` when a dimension is given: the minimum
#: values, as an ``immlib.Quantity``, and the index of the first minimum
#: along that dimension, as a plain array or tensor of integers.
min_result = namedtuple('min', ('values', 'indices'))
#: The result of ``immlib.math.max`` when a dimension is given; see
#: ``immlib.math.min_result``.
max_result = namedtuple('max', ('values', 'indices'))

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

def min(a, dim=None, keepdim=False, **kwargs):
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
    """
    return _minmax('min', a, dim, keepdim, kwargs)

def max(a, dim=None, keepdim=False, **kwargs):
    """Returns the maximum of `a`'s elements, preserving units; see
    ``min``, whose behavior this mirrors (including the ``(values,
    indices)`` result when `dim` is given)."""
    return _minmax('max', a, dim, keepdim, kwargs)

def amin(a, dim=None, keepdim=False, **kwargs):
    """Returns the minimum of `a`'s elements (optionally along `dim`),
    preserving units, as an ``immlib.Quantity``--never the ``(values,
    indices)`` tuple that ``min`` returns for a given `dim`. Unlike ``min``,
    several dimensions may be reduced at once."""
    (dim, keepdim) = _dimargs('amin', kwargs, dim=dim, keepdim=keepdim)
    a = quant(a)
    return quant(_reduce_mag(np.amin, 'amin', a.m, dim, keepdim), a.units)

def amax(a, dim=None, keepdim=False, **kwargs):
    """Returns the maximum of `a`'s elements (optionally along `dim`),
    preserving units; see ``amin``."""
    (dim, keepdim) = _dimargs('amax', kwargs, dim=dim, keepdim=keepdim)
    a = quant(a)
    return quant(_reduce_mag(np.amax, 'amax', a.m, dim, keepdim), a.units)

def any(a, dim=None, keepdim=False, **kwargs):
    """Returns whether any of `a`'s elements are truthy (optionally along
    `dim`), as a plain bool array or tensor (not an ``immlib.Quantity``)."""
    (dim, keepdim) = _dimargs('any', kwargs, dim=dim, keepdim=keepdim)
    a = quant(a)
    return _reduce_mag(np.any, 'any', a.m, dim, keepdim)

def all(a, dim=None, keepdim=False, **kwargs):
    """Returns whether all of `a`'s elements are truthy (optionally along
    `dim`), as a plain bool array or tensor (not an ``immlib.Quantity``)."""
    (dim, keepdim) = _dimargs('all', kwargs, dim=dim, keepdim=keepdim)
    a = quant(a)
    return _reduce_mag(np.all, 'all', a.m, dim, keepdim)

def std(a, dim=None, keepdim=False, correction=1, **kwargs):
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

def var(a, dim=None, keepdim=False, correction=1, **kwargs):
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

def cumsum(a, dim, **kwargs):
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

def reshape(a, *shape):
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

def transpose(a, dim0, dim1):
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

def permute(a, *dims):
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

def squeeze(a, dim=None, **kwargs):
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

def unsqueeze(a, dim, **kwargs):
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

def ravel(a):
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

def flatten(a, start_dim=0, end_dim=-1):
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

def conj(a):
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

def stack(seq, dim=_UNSET, **kwargs):
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

def cat(seq, dim=_UNSET, **kwargs):
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
sort_result = namedtuple('sort', ('values', 'indices'))
#: The result of ``immlib.math.median`` when a dimension is given; see
#: ``immlib.math.sort_result``.
median_result = namedtuple('median', ('values', 'indices'))

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

def sort(a, dim=-1, descending=False, stable=False, **kwargs):
    """Returns `a`'s elements sorted along `dim`, preserving units.

    The result is a ``(values, indices)`` named tuple, as ``torch.sort``
    returns: `values` is an ``immlib.Quantity`` and `indices` is a plain
    array or tensor giving the index each value came from. Sorting is along
    the last dimension by default, and ascending unless `descending`;
    `stable` keeps equal elements in their original order.
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

def argsort(a, dim=-1, descending=False, stable=False, **kwargs):
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

def argmin(a, dim=None, keepdim=False, **kwargs):
    """Returns the index of `a`'s smallest element, as a plain integer array
    or tensor: the index along `dim`, or, when `dim` is not given, the index
    into the flattened input, as in both NumPy and PyTorch."""
    return _argminmax('argmin', a, dim, keepdim, kwargs)

def argmax(a, dim=None, keepdim=False, **kwargs):
    """Returns the index of `a`'s largest element; see ``argmin``."""
    return _argminmax('argmax', a, dim, keepdim, kwargs)

def median(a, dim=None, keepdim=False, **kwargs):
    """Returns the median of `a`'s elements, preserving units.

    ``median(a)`` returns the median element itself and ``median(a, dim)``
    returns a ``(values, indices)`` named tuple, as ``torch.median`` does.

    For an even number of elements this is the lower of the two middle
    values--an element of `a`, which is why it has an index--rather than
    their mean, which is what ``numpy.median`` returns. The lower value is
    used for both backends, since one behavior must be chosen for both;
    ``mean(sort(a).values[..., k:k+2])`` gives the interpolated median.
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

def quantile(a, q, dim=None, keepdim=False, interpolation='linear',
             **kwargs):
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
        rmag = np.quantile(m, q, axis=dim, keepdims=keepdim,
                           method=interpolation)
    return quant(rmag, a.units)

def percentile(a, q, dim=None, keepdim=False, interpolation='linear',
               **kwargs):
    """Returns the `q`-th percentile of `a`'s elements, preserving units;
    this is ``quantile(a, q / 100)``, the scale ``numpy.percentile`` uses.
    PyTorch has no ``percentile`` of its own."""
    q = np.asarray(q) / 100 if not torch.is_tensor(q) else q / 100
    return quantile(a, q, dim, keepdim, interpolation, **kwargs)

def ptp(a, dim=None, keepdim=False, **kwargs):
    """Returns the range of `a`'s elements--the largest minus the smallest,
    "peak to peak"--preserving units.

    ``numpy.ptp`` is the origin of the name; PyTorch has no equivalent, so
    this is computed as ``amax(a, dim) - amin(a, dim)`` for both backends,
    which keeps a tensor's gradient tracking.
    """
    (dim, keepdim) = _dimargs('ptp', kwargs, dim=dim, keepdim=keepdim)
    return amax(a, dim, keepdim) - amin(a, dim, keepdim)

def average(a, dim=None, weights=None, keepdim=False, **kwargs):
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

def unique(a, sorted=True, return_inverse=False, return_counts=False,
           dim=None, **kwargs):
    """Returns `a`'s distinct elements in order, preserving units.

    ``return_inverse`` and ``return_counts`` add the index of each input
    element among the distinct ones, and the number of times each distinct
    element occurs, as plain arrays or tensors; the result is then a tuple.
    `sorted` is accepted for ``torch.unique``'s sake and is always true, as
    it is in ``numpy.unique``.

    This is not differentiable in either backend: a tensor is detached, and
    the result carries no gradient.
    """
    (dim, _) = _dimargs('unique', kwargs, dim=dim)
    ((m,), u, like) = _setop_mags('unique', a)
    r = np.unique(m, return_inverse=return_inverse,
                  return_counts=return_counts, axis=dim)
    if not (return_inverse or return_counts):
        return _setop_result(r, u, like)
    return (_setop_result(r[0], u, like),
            *(_setop_result(x, Ellipsis, like) for x in r[1:]))

def union1d(a, b):
    """Returns the sorted, distinct elements of `a` and `b` together, in the
    units of the first argument that has them; see ``unique`` regarding
    gradients."""
    (mags, u, like) = _setop_mags('union1d', a, b)
    return _setop_result(np.union1d(*mags), u, like)

def intersect1d(a, b, assume_unique=False):
    """Returns the sorted, distinct elements common to `a` and `b`, in the
    units of the first argument that has them; see ``unique`` regarding
    gradients."""
    (mags, u, like) = _setop_mags('intersect1d', a, b)
    return _setop_result(
        np.intersect1d(*mags, assume_unique=assume_unique), u, like)

def setdiff1d(a, b, assume_unique=False):
    """Returns the sorted, distinct elements of `a` that are not in `b`, in
    the units of the first argument that has them; see ``unique`` regarding
    gradients."""
    (mags, u, like) = _setop_mags('setdiff1d', a, b)
    return _setop_result(
        np.setdiff1d(*mags, assume_unique=assume_unique), u, like)

def setxor1d(a, b, assume_unique=False):
    """Returns the sorted, distinct elements of exactly one of `a` and `b`,
    in the units of the first argument that has them; see ``unique``
    regarding gradients."""
    (mags, u, like) = _setop_mags('setxor1d', a, b)
    return _setop_result(
        np.setxor1d(*mags, assume_unique=assume_unique), u, like)

def isin(elements, test_elements, assume_unique=False, invert=False):
    """Returns, for each element of `elements`, whether it occurs in
    `test_elements`, as a plain bool array or tensor of `elements`' shape.
    The two are unit-aligned as in ``maximum``; see ``unique`` regarding
    gradients."""
    (mags, _u, like) = _setop_mags('isin', elements, test_elements)
    r = np.isin(*mags, assume_unique=assume_unique, invert=invert)
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

def gather(a, dim, index, **kwargs):
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

def index_select(a, dim, index, **kwargs):
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

def take(a, index):
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

def masked_select(a, mask):
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

def flip(a, dims=None, **kwargs):
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

def roll(a, shifts, dims=None, **kwargs):
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

def repeat_interleave(a, repeats, dim=None, **kwargs):
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

def tile(a, dims):
    """Returns `a` tiled `dims` times along each dimension, preserving
    units. This is ``torch.tile`` and ``numpy.tile``."""
    a = quant(a)
    m = a.m
    if sps.issparse(m):
        raise _sparse_dense_error('tile')
    dd = tuple(dims) if isinstance(dims, (tuple, list)) else (dims,)
    rmag = torch.tile(m, dd) if torch.is_tensor(m) else np.tile(m, dd)
    return quant(rmag, a.units)


# Linear algebra ##################################################################

def matmul(a, b):
    """Returns ``a @ b``; see ``immlib.Quantity.__matmul__``.

    See ``dot`` for the 1-D inner product.
    """
    return quant(a) @ quant(b)

def dot(a, b):
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
