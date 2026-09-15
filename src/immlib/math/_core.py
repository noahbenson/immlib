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

from ..util._numeric import torch, torch_found
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

def _align_units(a, b, fname):
    """Returns ``(mag_a, mag_b, units)``, where `mag_b` has been converted
    into `a`'s units when both have real, compatible units. Raises when the
    two cannot be combined: one has units of ``None`` and the other has real
    units, or the two have dimensionally incompatible real units.
    """
    a = quant(a)
    b = quant(b)
    au = a.units
    bu = b.units
    if au is None and bu is None:
        return (a.m, b.m, None)
    if au is None or bu is None:
        raise TypeError(
            f"immlib.math.{fname}: cannot combine a unit-less quantity"
            f" (units=None) with a quantity that has real units")
    if au == bu:
        return (a.m, b.m, au)
    if not alike_units(au, bu):
        raise pint.DimensionalityError(au, bu)
    return (a.m, b.to(au).m, au)

def _reconcile_seq(seq, fname):
    """Returns ``(mags, units)`` for a sequence of quantities/arrays/tensors
    to be combined (``stack``/``concatenate``): every element must either
    share unit-less (``None``) status, or have mutually compatible real
    units (later elements are converted into the first element's units).
    """
    quants = [quant(x) for x in seq]
    if not quants:
        raise ValueError(f"immlib.math.{fname}: cannot combine an empty sequence")
    units0 = quants[0].units
    mags = [quants[0].m]
    for q in quants[1:]:
        if (q.units is None) != (units0 is None):
            raise TypeError(
                f"immlib.math.{fname}: cannot combine a unit-less quantity"
                f" (units=None) with one that has real units")
        if units0 is None:
            mags.append(q.m)
        elif q.units == units0:
            mags.append(q.m)
        else:
            if not alike_units(q.units, units0):
                raise pint.DimensionalityError(q.units, units0)
            mags.append(q.to(units0).m)
    return (mags, units0)

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
                return torch_fn(m, dim=dims, keepdim=True, **torch_kwargs)
            return torch_fn(m, **torch_kwargs)
        return torch_fn(m, dim=axis, keepdim=keepdims, **torch_kwargs)
    return np_fn(m, axis=axis, keepdims=keepdims, **np_kwargs)

def _unitless_elementwise(fname, np_fn, torch_name, a):
    # See _reduce_mag's docstring regarding why `torch_name` is a string,
    # looked up lazily, rather than an already-resolved `torch.foo`.
    a = quant(a)
    _require_unitless(a, fname)
    m = a.m
    rmag = getattr(torch, torch_name)(m) if torch.is_tensor(m) else np_fn(m)
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
    tensor (unit-aware: differing but compatible units are converted first;
    incompatible units, or a unit-less/real-units mismatch, compare unequal
    rather than raising)."""
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
    """Returns the elementwise maximum of `a` and `b`, converting `b` into
    `a`'s units first when both have real, compatible units."""
    (ma, mb, u) = _align_units(a, b, 'maximum')
    if _is_tensor_backed(ma, mb):
        (ma, mb) = promote(ma, mb)
        rmag = torch.maximum(ma, mb)
    else:
        rmag = np.maximum(ma, mb)
    return quant(rmag, u)

def minimum(a, b):
    """Returns the elementwise minimum of `a` and `b`; see ``maximum``."""
    (ma, mb, u) = _align_units(a, b, 'minimum')
    if _is_tensor_backed(ma, mb):
        (ma, mb) = promote(ma, mb)
        rmag = torch.minimum(ma, mb)
    else:
        rmag = np.minimum(ma, mb)
    return quant(rmag, u)

def where(cond, a, b):
    """Returns `a` where `cond` is true and `b` otherwise, elementwise;
    `cond` must be a plain (or unit-less-quantity) boolean array/tensor, and
    `a`/`b` are unit-aligned as in ``maximum``."""
    condm = mag(cond, None) if is_quant(cond) else cond
    (ma, mb, u) = _align_units(a, b, 'where')
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
    rmag = torch.sqrt(m) if torch.is_tensor(m) else np.sqrt(m)
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
    """Returns the elementwise ``arctan2(y, x)``; `y` and `x` must have
    unit-less or mutually compatible real units (which are then aligned as
    in ``maximum``, though the angle result is always unit-less)."""
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
    rmag = torch.floor(m) if torch.is_tensor(m) else np.floor(m)
    return quant(rmag, a.units)

def ceil(a):
    """Returns the elementwise ceiling of `a`, preserving units."""
    a = quant(a)
    m = a.m
    rmag = torch.ceil(m) if torch.is_tensor(m) else np.ceil(m)
    return quant(rmag, a.units)

def round(a, ndigits=0):
    """Returns `a` elementwise-rounded to `ndigits` decimal places (default
    0), preserving units."""
    a = quant(a)
    m = a.m
    if torch.is_tensor(m):
        rmag = torch.round(m, decimals=ndigits)
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
    if torch.is_tensor(m):
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
    rmag = m.reshape(shape) if torch.is_tensor(m) else np.reshape(m, shape)
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
    else:
        rmag = np.squeeze(m, axis=axis)
    return quant(rmag, a.units)

def stack(seq, axis=0):
    """Returns the quantities/arrays/tensors in `seq` stacked along a new
    `axis`; every element must be unit-less, or have mutually compatible
    real units (later elements are converted into the first element's
    units)."""
    (mags, u) = _reconcile_seq(seq, 'stack')
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
