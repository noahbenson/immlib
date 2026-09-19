# -*- coding: utf-8 -*-
###############################################################################
# immlib/util/_quantity.py


# Dependencies ################################################################

import inspect
import operator
import warnings

import pint
import numpy as np
from docshare import docwrap
import scipy.sparse as sps

from ._core import (is_set, is_str, unitregistry, _default_ureg,
                    _default_ureg_override)
from ._numeric import (
    torch, alttorch, checktorch, scipy__is_sparse,
    is_array, is_tensor, is_numeric, is_sparse, to_sparse,
    to_array, to_tensor, to_numeric, to_sparse, to_dense)


# Units and Quantities ########################################################

# Units are fundamentally treated as part of the immlib type-system. Immlib
# functios that deal with an object's type typically take an option `unit` that
# can be used to change the function's behavior depending on the units attached
# to an object.
# Setup pint / units:
from pint import UnitRegistry
def is_ureg(obj):
    """Returns ``True`` if an object is a ``ping.UnitRegistry`` object.

    ``is_ureg(obj)`` returns ``True`` if the given object `obj` is an instance
    of the ``pint.UnitRegistry`` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an ``UnitRegistry`` object is to be
        assessed.

    Returns
    -------
    bool
        ``True`` if `obj` is an instance of ``UnitRegistry``, otherwise
        ``False``.
    """
    return isinstance(obj, pint.UnitRegistry)
from pint import Unit
def is_unit(q, /, *, ureg=None):
    """Returns ``True`` if `q` is a ``pint.Unit`` object and ``False``
    otherwise.

    ``is_unit(q)`` returns ``True`` if `q` is a unit object (of type
    ``pint.Unit``) and ``False`` otherwise.

    Parameters
    ----------
    q : object
        The object whose quality as a ``pint.Unit`` is to be assessed.

    ureg : UnitRegistry, Ellipsis, or None, optional

        The ``pint.UnitRegistry`` object that the given unit object must belong
        to. If ``None`` (the default), then any unit registry is allowed. If
        ``Ellipsis``, then the ``immlib.units`` registry is used. Otherwise,
        this must be a specific ``pint.UnitRegistry`` object.

    Returns
    -------
    bool
        ``True`` if `q` is a ``pint.Unit`` object and ``False`` otherwise.

    Raises
    ------
    TypeError
        If the ``ureg`` parameter is not a ``pint.UnitRegistry``, ``Ellipsis``,
        or ``None``.
    """
    if ureg is None:
        return isinstance(q, Unit)
    elif ureg is Ellipsis:
        units = _default_ureg()
        return isinstance(q, units.Unit)
    elif is_ureg(ureg):
        return isinstance(q, ureg.Unit)
    else:
        raise TypeError("parameter ureg must be a UnitRegistry")
@docwrap(format='numpy', inheritraises=is_unit)
def is_quant(obj, /, unit=Ellipsis, *, ureg=None):
    """Returns ``True`` if given a ``pint.Quantity`` object and ``False``
    otherwise.

    ``is_quant(obj)`` returns ``True`` if `obj` is a ``pint.Quantity`` object
    and ``False`` otherwise. The optional parameter `unit` may additionally
    specify a unit that `obj` must be compatible with.

    .. Note:: The parameter value ``unit=None`` matches an object that is not
        a quantity at all, as well as an ``immlib.Quantity`` whose own
        ``units`` is ``None`` (immlib's representation of "no units"; see
        ``immlib.Quantity``). It does not match a quantity with real units,
        including a dimensionless one.

    Parameters
    ----------
    obj : object
        The object whose quality as a ``pint.Quantity`` object is to be
        assessed.
    unit : unit-like or None, optional
        The unit that the object must have in order to be considered
        valid. This may be a ``pint.Unit`` or unit-name (see also
        ``immlib.unit``), a list or tuple of such units/unit-names, or
        ``None``. If ``Ellipsis`` is given (the default), then the object must
        be a ``pint.Quantity`` object, but it doesn't matter what the unit of
        the object is. Otherwise, the object must have a unit equivalent to the
        unit or to one of the units given (`unit` may be a tuple of possible
        units). The ``pint.UnitRegistry`` objects for the units given via this
        parameter are ignored; only the `ureg` parameter influences the
        ``pint.UnitRegistry`` requirements.
    ureg : pint.UnitRegistry, Ellipsis, None, optional
        The ``pint.UnitRegistry`` object to use for units. If ``Ellipsis``,
        then value ``immlib.units`` is used. If `ureg` is ``None`` (the
        default), then a specific unit registry is not checked.

    Returns
    -------
    bool
        ``True`` if `obj` is a ``pint.Quantity`` whose unit is compatible with
        the requested `unit` and ``False`` otherwise.

    Raises
    ------

    """
    if ureg is None:
        if not isinstance(obj, pint.Quantity):
            return False
    else:
        if ureg is Ellipsis:
            units = _default_ureg()
            ureg = units
        elif not is_ureg(ureg):
            raise TypeError("parameter ureg must be a UnitRegistry")
        if not isinstance(obj, ureg.Quantity):
            return False
    if unit is Ellipsis:
        return True
    elif unit is None:
        # obj is a quantity (already established above); it counts as
        # matching unit=None exactly when its own units are None.
        return obj.units is None
    elif obj.units is None:
        # obj has no units at all, so it cannot be compatible with a real
        # unit (querying compatibility itself isn't well-defined for it).
        return False
    else:
        return obj.is_compatible_with(unit)
class default_ureg:
    """Context manager for setting the default ``immlib`` unit registry.

    The following code-block can be used to evaluate the code represented by
    ``...`` using the unit-registry ``ureg`` in place of ``immlib.units`` as
    the default registry of ``immlib`` functions such as ``immlib.quant``:

    .. code-block:: python

       with immlib.default_ureg(ureg) as u:
           ...  # here, u is ureg

    Inside the block, ``immlib.units`` is ``ureg``. The override applies
    only to the current thread (or asynchronous task), so ``default_ureg``
    can be used safely by several threads at once. Blocks may be nested.
    Assigning ``immlib.units = ureg`` instead changes the default registry
    for every thread.
    """
    __slots__ = ('ureg', '_tokens')
    def __init__(self, ureg):
        if not is_ureg(ureg):
            raise TypeError("ureg must be a pint.UnitRegistry")
        object.__setattr__(self, 'ureg', ureg)
        object.__setattr__(self, '_tokens', [])
    def __enter__(self):
        self._tokens.append(_default_ureg_override.set(self.ureg))
        return self.ureg
    def __exit__(self, exc_type, exc_val, exc_tb):
        _default_ureg_override.reset(self._tokens.pop())
        return False
    def __setattr__(self, name, val):
        raise TypeError("cannot change the original units registry")
# The Quantity Type ###########################################################

# immlib.Quantity extends pint.Quantity with two features: a `.backend`
# property that reports whether the quantity's magnitude is a NumPy array (or
# SciPy sparse array/matrix) or a PyTorch tensor, and support for a magnitude
# that has no units at all, represented by `units` (`.u`) being `None`, as
# opposed to Pint's own `dimensionless`, which is a real, measurable unit.
#
# A `units` of `None` is intentionally treated as meaning "this is really just
# a NumPy array or PyTorch tensor with some Quantity bookkeeping attached to
# it, not a physical quantity": the arithmetic overloads below operate
# directly on the magnitude(s) whenever self's or the other operand's units
# are `None` (unwrapping any other `None`-unit operand the same way), so a
# `None`-unit quantity behaves like its magnitude for essentially every
# purpose. This is a deliberate divergence from Pint's own semantics (in
# which a magnitude always has some unit, even if that unit is
# dimensionless), which is why immlib.Quantity overloads several of
# pint.Quantity's own methods, including some private ones
# (`_add_sub`/`_iadd_sub`/`_mul_div`/`_imul_div`, the choke points pint's own
# +, -, *, and / dunder methods funnel through) as well as `to`, `ito`, and
# `m_as`. None of this changes how plain pint.Quantity objects behave, nor
# how a pint.UnitRegistry that immlib hasn't touched behaves; the changes
# here are purely additive and apply only to immlib.Quantity instances
# themselves (see `UnitRegistry`, below, regarding how a unit registry comes
# to produce immlib.Quantity objects in the first place: only an
# immlib.UnitRegistry does so--a plain pint.UnitRegistry is never modified by
# immlib and always continues to produce ordinary pint.Quantity objects).
# A private sentinel distinguishing "the units argument was not passed at
# all" from an explicit units=None; see Quantity.__new__.
_omitted = object()


def _immlib_matmul(a, b):
    """Computes ``a @ b`` for a matmul in which at least one side has a
    PyTorch tensor or SciPy sparse magnitude.

    This is handled separately from ordinary multiplication because
    ``torch.matmul`` cannot be routed through pint's own NumPy-oriented
    ufunc/function dispatch machinery--the default
    ``pint.Quantity.__matmul__`` implementation calls
    ``np.matmul(self, other)``--without risking a silent conversion of the
    tensor magnitude into a plain NumPy array (or an outright crash for a
    tensor that requires grad). See ``Quantity.__matmul__``/``__rmatmul__``.

    The non-tensor side, if any, is promoted to a tensor using the same
    array/tensor promotion rule used elsewhere in immlib (see
    ``immlib.promote``): the tensor side's device is authoritative. Its
    dtype is also matched to the tensor side's--not merely as a
    convenience but because torch.matmul (unlike NumPy) requires its two
    operands to share an exact dtype and raises a RuntimeError otherwise
    (e.g. a plain NumPy array, which defaults to float64, multiplied
    against a tensor created from a Python float literal, which defaults
    to float32).  When both sides are already tensors, no dtype
    reconciliation is attempted; torch.matmul's own error for a genuine
    dtype mismatch between two tensors is left as-is.
    """
    a_is_q = isinstance(a, pint.Quantity)
    b_is_q = isinstance(b, pint.Quantity)
    a_mag = a._magnitude if a_is_q else a
    b_mag = b._magnitude if b_is_q else b
    if torch.is_tensor(a_mag):
        if not torch.is_tensor(b_mag):
            b_mag = to_tensor(b_mag, device=a_mag.device, dtype=a_mag.dtype)
        elif b_mag.device != a_mag.device:
            b_mag = b_mag.to(device=a_mag.device)
        result_mag = torch.matmul(a_mag, b_mag)
    elif torch.is_tensor(b_mag):
        a_mag = to_tensor(a_mag, device=b_mag.device, dtype=b_mag.dtype)
        result_mag = torch.matmul(a_mag, b_mag)
    else:
        # At least one side is a SciPy sparse array/matrix, which NumPy's
        # matmul (used by Pint) does not accept; SciPy's own @ does.
        result_mag = a_mag @ b_mag
    # Units combine the same way they do for ordinary multiplication: a
    # None-units quantity or a non-quantity contributes no units at all
    # (see immlib.Quantity's None-unit semantics).
    a_none = (not a_is_q) or a._units is None
    b_none = (not b_is_q) or b._units is None
    if a_none and b_none:
        qcls = a.__class__ if a_is_q else b.__class__
        return qcls(result_mag, None)
    elif a_none:
        return b.__class__(result_mag, b._units)
    elif b_none:
        return a.__class__(result_mag, a._units)
    else:
        return a.__class__(result_mag, a._units * b._units)


# Torch function names that should behave exactly like the corresponding
# Python operator applied to the Quantity objects themselves--i.e.,
# torch.add(q1, q2) behaves exactly like q1 + q2, reusing all of the
# existing (already-tested) unit and None-units arithmetic above rather
# than duplicating it. Each entry maps a torch function name to the
# (forward, reflected) dunder method names on Quantity.
#
# These are invoked as plain method calls--type(q).__add__(q, other)--
# rather than by evaluating the actual operator expression (q + other) or
# calling the operator module (operator.add(q, other)), which would be
# equivalent. That distinction matters here: when `other` is itself a raw
# (non-Quantity) torch.Tensor, evaluating the operator expression calls
# Tensor's own dunder method first, and PyTorch implements Tensor's
# arithmetic dunders (including __matmul__) via the same overridable
# torch.* functions this dispatch mechanism itself hooks into--so
# Tensor.__matmul__(tensor, quantity) re-enters torch's __torch_function__
# dispatch (because the Quantity argument is still present) and calls
# straight back into this function with the same arguments, forever. This
# was an actual, reproduced infinite-recursion bug (RecursionError) for
# e.g. ``plain_tensor @ some_quantity``. Calling our own dunder directly,
# as a plain function call, sidesteps Tensor's operator overloading (and
# therefore torch's dispatch) entirely.
_TORCH_OPERATOR_METHODS = {
    'add': ('__add__', '__radd__'),
    'sub': ('__sub__', '__rsub__'),
    'subtract': ('__sub__', '__rsub__'),
    'mul': ('__mul__', '__rmul__'),
    'multiply': ('__mul__', '__rmul__'),
    'div': ('__truediv__', '__rtruediv__'),
    'divide': ('__truediv__', '__rtruediv__'),
    'true_divide': ('__truediv__', '__rtruediv__'),
    'matmul': ('__matmul__', '__rmatmul__'),
}
# Same idea, for the unary operators.
_TORCH_UNARY_OPERATOR_METHODS = {
    'neg': '__neg__',
    'negative': '__neg__',
    'abs': '__abs__',
    'absolute': '__abs__',
    'pos': '__pos__',
    'positive': '__pos__',
}
# Torch function names that apply directly to a single quantity's
# magnitude and leave its units unchanged: shape/view-type operations,
# the reductions (sum/mean/cumsum/amin/amax) whose output unit is, by
# ordinary physical reasoning, the same as the input's--exactly the rule
# pint itself applies to the NumPy equivalents (see op_units_output_ufuncs
# in pint's numpy_func module, which maps 'sum'/'cumsum' to its own
# identity "sum" unit-output category)--and the elementwise
# floor/ceil/round, whose result is always in the same units as the
# input regardless of what PyTorch's own (torch-native, not
# immlib.math's NumPy-style) keyword arguments happen to be, so they can
# be passed straight through to `func` without any translation. This
# deliberately excludes anything whose output unit differs from its
# input's (var/std/prod), which is handled below via
# _TORCH_MATH_REDUCTION_FUNCS, and torch.min/torch.max, which are
# deliberately left unhandled (see the note above
# _TORCH_MATH_REDUCTION_FUNCS).
_TORCH_UNIT_PRESERVING_FUNCS = frozenset([
    'reshape', 'transpose', 'permute', 'flatten', 'squeeze', 'unsqueeze',
    't', 'clone', 'detach', 'sum', 'mean', 'cumsum', 'amin', 'amax',
    'floor', 'ceil', 'round',
])
# Torch functions that combine a sequence of tensors/quantities along an
# existing or new dimension.
_TORCH_CONCAT_FUNCS = frozenset(['cat', 'concat', 'concatenate', 'stack'])
# Torch function names that are single- (or, for atan2, two-) argument,
# take no keyword arguments, and whose magnitude computation and unit
# handling exactly match an immlib.math function--of a different name in
# the inverse-trig/atan2 cases, since PyTorch's own naming
# ('asin'/'acos'/'atan'/'atan2') differs from NumPy's/immlib.math's
# ('arcsin'/'arccos'/'arctan'/'arctan2'). Safe to delegate directly since
# there are no PyTorch-native keyword arguments to translate (contrast
# _TORCH_MATH_REDUCTION_FUNCS below).
_TORCH_MATH_DELEGATE_FUNCS = {
    'exp': 'exp', 'log': 'log', 'log10': 'log10',
    'sin': 'sin', 'cos': 'cos', 'tan': 'tan',
    'asin': 'arcsin', 'acos': 'arccos', 'atan': 'arctan',
    'atan2': 'arctan2', 'sqrt': 'sqrt',
}
# Reduction functions whose output unit is *not* the input's own units
# (var/prod), or which otherwise need immlib.math's unit handling rather
# than PyTorch's. immlib.math takes PyTorch's own argument names and
# defaults, so the arguments of a call such as torch.var(q, dim=0,
# correction=0) are simply passed along.
#
# torch.min and torch.max are included: immlib.math.min/max return the same
# ``(values, indices)`` pair that PyTorch's own do (with the values as a
# quantity), so the contract PyTorch documents holds for a Quantity too.
_TORCH_MATH_REDUCTION_FUNCS = {
    'std': 'std', 'var': 'var', 'prod': 'prod',
    'min': 'min', 'max': 'max', 'amin': 'amin', 'amax': 'amax',
    'sum': 'sum', 'mean': 'mean', 'cumsum': 'cumsum'}
# Reductions whose result is a plain bool tensor, never a Quantity (see
# the immlib.math module docstring regarding comparisons/any/all).
_TORCH_BOOL_REDUCTION_FUNCS = frozenset(['any', 'all'])

# NumPy ufunc/function names whose calling convention (positional
# arguments, NumPy-style keyword names such as axis/keepdims/ddof)
# already matches the corresponding immlib.math function of the same
# name, so __array_ufunc__/__array_function__ (below) can delegate to it
# directly. This is what lets e.g. np.exp(q) work correctly for a
# units=None quantity, where Pint's own NumPy dispatch (inherited
# unchanged from ordinary pint.Quantity, and written without any
# awareness of immlib's None convention) otherwise crashes outright with
# an unrelated internal error--see the immlib.math module docstring.
# Everything not in this set, and any call shape this set can't handle
# (see _numpy_math_dispatch), falls back to Pint's own existing
# implementation unchanged, so real-unit behavior for anything outside
# this curated, tested subset is not affected.
#
# Deliberately excludes matmul and every operator-mirroring/comparison
# function (add, subtract, multiply, divide, power, negative, positive,
# absolute, equal, not_equal, less, less_equal, greater, greater_equal):
# each of those immlib.math functions is implemented as the
# corresponding Python expression on Quantity objects (e.g.
# immlib.math.multiply is literally ``quant(a) * quant(b)``, and
# immlib.math.matmul is ``quant(a) @ quant(b)``), so delegating them
# here would evaluate that expression again from inside
# __array_ufunc__ itself. For matmul, and for the *=/None-units path
# through _binop_none (which combines a bare, already-unwrapped
# magnitude with the *other*, still-whole operand), that re-evaluation
# lands right back on the very same NumPy ufunc/array-function call
# that got us here, recursing forever--a real, reproduced
# RecursionError for cases including plain array-backed matmul
# (``qaa @ qab``) and a None-units quantity multiplied by a real-unit
# one (``none_q * real_q``). Those names are instead handled directly
# by _NUMPY_OPERATOR_METHODS/_NUMPY_UNARY_OPERATOR_METHODS below (which
# call the Quantity's own dunder method as a plain function, exactly
# mirroring how _TORCH_OPERATOR_METHODS above avoids the analogous
# recursion for PyTorch), or, for matmul, left to Pint's own existing
# __array_ufunc__/Quantity.__matmul__ handling, which needs no
# None-units-specific delegation at all.
_NUMPY_MATH_DELEGATE_FUNCS = frozenset([
    'maximum', 'minimum', 'where',
    'sqrt', 'exp', 'log', 'log10', 'sin', 'cos', 'tan',
    'arcsin', 'arccos', 'arctan', 'arctan2',
    'floor', 'ceil', 'round',
    'sum', 'prod', 'mean', 'min', 'max', 'any', 'all', 'std', 'var',
    'reshape', 'transpose', 'squeeze', 'stack', 'concatenate',
])

def _numpy_math_dispatch(name, args, kwargs):
    """Delegates a NumPy ufunc/function call to the matching
    ``immlib.math`` function, when one exists and the call's argument
    shape (positional-argument count, keyword-argument names) actually
    matches that function's signature; returns ``NotImplemented``
    otherwise (an unrecognized name, or a call shape immlib.math's
    function doesn't accept--e.g. an ``out=`` ufunc keyword), so that the
    caller falls back to Pint's own existing NumPy dispatch. A call whose
    shape *does* match but that fails for a legitimate domain reason
    (e.g. ``np.exp`` on a real-unit quantity) is deliberately not caught
    here: that informative error should propagate exactly as it would
    from calling the immlib.math function directly, not be swallowed
    into a fallback.
    """
    if name not in _NUMPY_MATH_DELEGATE_FUNCS:
        return NotImplemented
    from ..math import _core as _immath
    fn = getattr(_immath, name, None)
    if fn is None:
        return NotImplemented
    try:
        inspect.signature(fn).bind(*args, **kwargs)
    except TypeError:
        return NotImplemented
    return fn(*args, **kwargs)


def _map_quantities(obj, fn):
    """Returns a copy of `obj` in which every ``pint.Quantity`` found inside
    (possibly nested) lists, tuples, and dicts is replaced by ``fn(q)``."""
    if isinstance(obj, pint.Quantity):
        return fn(obj)
    elif isinstance(obj, (list, tuple)):
        res = [_map_quantities(x, fn) for x in obj]
        if isinstance(obj, tuple):
            return type(obj)(*res) if hasattr(obj, '_fields') else tuple(res)
        return res
    elif isinstance(obj, dict):
        return {k: _map_quantities(v, fn) for (k,v) in obj.items()}
    else:
        return obj
#: The PyTorch name for a NumPy ufunc that PyTorch spells differently. Most
#: are spelled the same (PyTorch defines NumPy's names as aliases), so only
#: the exceptions are listed. Note ``equal``: NumPy's ufunc is elementwise,
#: and PyTorch's elementwise comparison is ``eq``--``torch.equal`` is the
#: whole-array test instead.
_TORCH_UFUNC_NAMES = {
    'equal': 'eq',
    'power': 'pow',
    'conjugate': 'conj',
    'invert': 'bitwise_not',
    'degrees': 'rad2deg',
    'radians': 'deg2rad',
    'left_shift': 'bitwise_left_shift',
    'right_shift': 'bitwise_right_shift'}

def _torch_ufunc(ufunc):
    """Returns the PyTorch function equivalent to the given NumPy ufunc, or
    ``None`` if PyTorch has none.

    A NumPy ufunc cannot simply be applied to a tensor: NumPy converts the
    tensor with ``Tensor.__array__``, which computes a NumPy array and so
    both changes the backend and discards gradient tracking--and raises
    outright for a tensor that requires grad. The equivalent PyTorch
    function is used instead.
    """
    name = _TORCH_UFUNC_NAMES.get(ufunc.__name__, ufunc.__name__)
    return getattr(torch, name, None)

def _none_ufunc(ufunc, method, inputs, kwargs):
    """Applies a NumPy ufunc (with any method: ``__call__``, ``reduce``,
    ``accumulate``, ``outer``, etc., and any keywords, including ``out``)
    directly to the magnitudes of its arguments when every quantity among
    them has units of ``None``; returns ``NotImplemented`` otherwise.

    Numerical results are returned as quantities with units of ``None``,
    except for boolean results, which are returned as plain arrays (as for
    the comparisons in ``immlib.math``). If ``out`` is given, the output
    quantities themselves are returned.
    """
    quants = []
    _map_quantities((inputs, kwargs), quants.append)
    if not quants or any(q._units is not None for q in quants):
        return NotImplemented
    unwrap = lambda q: q._magnitude
    mags = _map_quantities(inputs, unwrap)
    kw = _map_quantities(kwargs, unwrap)
    torch_fn = None
    if method == '__call__' and any(torch.is_tensor(m) for m in mags):
        torch_fn = _torch_ufunc(ufunc)
    if torch_fn is not None:
        result = torch_fn(*mags, **kw)
    else:
        result = getattr(ufunc, method)(*mags, **kw)
    out = kwargs.get('out')
    if out is not None:
        if isinstance(out, tuple):
            return out[0] if len(out) == 1 else out
        return out
    qcls = quants[0].__class__
    def wrap(r):
        if isinstance(r, (np.ndarray, np.generic)):
            if r.dtype == np.bool_:
                return r
            return qcls(r, None)
        if torch.is_tensor(r):
            # A tensor result is wrapped exactly as an array result is, so
            # that the two backends agree about what comes back.
            if r.dtype == torch.bool:
                return r
            return qcls(r, None)
        return r
    if isinstance(result, tuple):
        return tuple(wrap(r) for r in result)
    return wrap(result)
def _pint_numpy_dispatch(call, args, kwargs):
    """Calls Pint's own NumPy dispatch, ``call(args, kwargs)``, after
    adapting any quantities with units of ``None``, which Pint does not
    understand.

    If every quantity among the arguments has units of ``None``, they are
    passed to Pint as dimensionless quantities, and every dimensionless
    quantity in Pint's result is converted back to units of ``None`` (so
    Pint still decides which results are quantities, e.g. ``np.cumsum``
    versus ``np.argmax``). Otherwise, the quantities with units of ``None``
    are also passed to Pint as dimensionless quantities, which is how Pint
    treats bare values (the same rule as for ``immlib.Quantity``'s
    operators), and the result is returned as Pint computes it. (Passing
    bare arrays instead would be equivalent, but some Pint versions' NumPy
    functions, e.g. ``np.dot`` in Pint 0.24, fail when given a bare array
    alongside a quantity.)
    """
    quants = []
    _map_quantities((args, kwargs), quants.append)
    nones = [q for q in quants if q._units is None]
    if not nones:
        return call(args, kwargs)
    def to_dimless(q):
        if q._units is None:
            return q._REGISTRY.Quantity(q._magnitude, 'dimensionless')
        return q
    (args, kwargs) = _map_quantities((args, kwargs), to_dimless)
    result = call(args, kwargs)
    if len(nones) < len(quants):
        return result
    def to_none(q):
        if isinstance(q, Quantity) and q._units == q.UnitsContainer():
            return q.__class__(q._magnitude, None)
        return q
    return _map_quantities(result, to_none)


# NumPy ufunc names that should behave exactly like the corresponding
# Python operator applied to the Quantity objects themselves--the NumPy
# analogue of _TORCH_OPERATOR_METHODS above, and for exactly the same
# reason: calling the Quantity's own dunder method directly, as a plain
# function--type(q).__mul__(q, other)--rather than letting
# immlib.math's wrapper re-evaluate the full expression (which is what
# caused the recursion described in the comment above
# _NUMPY_MATH_DELEGATE_FUNCS), sidesteps NumPy's ufunc-override
# protocol entirely for this call. It still correctly restores
# None-units support for the explicit ``np.add(q, ...)``-style
# function-call form (Pint's own, unmodified __array_ufunc__
# implementation for these ufuncs assumes real, dimensional units and
# crashes outright for a units=None operand--the same class of gap
# __array_ufunc__ exists to close for np.exp/np.sqrt/etc--and, for the
# comparison ufuncs, additionally corrects a real semantic
# inconsistency: Pint's own default array_ufunc for e.g. 'equal'
# compares bare magnitudes without regard to units at all, so
# ``np.equal(none_q, real_q)`` would otherwise disagree with
# ``none_q == real_q``, which is None-units-aware via
# Quantity.__eq__/_binop_none_bool).
_NUMPY_OPERATOR_METHODS = {
    'add': ('__add__', '__radd__'),
    'subtract': ('__sub__', '__rsub__'),
    'multiply': ('__mul__', '__rmul__'),
    'divide': ('__truediv__', '__rtruediv__'),
    'power': ('__pow__', '__rpow__'),
    'equal': ('__eq__', '__eq__'),
    'not_equal': ('__ne__', '__ne__'),
    'less': ('__lt__', '__gt__'),
    'less_equal': ('__le__', '__ge__'),
    'greater': ('__gt__', '__lt__'),
    'greater_equal': ('__ge__', '__le__'),
}
# (np.divide and np.true_divide are the same ufunc object, whose
# __name__ is always 'divide', so no separate 'true_divide' entry is
# needed above.) Same idea as _NUMPY_OPERATOR_METHODS, for the unary
# operators.
_NUMPY_UNARY_OPERATOR_METHODS = {
    'negative': '__neg__',
    'positive': '__pos__',
    'absolute': '__abs__',
}


def _check_out_persistent(kwargs):
    """Raises a ``TypeError`` if the ``out`` argument in `kwargs` names a
    persistent quantity.

    NumPy's ``out=`` argument writes into the object it is given, so it is
    one of the ways a quantity's magnitude can be changed in place. It is
    checked here, before any dispatch, because the quantity that ``out``
    names need not be one of the inputs and need not be the ``self`` whose
    ``__array_ufunc__`` or ``__array_function__`` NumPy chose to call.
    """
    out = kwargs.get('out')
    if out is None:
        return
    if not isinstance(out, tuple):
        out = (out,)
    for o in out:
        if isinstance(o, Quantity) and o._persistent:
            raise o._refuse("writing into it with a NumPy out= argument")

_REFLECTED_COMPARISONS = {
    operator.lt: operator.gt, operator.le: operator.ge,
    operator.gt: operator.lt, operator.ge: operator.le,
    operator.eq: operator.eq, operator.ne: operator.ne}

def _promote_mags(xm, ym):
    """Promotes two bare magnitudes as ``immlib.promote`` does (a non-tensor
    becomes a tensor on the tensor's device when the other is a tensor)."""
    if torch.is_tensor(xm):
        if not torch.is_tensor(ym):
            ym = to_tensor(ym, device=xm.device)
    elif torch.is_tensor(ym):
        xm = to_tensor(xm, device=ym.device)
    return (xm, ym)

def _hashable_mag(mag):
    if isinstance(mag, np.ndarray) and mag.ndim == 0:
        return mag.item()
    return mag

def _compare_units_error(units):
    """Returns the error for ordering a value that has no units against a
    quantity that has real dimensions.

    A quantity whose units are ``None``, like any bare value, is
    dimensionless for the purposes of a comparison, so ordering it against
    a dimensional quantity is a dimensionality failure. Pint raises a
    ``ValueError`` naming its own internal class for this in one operand
    order and a ``DimensionalityError`` in the other; immlib always raises
    the latter, which is also what ``immlib.math.maximum`` and friends raise
    for the same units.
    """
    return pint.DimensionalityError(
        'dimensionless', units,
        extra_msg=(" A value with no units cannot be ordered against a"
                   " quantity with dimensions."))

def _math():
    """Returns the ``immlib.math`` module.

    The import is made here, on demand, rather than at the top of this
    module: ``immlib.math`` is built on the ``Quantity`` class defined
    below, so it cannot be imported while this module is being defined.
    """
    from .. import math as _math_module
    return _math_module

def _unpickle_quantity(mag, units, persistent=False):
    ureg = _default_ureg()
    if not issubclass(ureg.Quantity, Quantity):
        ureg = _initial_global_ureg
    q = ureg.Quantity(mag, units)
    # `persistent` has a default because pickles written before
    # Quantity.persist existed call this function with two arguments.
    return q.persist() if persistent else q


class Quantity(pint.Quantity):
    """The ``immlib`` extension of ``pint.Quantity``.

    ``immlib.Quantity`` is a subclass of ``pint.Quantity`` that adds a
    ``backend`` property (``numpy`` or ``torch``, depending on the type of the
    quantity's magnitude) and support for a ``units``/``u`` value of
    ``None``, which ``immlib`` uses to represent "no units" (as opposed to
    Pint's own ``dimensionless``, a real, measurable unit). A quantity whose
    ``units`` is ``None`` behaves, for essentially all purposes, like its own
    magnitude: arithmetic between two ``None``-unit quantities (or between a
    ``None``-unit quantity and a plain array/tensor/number) is performed
    directly on the magnitude(s) and produces another ``None``-unit quantity,
    while arithmetic between a ``None``-unit quantity and a quantity with
    real units defers entirely to the real-unit quantity's own rules
    (exactly as if the ``None``-unit operand had been replaced by its bare
    magnitude).

    Every ``immlib.Quantity`` is associated with a specific
    ``pint.UnitRegistry`` (see ``immlib.unitregistry``); specifically, it is
    associated with an ``immlib.UnitRegistry`` (see below), which is the only
    kind of registry that produces ``immlib.Quantity`` objects. A plain
    ``pint.UnitRegistry`` is never modified by immlib--not even one passed
    explicitly via the ``ureg`` option of functions like ``immlib.quant``--so
    code that shares such a registry with immlib continues to receive
    ordinary ``pint.Quantity`` objects from it, unaffected by immlib's
    presence elsewhere in the process. ``isinstance(q, pint.Quantity)``
    remains ``True`` for every ``immlib.Quantity``, since ``immlib.Quantity``
    is (and must remain) a subclass of ``pint.Quantity``.

    .. Note:: Because ``immlib.Quantity`` allows ``units`` to be ``None``, it
        cannot be assumed to be interchangeable with every function that
        expects an ordinary ``pint.Quantity``; code that specifically
        inspects or relies on ``.units``/``.u`` should be prepared for a
        ``None`` value.

    .. Warning:: Like ``pint.Quantity``, ``immlib.Quantity`` is mutable, but
        its mutating features are strongly discouraged: the in-place
        operators (``+=``, ``*=``, etc.), the ``ito``-family of methods
        (``ito``, ``ito_base_units``, ``ito_reduced_units``,
        ``ito_root_units``, ``ito_preferred``), item assignment
        (``q[k] = v``), and in-place NumPy methods such as ``fill`` and
        ``put``. These change a quantity for every part of a program that
        refers to it, and they are not thread-safe: some of them replace the
        magnitude and the units one after the other, so another thread can
        briefly observe the new magnitude with the old units. Prefer the
        equivalent non-mutating forms (``q = q + x``, ``q.to(u)``, etc.),
        which return new quantities.

        The ``persist`` method makes a quantity immutable in place, and is
        the way to be sure none of this happens to a quantity that is
        shared. Newly created quantities are mutable, so that ``pint``'s
        own internal operations continue to work; a persistent quantity
        can be copied into a new, mutable quantity with ``quant(q.m,
        q.u)``, but never made mutable again.
    """
    __slots__ = ()
    def __new__(cls, value, units=_omitted):
        if units is _omitted:
            # No units argument was given at all: defer entirely to pint's
            # own default handling (dimensionless, or string-expression
            # parsing). This matters because pint's own internal code calls
            # things like self.Quantity(1) (e.g. while parsing the literal
            # word "dimensionless", or an empty expression string) relying
            # on the omitted-argument default meaning dimensionless; only an
            # *explicit* units=None--which is all immlib's own code ever
            # passes--means immlib's "no units" (see below).
            return super().__new__(cls, value)
        if units is None and isinstance(value, str):
            # Preserve pint's own string-expression parsing (e.g.,
            # Quantity('5 mm')); pint already handles units=None specially
            # for strings, and we don't want to interfere with that.
            return super().__new__(cls, value, units)
        if units is None:
            # Build a Quantity with no units at all. We extract the raw
            # magnitude (without performing any unit conversion) and
            # construct an ordinary (dimensionless) quantity from it using
            # pint's own machinery, in order to reuse pint's magnitude
            # coercion (e.g., list -> array); we then simply mark the result
            # as unitless. Note that this never converts a pre-existing
            # quantity's magnitude--if value already has real units, those
            # units are dropped, not converted from.
            mag = value._magnitude if isinstance(value, pint.Quantity) else value
            inst = super().__new__(cls, mag, None)
            inst._units = None
            return inst
        return super().__new__(cls, value, units)
    # Matching argument types -------------------------------------------
    def as_input_type(self, *args):
        """Returns this quantity if any argument is a quantity, and its
        magnitude otherwise.

        ``q.as_input_type(a, b, ...)`` returns ``q`` itself if any of the
        arguments ``a``, ``b``, etc. is a ``pint.Quantity`` (including an
        ``immlib.Quantity``); otherwise it returns ``q``'s magnitude (a NumPy
        array, PyTorch tensor, or SciPy sparse array). With no arguments, the
        magnitude is returned.

        This is intended for the end of a numerical function that converts its
        arguments into quantities (e.g. with ``immlib.math.quant``) so that it
        can support quantities, arrays, and tensors uniformly, but that should
        return a plain array or tensor when it was called with plain arrays or
        tensors.

        Examples
        --------
        >>> import immlib.math as im
        >>> def hypot(a, b):
        ...     (qa, qb) = (im.quant(a, 'mm'), im.quant(b, 'mm'))
        ...     return im.sqrt(qa**2 + qb**2).as_input_type(a, b)
        >>> hypot(3.0, 4.0)
        array(5.)
        >>> hypot(im.quant(3.0, 'cm'), 4.0)
        <Quantity(30.265491900843113, 'millimeter')>
        """
        for arg in args:
            if isinstance(arg, pint.Quantity):
                return self
        return self._magnitude
    # Backend ------------------------------------------------------------
    @property
    def backend(self):
        """``numpy`` if this quantity's magnitude is a NumPy array or a
        SciPy sparse array/matrix, or ``torch`` if it is a PyTorch tensor.
        """
        return torch if torch.is_tensor(self._magnitude) else np
    # Persistence ---------------------------------------------------------
    #: Whether this quantity has been made immutable. It is a class
    #: attribute so that every quantity has it without paying for an
    #: instance attribute it does not need; ``persist`` sets an instance
    #: attribute that shadows it.
    _persistent = False

    @property
    def is_persistent(self):
        """Whether this quantity has been made immutable by ``persist``."""
        return self._persistent
    def persist(self):
        """Makes this quantity immutable, and returns it.

        A persistent quantity's ``units`` and ``magnitude`` refer to the
        same unit and the same array or tensor for the rest of its life.
        Anything that would change either of them is refused: ``ito``,
        ``ito_root_units``, ``ito_base_units``, item assignment, attribute
        assignment, and a NumPy ``out=`` argument naming it all raise a
        ``TypeError``.

        The in-place *operators* are not refused. ``q += 1`` on a
        persistent quantity computes a new quantity and rebinds the name to
        it, leaving the original untouched, exactly as ``n += 1`` does for
        an ``int`` and for every other immutable Python object. Code
        written for mutable quantities therefore keeps working, without
        mutating anything another reference can see.

        .. Note:: This method is named ``persist`` rather than
            ``persistent``, which is what ``pcollections`` calls the
            corresponding method, because it does something different. A
            transient collection's ``d.persistent()`` returns a *new*,
            persistent collection and leaves ``d`` alone; a quantity's
            ``q.persist()`` changes ``q`` itself into a persistent
            quantity, and returns it only so that it can be called in an
            expression. The different name is meant to keep the two from
            being confused.

        What is frozen is this quantity: its units and which array or
        tensor is its magnitude. The *contents* of that magnitude are not,
        since a quantity does not own them--``q.m[0] = 5`` still works, as
        mutating a list stored in a ``pcollections.pdict`` still works.
        Freeze a NumPy magnitude with ``immlib.freezearray`` before
        persisting the quantity if the values must not change either;
        PyTorch has no equivalent, which is the other reason this method
        does not attempt it.

        Copying a persistent quantity, with ``copy.copy``,
        ``copy.deepcopy`` or ``pickle``, gives a persistent quantity.
        ``immlib.quant(q.m, q.u)`` gives an equal quantity that is not.

        **Thread safety.** A persistent quantity can be read from any
        number of threads at once, including in a free-threaded
        interpreter, without a lock. This is what persisting one is for:
        the mutating methods assign the magnitude and the units one after
        the other, so while any of them can run, another thread can see
        the new magnitude with the old units--a value wrong by whatever
        the conversion factor was, not merely stale. Persisting the
        quantity removes that window by removing the assignments.

        Two things are outside that guarantee. The first is the contents
        of the magnitude, as above: a NumPy array that another thread
        writes to is a data race whoever holds it, which
        ``immlib.freezearray`` is the answer to. The second is
        ``persist`` itself, which is a write like any other--persist a
        quantity before sharing it with other threads, not afterwards.

        Returns
        -------
        immlib.Quantity
            This quantity, now persistent.
        """
        if not self._persistent:
            # Pint computes `dimensionality` lazily and caches it on the
            # quantity itself, which is a write--the one write a
            # persistent quantity would otherwise still make, and it
            # would make it on whichever thread asked first. Doing it
            # here means the object's __dict__ stops changing at the
            # moment it becomes persistent. A quantity with units of None
            # has no dimensionality (immlib's property raises), and a
            # failure to precompute a cache is never a reason for
            # persist to fail, so the attempt is guarded.
            try:
                self.dimensionality
            except Exception:
                pass
            # The assignment must happen last, or __setattr__ would
            # refuse the line above.
            self.__dict__['_persistent'] = True
        return self
    def _refuse(self, what):
        """Returns the error raised when `what` is attempted on a
        persistent quantity."""
        return TypeError(
            f"{what} is not possible for a persistent quantity; see"
            f" immlib.Quantity.persist. Use immlib.quant(q.m, q.u) for a"
            f" quantity with the same value that can be changed")
    #: The private attributes that a persistent quantity refuses to have
    #: assigned. Other private attributes are allowed through because Pint
    #: uses them for values it computes lazily and caches: the
    #: ``dimensionality`` property computes ``self._dimensionality`` the
    #: first time it is asked for and assigns it, so refusing every private
    #: assignment would make ``q + q`` raise on a persistent quantity. A
    #: cache is not part of the quantity's value; the two names here are.
    _FROZEN_ATTRS = frozenset(['_magnitude', '_units', '_persistent'])
    @classmethod
    def _is_frozen_attr(cls, name):
        """Returns whether `name` is an attribute a persistent quantity
        refuses to have assigned or deleted."""
        return name in cls._FROZEN_ATTRS or not name.startswith('_')
    def __setattr__(self, name, value):
        if self._persistent and self._is_frozen_attr(name):
            raise self._refuse(f"setting the attribute '{name}'")
        return super().__setattr__(name, value)
    def __delattr__(self, name):
        if self._persistent and self._is_frozen_attr(name):
            raise self._refuse(f"deleting the attribute '{name}'")
        return super().__delattr__(name)
    def __copy__(self):
        r = super().__copy__()
        return r.persist() if self._persistent else r
    def __deepcopy__(self, memo):
        r = super().__deepcopy__(memo)
        return r.persist() if self._persistent else r

    # Units of None ------------------------------------------------------
    @property
    def units(self):
        """This quantity's units, or ``None`` if it has none.

        ``None`` is immlib's "no units at all", which is not Pint's
        ``dimensionless``--a real, measurable unit. See ``immlib.Quantity``.
        """
        if self._units is None:
            return None
        return self._REGISTRY.Unit(self._units)
    @property
    def u(self):
        "An alias of ``units``."
        return self.units
    @property
    def dimensionless(self):
        """Whether this quantity's units are dimensionless.

        A quantity with no units (``units is None``) is *not*
        dimensionless: it has no dimension to be trivial, so this is
        ``False`` for it, where Pint's own property would be ``True``.
        """
        if self._units is None:
            return False
        return super().dimensionless
    @property
    def dimensionality(self):
        """This quantity's dimensionality, as a mapping of base dimension
        to exponent.

        Raises
        ------
        TypeError
            If this quantity has no units (``units is None``), which has no
            dimensionality rather than a trivial one.
        """
        if self._units is None:
            raise TypeError(
                "quantity has no units (units is None); dimensionality is"
                " undefined for a unitless immlib quantity")
        return super().dimensionality
    @docwrap(format='numpy', inheritparams=pint.Quantity.check)
    def check(self, dimension):
        """Returns whether this quantity's dimensionality matches
        `dimension`; a quantity with no units matches nothing, not even a
        dimensionless one.

        Parameters
        ----------

        Returns
        -------
        bool
            Whether the dimensionality matches.
        """
        if self._units is None:
            return False
        return super().check(dimension)
    @docwrap(format='numpy', inheritparams=pint.Quantity.to)
    def to(self, other=None, *contexts, **ctx_kwargs):
        """Returns this quantity converted into the units `other`.

        Converting to or from ``None`` units is always possible and never
        changes the magnitude: ``q.to(None)`` strips whatever units `q` has
        and ``q.to('mm')``, for a `q` with no units, simply attaches them.
        A context cannot be given for either of those, since no conversion
        takes place.

        Parameters
        ----------

        Returns
        -------
        immlib.Quantity
            This quantity in the units `other`.

        Raises
        ------
        ValueError
            If a context is given while converting to or from ``None``
            units.
        """
        if self._units is None or other is None:
            if contexts or ctx_kwargs:
                raise ValueError(
                    "to: contexts are not supported when converting to or"
                    " from a unitless (units=None) quantity")
            return self.__class__(self._magnitude, other)
        return super().to(other, *contexts, **ctx_kwargs)
    @docwrap(format='numpy', inheritparams=pint.Quantity.to)
    def ito(self, other=None, *contexts, **ctx_kwargs):
        """Converts this quantity into the units `other` in place, and
        returns ``None``.

        This is one of the mutating operations that ``immlib.Quantity``
        keeps from Pint but discourages; see ``immlib.Quantity``. ``to``
        returns a new quantity instead, and is the only choice for a
        persistent quantity, which refuses this.
        """
        if self._persistent:
            raise self._refuse("converting the units in place (ito)")
        if self._units is None or other is None or self._is_0d():
            new = self.to(other, *contexts, **ctx_kwargs)
            self._magnitude = new._magnitude
            self._units = new._units
            return None
        return super().ito(other, *contexts, **ctx_kwargs)
    def __str__(self):
        if self._units is None:
            return str(self._magnitude)
        return super().__str__()
    def __repr__(self):
        # Pint's own __repr__ unconditionally quotes the units part of the
        # repr (`f"...'{self._units}'..."`), which, when self._units is
        # None, renders it as the *string* "'None'"--indistinguishable at
        # a glance from a real unit literally named "None". Swap in the
        # bare (unquoted) None literal instead, without otherwise
        # touching Pint's own (version-dependent) magnitude-formatting
        # rules: fall back to Pint's raw repr unchanged if its format
        # ever stops matching the trailing "'None')>" this expects.
        body = super().__repr__()
        if self._units is None:
            # Pint < 0.26 formats reprs as <Quantity(mag, 'units')>; Pint
            # 0.26 formats them as Quantity(mag, "units").
            for (quoted, bare) in (("'None')>", 'None)>'),
                                   ('"None")', 'None)')):
                if body.endswith(quoted):
                    return body[:-len(quoted)] + bare
        return body
    # Pint's own format mini-language recognizes a handful of extra
    # "format type" letters/modifiers, beyond Python's standard mini-
    # language, that control how *units* are rendered ('Lx'/'L' for LaTeX,
    # 'H' for HTML, 'P' for pretty, 'C' for compact, 'D' for the explicit
    # default, plus the '~' short-unit modifier and the '#' compact-unit
    # flag). None of these mean anything for a None-units quantity -- there
    # are no units to render -- but Pint's own machinery still passes them
    # in: notably, pint.util.PrettyIPython._repr_latex_ (inherited by
    # Quantity, and invoked automatically by Jupyter/IPython whenever a
    # quantity is displayed) always formats with spec 'L'. Passing 'L'
    # straight through to a bare NumPy array or Python number raises
    # TypeError ("unsupported format string"), since it isn't part of
    # Python's own mini-language, so these flags must be stripped first.
    # 'Lx' is listed before 'L' so it is removed as a whole rather than
    # leaving a stray 'x' behind.
    _PINT_FORMAT_FLAGS = ('Lx', 'L', 'H', 'P', 'C', 'D', '~', '#')
    def __format__(self, spec):
        if self._is_0d():
            # See the note on 0-dimensional magnitudes, below.
            return format(
                self.__class__(self._magnitude[()], self._units), spec)
        if self._units is None:
            mspec = spec
            for flag in self._PINT_FORMAT_FLAGS:
                mspec = mspec.replace(flag, '')
            try:
                return format(self._magnitude, mspec)
            except (TypeError, ValueError):
                # Fall back to a plain string if even the stripped spec
                # doesn't apply to this magnitude (e.g. an exotic spec we
                # didn't anticipate); this should be rare in practice.
                return str(self._magnitude)
        return super().__format__(spec)
    def m_as(self, units):
        """Returns this quantity's magnitude in the units `units`.

        ``q.m_as(None)`` returns the bare magnitude, whatever units `q` has,
        symmetric with ``q.to(None)``.

        Parameters
        ----------
        units : unit-like or None
            The units to express the magnitude in.

        Returns
        -------
        object
            The magnitude, as a NumPy array, a PyTorch tensor, or whatever
            else this quantity's magnitude is.

        Raises
        ------
        ValueError
            If this quantity has no units and real `units` are requested;
            use ``to`` to attach units, or ``magnitude`` for the raw value.
        """
        if units is None:
            # Requesting "no units" always succeeds and returns the bare
            # magnitude, symmetric with .to(None); this is true whether
            # self already has no units or has real ones.
            return self._magnitude
        if self._units is None:
            raise ValueError(
                "m_as: quantity has no units (units is None); use .to() to"
                " attach units or .magnitude/.m to obtain the raw value")
        return super().m_as(units)
    def __bool__(self):
        if self._units is None:
            return bool(self._magnitude)
        return super().__bool__()
    __nonzero__ = __bool__

    def ito_root_units(self):
        """Converts this quantity into its root units in place, and returns
        ``None``; ``to_root_units`` returns a new quantity instead."""
        if self._persistent:
            raise self._refuse(
                "converting the units in place (ito_root_units)")
        return super().ito_root_units()
    def ito_base_units(self):
        """Converts this quantity into its base units in place, and returns
        ``None``; ``to_base_units`` returns a new quantity instead."""
        if self._persistent:
            raise self._refuse(
                "converting the units in place (ito_base_units)")
        return super().ito_base_units()

    # Numerical methods ######################################################
    # Pint implements the methods below against NumPy arrays: it converts the
    # magnitude with numpy.asarray, then wraps the result up again. For a
    # tensor magnitude that silently changes the backend (q.reshape() on a
    # tensor quantity returned a NumPy array), which breaks the rule that a
    # method must give equal results for equal arrays and tensors, and it
    # destroys gradient tracking--for a tensor that requires grad it does not
    # merely lose the graph, it raises. They are therefore implemented here
    # by delegating to immlib.math, which honors both rules, and they take
    # immlib.math's (i.e. PyTorch's) arguments: `dim` and `keepdim`, with
    # `axis` and `keepdims` accepted as aliases.
    def sum(self, dim=None, keepdim=False, **kwargs):
        "Returns the sum of this quantity's elements; see ``immlib.math.sum``."
        return _math().sum(self, dim, keepdim, **kwargs)
    def prod(self, dim=None, keepdim=False, **kwargs):
        "Returns the product of the elements; see ``immlib.math.prod``."
        return _math().prod(self, dim, keepdim, **kwargs)
    def mean(self, dim=None, keepdim=False, **kwargs):
        "Returns the mean of the elements; see ``immlib.math.mean``."
        return _math().mean(self, dim, keepdim, **kwargs)
    def std(self, dim=None, keepdim=False, correction=1, **kwargs):
        """Returns the standard deviation of the elements; see
        ``immlib.math.std``, including the default ``correction=1``."""
        return _math().std(self, dim, keepdim, correction, **kwargs)
    def var(self, dim=None, keepdim=False, correction=1, **kwargs):
        "Returns the variance of the elements; see ``immlib.math.var``."
        return _math().var(self, dim, keepdim, correction, **kwargs)
    def min(self, dim=None, keepdim=False, **kwargs):
        """Returns the minimum element, or, for a given `dim`, a ``(values,
        indices)`` tuple; see ``immlib.math.min``."""
        return _math().min(self, dim, keepdim, **kwargs)
    def max(self, dim=None, keepdim=False, **kwargs):
        "Returns the maximum element; see ``immlib.math.max`` and ``min``."
        return _math().max(self, dim, keepdim, **kwargs)
    def amin(self, dim=None, keepdim=False, **kwargs):
        "Returns the minimum value(s) alone; see ``immlib.math.amin``."
        return _math().amin(self, dim, keepdim, **kwargs)
    def amax(self, dim=None, keepdim=False, **kwargs):
        "Returns the maximum value(s) alone; see ``immlib.math.amax``."
        return _math().amax(self, dim, keepdim, **kwargs)
    def all(self, dim=None, keepdim=False, **kwargs):
        """Returns whether every element is truthy, as a plain bool array or
        tensor; see ``immlib.math.all``."""
        return _math().all(self, dim, keepdim, **kwargs)
    def any(self, dim=None, keepdim=False, **kwargs):
        """Returns whether any element is truthy, as a plain bool array or
        tensor; see ``immlib.math.any``."""
        return _math().any(self, dim, keepdim, **kwargs)
    def cumsum(self, dim, **kwargs):
        """Returns the cumulative sum along `dim`, which is required, as it is
        in ``torch.cumsum``; see ``immlib.math.cumsum``."""
        return _math().cumsum(self, dim, **kwargs)
    def round(self, decimals=0):
        "Returns the elements rounded; see ``immlib.math.round``."
        return _math().round(self, decimals)
    def conj(self):
        "Returns the elementwise conjugate; see ``immlib.math.conj``."
        return _math().conj(self)
    def conjugate(self):
        "An alias of ``conj``."
        return _math().conj(self)
    def reshape(self, *shape):
        "Returns this quantity reshaped; see ``immlib.math.reshape``."
        return _math().reshape(self, *shape)
    def transpose(self, dim0=None, dim1=None):
        """Returns this quantity with two dimensions exchanged; see
        ``immlib.math.transpose``. Given no arguments, every dimension is
        reversed, as ``permute`` does with none--the one concession to
        ``numpy.ndarray.transpose``, whose no-argument form is common."""
        if dim0 is None and dim1 is None:
            return _math().permute(self)
        return _math().transpose(self, dim0, dim1)
    def permute(self, *dims):
        "Returns this quantity with its dimensions permuted; see ``permute``."
        return _math().permute(self, *dims)
    def squeeze(self, dim=None, **kwargs):
        "Returns this quantity with size-1 dimensions removed; see ``squeeze``."
        return _math().squeeze(self, dim, **kwargs)
    def unsqueeze(self, dim, **kwargs):
        "Returns this quantity with a dimension inserted; see ``unsqueeze``."
        return _math().unsqueeze(self, dim, **kwargs)
    def ravel(self):
        "Returns this quantity flattened; see ``immlib.math.ravel``."
        return _math().ravel(self)
    def flatten(self, start_dim=0, end_dim=-1):
        "Returns this quantity flattened; see ``immlib.math.flatten``."
        return _math().flatten(self, start_dim, end_dim)
    def astype(self, dtype, **kwargs):
        """Returns this quantity with its magnitude cast to `dtype`, keeping
        its backend: a tensor magnitude stays a tensor (PyTorch spells this
        ``Tensor.to``, a name Pint has already given to unit conversion). The
        dtype may be given in either library's spelling."""
        from ._numeric import to_numpydtype, to_torchdtype
        m = self._magnitude
        if torch.is_tensor(m):
            return self.__class__(m.to(to_torchdtype(dtype), **kwargs),
                                  self.units)
        return self.__class__(m.astype(to_numpydtype(dtype), **kwargs),
                              self.units)
    def __round__(self, ndigits=None):
        mag = self._magnitude
        if self._is_0d():
            mag = np.asarray(round(mag[()], ndigits))
        elif isinstance(mag, np.ndarray):
            mag = np.round(mag, 0 if ndigits is None else ndigits)
        elif torch.is_tensor(mag):
            mag = torch.round(mag, decimals=0 if ndigits is None else ndigits)
        else:
            mag = round(mag, ndigits)
        return self.__class__(mag, self._units)
    def __int__(self):
        if self._units is None:
            return int(self._magnitude)
        return super().__int__()
    def __float__(self):
        if self._units is None:
            return float(self._magnitude)
        return super().__float__()
    def __complex__(self):
        if self._units is None:
            return complex(self._magnitude)
        return super().__complex__()
    # 0-dimensional magnitudes -------------------------------------------
    # quant() stores scalars as 0-dimensional NumPy arrays. Pint treats any
    # ndarray magnitude as an array, which is wrong for a few scalar-like
    # behaviors: in-place operators and ito() modify the array in place
    # (which fails under NumPy's casting rules for an integer array, e.g.
    # ``q *= 2.5``, and would also modify any other reference to the
    # array), round() is not defined for arrays, and Pint's LaTeX/HTML
    # formatting renders the array as an (empty) matrix. For a 0-d magnitude
    # we therefore behave as Pint does for a Python scalar: in-place
    # operators compute a new magnitude and rebind it, and formatting uses
    # the scalar the array contains.
    def _is_0d(self):
        mag = self._magnitude
        return isinstance(mag, np.ndarray) and mag.ndim == 0
    def _inplace_0d(self, op, other):
        result = op(self, other)
        if result is NotImplemented:
            return result
        if isinstance(result, pint.Quantity):
            self._magnitude = result._magnitude
            self._units = result._units
        else:
            self._magnitude = result
            self._units = None
        return self
    def __iadd__(self, other):
        if self._is_0d():
            return self._inplace_0d(operator.add, other)
        return super().__iadd__(other)
    def __isub__(self, other):
        if self._is_0d():
            return self._inplace_0d(operator.sub, other)
        return super().__isub__(other)
    def __imul__(self, other):
        if self._is_0d():
            return self._inplace_0d(operator.mul, other)
        return super().__imul__(other)
    def __itruediv__(self, other):
        if self._is_0d():
            return self._inplace_0d(operator.truediv, other)
        return super().__itruediv__(other)
    __idiv__ = __itruediv__
    def __ifloordiv__(self, other):
        if self._persistent:
            return self.__floordiv__(other)
        if self._has_none_units(other):
            return self._inplace_result(
                self._binop_none(other, operator.floordiv))
        if self._is_0d():
            return self._inplace_0d(operator.floordiv, other)
        return super().__ifloordiv__(other)
    def __imod__(self, other):
        if self._persistent:
            return self.__mod__(other)
        if self._has_none_units(other):
            return self._inplace_result(
                self._binop_none(other, operator.mod))
        if self._is_0d():
            return self._inplace_0d(operator.mod, other)
        return super().__imod__(other)
    # Arithmetic -----------------------------------------------------------
    # When self or other has units of None, we delegate directly to the
    # magnitude(s): a None-unit quantity behaves exactly like its magnitude
    # for arithmetic purposes (see the module-level discussion above). Both
    # `_add_sub`/`_iadd_sub` (+ and -) and `_mul_div`/`_imul_div` (* and /)
    # are the choke points that pint.Quantity's own dunder methods funnel
    # through, so overloading these four covers +, -, *, and / (in place and
    # not) without needing to overload each dunder method individually. `**`
    # has no single choke point in pint (`__pow__`/`__ipow__`/`__rpow__` are
    # each self-contained), so each is overloaded directly below; likewise
    # `__eq__`/`__ne__` are self-contained, while `<`, `<=`, `>`, and `>=`
    # all funnel through `compare()`, which is overloaded once for all four.
    # (`@` is handled separately, above `__torch_function__` below; NumPy and
    # PyTorch dispatch protocol None-awareness are left to a later phase.)
    def _none_operands(self, other):
        """Returns the operands ``(x, y)`` to use for ``op(self, other)`` when
        either operand has units of ``None``.

        A quantity whose units are ``None`` is replaced by its bare magnitude,
        and the bare magnitudes are promoted together (see
        ``immlib.promote``) so that a NumPy value meeting a tensor becomes a
        tensor on that tensor's device. Thus ``op(quant(x, None), q)`` is
        equivalent to ``(x, y) = promote(x, q.m); op(x, quant(y, q.u))``. A
        quantity with real units keeps them (with its promoted magnitude).
        """
        other_is_q = isinstance(other, pint.Quantity)
        xm = self._magnitude
        ym = other._magnitude if other_is_q else other
        if torch.is_tensor(xm) != torch.is_tensor(ym) and (
                is_numeric(ym) or isinstance(ym, (list, tuple))):
            (xm, ym) = _promote_mags(xm, ym)
        if self._units is None:
            x = xm
        elif xm is self._magnitude:
            x = self
        else:
            x = self.__class__(xm, self._units)
        if not other_is_q or other._units is None:
            y = ym
        elif ym is other._magnitude:
            y = other
        else:
            y = other.__class__(ym, other._units)
        return (x, y)
    def __setitem__(self, key, value):
        """Assigns into this quantity's magnitude, converting `value` into
        this quantity's units first.

        Pint's own implementation is used for a NumPy magnitude with real
        units. It cannot be used for the other two cases: for a unit-less
        quantity it computes ``self._units ** -1``, which has no meaning
        here, and for a tensor magnitude it assigns a NumPy value into the
        tensor. Both fail with a ``TypeError`` that Pint reports as "does
        not support indexing".
        """
        if self._persistent:
            raise self._refuse("item assignment")
        m = self._magnitude
        value_is_none_q = (isinstance(value, pint.Quantity)
                           and value._units is None)
        if (self._units is not None and not torch.is_tensor(m)
                and not value_is_none_q):
            return super().__setitem__(key, value)
        if self._units is None:
            # A unit-less quantity assigns like its bare magnitude would.
            if isinstance(value, pint.Quantity):
                if value._units is not None:
                    raise pint.DimensionalityError(
                        value.units, 'dimensionless',
                        extra_msg=(" Cannot assign a quantity with units"
                                   " into a unit-less (units=None)"
                                   " quantity."))
                value = value._magnitude
        else:
            # Convert value into this quantity's units. A bare value is
            # dimensionless, as everywhere else, so assigning one into a
            # quantity with real dimensions raises.
            if isinstance(value, pint.Quantity) and value._units is not None:
                q = value
            else:
                v = value._magnitude if isinstance(value, pint.Quantity) \
                    else value
                q = self.__class__(v, self.UnitsContainer())
            value = q.m_as(self._units)
        # The value must be in the magnitude's own backend.
        if torch.is_tensor(m):
            if not torch.is_tensor(value):
                value = to_tensor(value, dtype=m.dtype, device=m.device)
        elif torch.is_tensor(value):
            value = to_array(value)
        m[key] = value
    def _has_none_units(self, other):
        """Returns whether this quantity or `other` has units of ``None``,
        in which case the operation is performed on bare magnitudes."""
        return self._units is None or (
            isinstance(other, pint.Quantity) and other._units is None)
    def _rbinop_none(self, other, op):
        "As ``_binop_none``, for a reflected operator (``op(other, self)``)."
        return self._binop_none(other, lambda x, y: op(y, x))
    def _divmod(self, other, reflected=False):
        """Implements ``divmod`` as the pair ``(self // other, self %
        other)``.

        Pint's own implementation calls ``divmod`` on the magnitudes, which
        a PyTorch tensor does not support although it supports both ``//``
        and ``%``; computing the two separately is the same answer for a
        NumPy magnitude and works for both (Rule 1).
        """
        (x, y) = (other, self) if reflected else (self, other)
        return (x // y, x % y)
    def __floordiv__(self, other):
        if self._has_none_units(other):
            return self._binop_none(other, operator.floordiv)
        return super().__floordiv__(other)
    def __rfloordiv__(self, other):
        if self._has_none_units(other):
            return self._rbinop_none(other, operator.floordiv)
        return super().__rfloordiv__(other)
    def __mod__(self, other):
        if self._has_none_units(other):
            return self._binop_none(other, operator.mod)
        return super().__mod__(other)
    def __rmod__(self, other):
        if self._has_none_units(other):
            return self._rbinop_none(other, operator.mod)
        return super().__rmod__(other)
    def __divmod__(self, other):
        return self._divmod(other)
    def __rdivmod__(self, other):
        return self._divmod(other, reflected=True)
    def _binop_none(self, other, op):
        (x, y) = self._none_operands(other)
        result = op(x, y)
        if result is NotImplemented:
            return result
        if isinstance(result, pint.Quantity):
            if isinstance(result, type(self)):
                return result
            qcls = self._REGISTRY.Quantity
            return qcls(result._magnitude, result._units)
        return self.__class__(result, None)
    # Real-units operations involving a PyTorch tensor. Pint's arithmetic and
    # comparison code checks a bare operand with `zero_or_nan(other, True)`,
    # which cannot reduce a tensor to a single bool (Pint's
    # `is_duck_array_type` does not recognize tensors) and so fails with
    # "only one element tensors can be converted to Python scalars". We
    # therefore promote NumPy/tensor magnitudes (as immlib.promote does) and
    # replace a bare tensor operand with an equivalent quantity before
    # deferring to Pint: an all-zero/NaN tensor takes this quantity's units
    # (Pint allows e.g. `q + 0` for any units) and any other tensor is
    # dimensionless (as Pint treats bare values).
    def _real_tensor_operands(self, other, compare=False):
        other_is_q = isinstance(other, pint.Quantity)
        xm = self._magnitude
        ym = other._magnitude if other_is_q else other
        if torch.is_tensor(xm) == torch.is_tensor(ym):
            if not torch.is_tensor(xm):
                return None
            promoted = False
        elif is_numeric(ym) or isinstance(ym, (list, tuple)):
            (xm, ym) = _promote_mags(xm, ym)
            promoted = True
        else:
            return None
        x = self.__class__(xm, self._units) if promoted else self
        if other_is_q:
            y = other.__class__(ym, other._units) if promoted else other
        else:
            if bool(((ym == 0) | torch.isnan(ym)).all()) and (
                    self._is_multiplicative):
                y = self.__class__(ym, self._units)
            elif compare and not self.dimensionless:
                raise _compare_units_error(self.units)
            else:
                y = self.__class__(ym, self.UnitsContainer())
        return (x, y, promoted)
    _INPLACE_OPS = {
        operator.iadd: operator.add, operator.isub: operator.sub,
        operator.imul: operator.mul, operator.itruediv: operator.truediv,
        operator.ifloordiv: operator.floordiv}
    def _inplace_result(self, result):
        self._magnitude = result._magnitude
        self._units = result._units
        return self
    def _add_sub(self, other, op):
        if self._units is None or (
                isinstance(other, pint.Quantity) and other._units is None):
            return self._binop_none(other, op)
        ops = self._real_tensor_operands(other)
        if ops is not None:
            (x, y, _) = ops
            return pint.Quantity._add_sub(x, y, op)
        return super()._add_sub(other, op)
    def _iadd_sub(self, other, op):
        if self._persistent:
            # An in-place operator on an immutable object computes a new
            # one, as it does for an int or a tuple; the name the caller
            # used is then rebound to it by Python itself.
            return self._add_sub(other, self._INPLACE_OPS.get(op, op))
        if self._units is None or (
                isinstance(other, pint.Quantity) and other._units is None):
            return self._inplace_result(self._binop_none(other, op))
        ops = self._real_tensor_operands(other)
        if ops is not None:
            (x, y, promoted) = ops
            if promoted:
                op = self._INPLACE_OPS.get(op, op)
                return self._inplace_result(pint.Quantity._add_sub(x, y, op))
            return super()._iadd_sub(y, op)
        return super()._iadd_sub(other, op)
    def _mul_div(self, other, magnitude_op, units_op=None):
        if self._units is None or (
                isinstance(other, pint.Quantity) and other._units is None):
            return self._binop_none(other, magnitude_op)
        ops = self._real_tensor_operands(other)
        if ops is not None and ops[2]:
            (x, y, _) = ops
            return pint.Quantity._mul_div(x, y, magnitude_op, units_op)
        return super()._mul_div(other, magnitude_op, units_op)
    def _imul_div(self, other, magnitude_op, units_op=None):
        if self._persistent:
            return self._mul_div(
                other, self._INPLACE_OPS.get(magnitude_op, magnitude_op),
                self._INPLACE_OPS.get(units_op, units_op))
        if self._units is None or (
                isinstance(other, pint.Quantity) and other._units is None):
            return self._inplace_result(self._binop_none(other, magnitude_op))
        ops = self._real_tensor_operands(other)
        if ops is not None and ops[2]:
            (x, y, _) = ops
            mop = self._INPLACE_OPS.get(magnitude_op, magnitude_op)
            uop = self._INPLACE_OPS.get(units_op, units_op)
            return self._inplace_result(
                pint.Quantity._mul_div(x, y, mop, uop))
        return super()._imul_div(other, magnitude_op, units_op)
    def __pow__(self, other):
        if self._units is None or (
                isinstance(other, pint.Quantity) and other._units is None):
            return self._binop_none(other, operator.pow)
        return super().__pow__(other)
    def __ipow__(self, other):
        if self._persistent:
            return self.__pow__(other)
        if self._is_0d():
            return self._inplace_0d(operator.pow, other)
        if self._units is None or (
                isinstance(other, pint.Quantity) and other._units is None):
            result = self._binop_none(other, operator.pow)
            self._magnitude = result._magnitude
            self._units = result._units
            return self
        return super().__ipow__(other)
    def __rpow__(self, other):
        if self._units is None:
            (x, y) = self._none_operands(other)
            result = y ** x
            if isinstance(result, pint.Quantity):
                return result
            return self.__class__(result, None)
        return super().__rpow__(other)
    # Comparisons ------------------------------------------------------------
    # As with arithmetic above, a None-unit operand on either side bypasses
    # pint's own unit-aware comparison machinery (which assumes real units
    # or true dimensionlessness--neither of which describes a None-unit
    # quantity--and would otherwise raise or crash) and instead compares
    # magnitude(s) directly, deferring to the magnitude's own comparison
    # (and thus to the other operand's __eq__/__lt__/etc. via Python's
    # normal reflection when the other side is itself a real-unit
    # quantity--e.g. a None-unit 5 compared to 5 meters is simply unequal
    # rather than an error).
    def _binop_none_bool(self, other, op):
        (x, y) = self._none_operands(other)
        if isinstance(x, pint.Quantity) or isinstance(y, pint.Quantity):
            # Exactly one side has real units; compare exactly as the bare
            # value would compare with that quantity.
            if isinstance(x, pint.Quantity):
                (q, v, qop) = (x, y, op)
            else:
                (q, v, qop) = (y, x, _REFLECTED_COMPARISONS[op])
            if qop is operator.eq:
                return q.__eq__(v)
            elif qop is operator.ne:
                return q.__ne__(v)
            try:
                return q.compare(v, qop)
            except ValueError as e:
                # Pint raises a ValueError naming its own class here ("Cannot
                # compare PlainQuantity and <class 'numpy.ndarray'>"). The
                # same comparison written the other way round raises a
                # DimensionalityError, in Pint and here, as does
                # immlib.math.maximum for the same units; this one is raised
                # that way too rather than by operand order.
                raise _compare_units_error(q.units) from e
        return op(x, y)
    # Pint's own `__eq__`/`__ne__` produce an elementwise result for a NumPy
    # magnitude, but for a PyTorch magnitude they misbehave: Pint's
    # `is_duck_array_type` does not recognize tensors, so its "compare to
    # zero" shortcut leaves a multi-element boolean tensor un-reduced and then
    # uses it in a Python `and` (raising "Boolean value of Tensor with more
    # than one value is ambiguous"), and its "incompatible units" result is a
    # scalar `False` rather than an all-false tensor. `_tensor_eq` reproduces
    # Pint's NumPy semantics for tensor magnitudes: a comparison with a bare
    # value treats the value as dimensionless (all-false unless this quantity
    # is dimensionless, except that an all-zero/NaN value is compared by
    # magnitude, as Pint does); a comparison between quantities converts
    # compatible units and is all-false for incompatible units.
    def _tensor_eq(self, other, ne):
        other_is_q = isinstance(other, pint.Quantity)
        xm = self._magnitude
        ym = other._magnitude if other_is_q else other
        (xm, ym) = _promote_mags(xm, ym)
        def full(value):
            shape = torch.broadcast_shapes(tuple(xm.shape), tuple(ym.shape))
            return torch.full(shape, value != ne, dtype=torch.bool,
                              device=xm.device)
        op = operator.ne if ne else operator.eq
        if not other_is_q:
            if bool(((ym == 0) | torch.isnan(ym)).all()):
                if not self._is_multiplicative:
                    # Pint compares a non-multiplicative quantity (degC,
                    # say) against zero by converting it to base units,
                    # but only when the registry is set to make that
                    # conversion; otherwise the comparison is ambiguous
                    # and Pint raises. Rule 1 requires the tensor path to
                    # do the same as the NumPy path in both settings.
                    if not self._REGISTRY.autoconvert_offset_to_baseunit:
                        raise pint.OffsetUnitCalculusError(self._units)
                    xm = self.to_base_units()._magnitude
                return op(xm, ym)
            if self.dimensionless:
                q = self.__class__(xm, self._units)
                return op(
                    q._convert_magnitude_not_inplace(self.UnitsContainer()),
                    ym)
            return full(False)
        if self._units == other._units:
            return op(xm, ym)
        if not alike_units(self.units, other.units, ureg=self._REGISTRY):
            return full(False)
        q = self.__class__(xm, self._units)
        return op(q._convert_magnitude_not_inplace(other._units), ym)
    def _is_tensor_op(self, other):
        if torch.is_tensor(self._magnitude):
            return True
        if isinstance(other, pint.Quantity):
            return torch.is_tensor(other._magnitude)
        return torch.is_tensor(other)
    def __eq__(self, other):
        if self._units is None or (
                isinstance(other, pint.Quantity) and other._units is None):
            return self._binop_none_bool(other, operator.eq)
        if self._is_tensor_op(other):
            return self._tensor_eq(other, False)
        return super().__eq__(other)
    def __ne__(self, other):
        if self._units is None or (
                isinstance(other, pint.Quantity) and other._units is None):
            return self._binop_none_bool(other, operator.ne)
        if self._is_tensor_op(other):
            return self._tensor_eq(other, True)
        return super().__ne__(other)
    def __hash__(self):
        # Mirrors pint.Quantity.__hash__, except that a quantity with units of
        # None hashes like its magnitude (just as it compares equal to its
        # magnitude), and a 0-dimensional NumPy magnitude hashes like the
        # scalar it contains (so quant(5) hashes like 5). Magnitudes that are
        # themselves unhashable (arrays with dimensions) remain unhashable.
        if self._units is None:
            return hash(_hashable_mag(self._magnitude))
        base = self.to_base_units()
        mag = _hashable_mag(base._magnitude)
        if base.dimensionless:
            return hash(mag)
        return hash((pint.Quantity, mag, base._units))
    def __reduce__(self):
        # Pint pickles a quantity as a plain pint.Quantity attached to Pint's
        # application registry, which would lose both the immlib.Quantity type
        # and units of None. We instead reattach unpickled quantities to the
        # global immlib registry (unit registries themselves are not pickled,
        # just as in Pint). Persistence is carried across as well, so that
        # unpickling a persistent quantity gives a persistent quantity.
        return (
            _unpickle_quantity,
            (self._magnitude, self._units, self._persistent))
    def compare(self, other, op):
        """Returns the elementwise result of the ordered comparison `op`
        between this quantity and `other`.

        This is Pint's hook for ``<``, ``<=``, ``>`` and ``>=``, and is
        overridden here so that those operators obey immlib's rules for
        units of ``None`` and work with a PyTorch magnitude.

        Parameters
        ----------
        other : quantity or array or tensor or number
            The value to compare with.
        op : callable
            The comparison, as one of the ``operator`` module's functions.

        Returns
        -------
        array or tensor of bool
            The elementwise result, as a plain array or tensor.

        Raises
        ------
        pint.DimensionalityError
            If the two are not dimensionally comparable, which includes
            ordering a value with no units against one with real
            dimensions.
        """
        if self._units is None or(
                isinstance(other, pint.Quantity) and other._units is None):
            return self._binop_none_bool(other, op)
        ops = self._real_tensor_operands(other, compare=True)
        if ops is not None:
            (x, y, _) = ops
            return pint.Quantity.compare(x, y, op)
        return super().compare(other, op)
    def __matmul__(self, other):
        other_mag = (
            other._magnitude if isinstance(other, pint.Quantity) else other)
        if (torch.is_tensor(self._magnitude) or torch.is_tensor(other_mag)
                or scipy__is_sparse(self._magnitude)
                or scipy__is_sparse(other_mag)):
            return _immlib_matmul(self, other)
        return super().__matmul__(other)
    def __rmatmul__(self, other):
        # Computes other @ self; note the operand order relative to
        # __matmul__ above.
        other_mag = (
            other._magnitude if isinstance(other, pint.Quantity) else other)
        if (torch.is_tensor(self._magnitude) or torch.is_tensor(other_mag)
                or scipy__is_sparse(self._magnitude)
                or scipy__is_sparse(other_mag)):
            return _immlib_matmul(other, self)
        return super().__rmatmul__(other)
    # NumPy dispatch ----------------------------------------------------------
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Extends Pint's own NumPy ufunc dispatch to support units of None.

        If every quantity among the arguments has units of ``None``, the
        ufunc (with any method, e.g. ``np.add.reduce``, and any keywords,
        e.g. ``out``) is applied directly to the magnitudes (see
        ``_none_ufunc``). Otherwise, the curated ``immlib.math``-backed subset
        described by ``_numpy_math_dispatch`` is used when it applies, and
        Pint's own behavior is used for everything else, with any unit-less
        quantities passed to Pint as bare values (see
        ``_pint_numpy_dispatch``).

        Operator-mirroring and comparison ufuncs (see
        _NUMPY_OPERATOR_METHODS/_NUMPY_UNARY_OPERATOR_METHODS) are
        handled first, and separately from ``_numpy_math_dispatch``, by
        calling the Quantity operand's own dunder method directly--see
        the comment above _NUMPY_MATH_DELEGATE_FUNCS for why.
        """
        _check_out_persistent(kwargs)
        result = _none_ufunc(ufunc, method, inputs, kwargs)
        if result is not NotImplemented:
            return result
        if method == '__call__' and not kwargs:
            name = ufunc.__name__
            if name in _NUMPY_OPERATOR_METHODS and len(inputs) == 2:
                (a, b) = inputs
                (fwd, rev) = _NUMPY_OPERATOR_METHODS[name]
                if isinstance(a, pint.Quantity):
                    result = getattr(type(a), fwd)(a, b)
                    if result is not NotImplemented:
                        return result
                if isinstance(b, pint.Quantity):
                    return getattr(type(b), rev)(b, a)
                return NotImplemented
            if name in _NUMPY_UNARY_OPERATOR_METHODS and len(inputs) == 1:
                (a,) = inputs
                if isinstance(a, pint.Quantity):
                    return getattr(
                        type(a), _NUMPY_UNARY_OPERATOR_METHODS[name])(a)
        if method == '__call__':
            result = _numpy_math_dispatch(ufunc.__name__, inputs, kwargs)
            if result is not NotImplemented:
                return result
        return _pint_numpy_dispatch(
            lambda a, k: super(Quantity, self).__array_ufunc__(
                ufunc, method, *a, **k),
            inputs, kwargs)
    def __array_function__(self, func, types, args, kwargs):
        """Extends Pint's own NumPy function dispatch (``np.concatenate``,
        ``np.sum``, etc.) to support units of None: the curated
        ``immlib.math``-backed subset is used when it applies, and Pint's own
        implementation otherwise (see ``_pint_numpy_dispatch`` for how
        unit-less quantities are passed to Pint).
        """
        _check_out_persistent(kwargs)
        result = _numpy_math_dispatch(func.__name__, args, kwargs)
        if result is not NotImplemented:
            return result
        return _pint_numpy_dispatch(
            lambda a, k: super(Quantity, self).__array_function__(
                func, types, a, k),
            args, kwargs)
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Implements a deliberately minimal subset of PyTorch's
        ``__torch_function__`` dispatch protocol (see
        https://docs.pytorch.org/docs/stable/notes/extending.html), so that
        calling a handful of common ``torch.*`` functions directly on a
        ``Quantity`` (rather than via the Python operators, which already
        work without this) does the natural, unit-aware thing instead of
        failing outright. Anything not covered here returns
        ``NotImplemented``, in which case PyTorch raises its own error;
        broader coverage is left to ``immlib.math`` and to later
        refinement, per the design spec's phased implementation plan.
        """
        kwargs = kwargs or {}
        name = func.__name__
        if name in _TORCH_OPERATOR_METHODS:
            if kwargs or len(args) != 2:
                return NotImplemented
            (a, b) = args
            (fwd, rev) = _TORCH_OPERATOR_METHODS[name]
            if isinstance(a, pint.Quantity):
                result = getattr(type(a), fwd)(a, b)
                if result is not NotImplemented:
                    return result
            if isinstance(b, pint.Quantity):
                return getattr(type(b), rev)(b, a)
            return NotImplemented
        elif name in _TORCH_UNARY_OPERATOR_METHODS:
            if kwargs or len(args) != 1:
                return NotImplemented
            (a,) = args
            if not isinstance(a, pint.Quantity):
                return NotImplemented
            return getattr(type(a), _TORCH_UNARY_OPERATOR_METHODS[name])(a)
        elif name in _TORCH_UNIT_PRESERVING_FUNCS:
            if not args or not isinstance(args[0], pint.Quantity):
                return NotImplemented
            (self, rest) = (args[0], args[1:])
            mag = func(self._magnitude, *rest, **kwargs)
            return self.__class__(mag, self._units)
        elif name in _TORCH_CONCAT_FUNCS:
            return cls._torch_concat(func, args, kwargs)
        elif name in _TORCH_MATH_DELEGATE_FUNCS:
            nargs = 2 if name == 'atan2' else 1
            if kwargs or len(args) != nargs:
                return NotImplemented
            if not any(isinstance(a, pint.Quantity) for a in args):
                return NotImplemented
            from ..math import _core as _immath
            fn = getattr(_immath, _TORCH_MATH_DELEGATE_FUNCS[name])
            return fn(*args)
        elif name in _TORCH_MATH_REDUCTION_FUNCS:
            if not args or not isinstance(args[0], pint.Quantity):
                return NotImplemented
            (self, rest) = (args[0], args[1:])
            from ..math import _core as _immath
            fn = getattr(_immath, _TORCH_MATH_REDUCTION_FUNCS[name])
            # immlib.math uses PyTorch's own argument names and defaults, so
            # the call is passed through as it was written.
            return fn(self, *rest, **kwargs)
        elif name in _TORCH_BOOL_REDUCTION_FUNCS:
            if not args or not isinstance(args[0], pint.Quantity):
                return NotImplemented
            (self, rest) = (args[0], args[1:])
            return func(self._magnitude, *rest, **kwargs)
        return NotImplemented
    @classmethod
    def _torch_concat(cls, func, args, kwargs):
        """Implements ``torch.cat``/``torch.stack``-style dispatch; units are
        reconciled exactly as by ``immlib.math.concatenate`` (see
        ``__torch_function__``).
        """
        if not args:
            return NotImplemented
        (seq, rest) = (args[0], args[1:])
        quants = [a for a in seq if isinstance(a, pint.Quantity)]
        if not quants:
            return NotImplemented
        from ..math._core import _reconcile_seq
        (mags, u) = _reconcile_seq(seq, func.__name__)
        mags = promote(*mags)
        result_mag = func(mags, *rest, **kwargs)
        return quants[0]._REGISTRY.Quantity(result_mag, u)


class UnitRegistry(pint.UnitRegistry):
    """The ``immlib`` extension of ``pint.UnitRegistry``.

    ``immlib.UnitRegistry`` is a ``pint.UnitRegistry`` subclass whose
    ``Quantity`` type is ``immlib.Quantity``: any quantity produced by an
    ``immlib.UnitRegistry``--via its own ``Quantity(...)`` constructor, unit
    multiplication (``5 * ureg.meter``), string/expression parsing, quantity
    arithmetic, or NumPy's array-function dispatch (e.g. ``np.concatenate``
    on a list of such quantities)--is automatically an ``immlib.Quantity``.
    This works because every one of those pint code paths ultimately builds
    new quantities via ``self.Quantity`` (either on the registry itself or
    via a quantity's own ``self.__class__``/``self._REGISTRY.Quantity``), so
    overriding the ``Quantity`` class attribute here, in the ordinary,
    supported way that pint itself uses to assemble ``UnitRegistry`` out of
    its own facet mixins, is sufficient--no monkey-patching of any
    ``pint.UnitRegistry`` instance is needed or performed.

    A plain ``pint.UnitRegistry`` is never touched by immlib: only an actual
    ``immlib.UnitRegistry`` instance (such as ``immlib.units``, the default
    global registry) supports ``immlib.Quantity`` features, including a
    ``units``/``u`` of ``None``. Passing a plain ``pint.UnitRegistry`` as the
    ``ureg`` option of a function like ``immlib.quant`` continues to yield
    ordinary ``pint.Quantity`` objects from that registry, and requesting a
    unit-less (``unit=None``) quantity from one raises an error, since a
    plain ``pint.UnitRegistry`` has no way to represent "no units".

    .. Note:: An ``immlib.UnitRegistry``'s ``Quantity`` type interprets only
        an explicit ``units=None`` as "no units". Pint's own internal code
        paths that construct a quantity without passing units at all--such
        as parsing the literal word ``"dimensionless"`` or an empty
        expression string--still produce Pint's real ``dimensionless``
        unit.
    """
    Quantity = type('ImmlibQuantity', (Quantity, pint.UnitRegistry.Quantity), {})


_initial_global_ureg = UnitRegistry()
from ._core import _global_ureg
_global_ureg[0] = _initial_global_ureg
# We want to disable the awful pint warning for numpy if it's present:
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _initial_global_ureg.Quantity([])
# Make sure there's a pixel unit
if not hasattr(_initial_global_ureg, 'pixels'):
    _initial_global_ureg.define('pixel = [image_length] = px')
def like_unit(obj, /, *, ureg=Ellipsis):
    """Returns ``True`` if `obj` is or names a ``pint.Unit`` and ``False``
    otherwise.

    ``like_unit(obj)`` returns ``True`` if `obj` is a ``pint.Unit`` object or a
    string that names a ``pint.Unit`` and ``False`` otherwise.

    Parameters
    ----------
    obj : object
        The object whose quality as a ``pint.Unit`` is to be assessed.

    ureg : pint.UnitRegistry, Ellipsis, or None, optional
        The ``pint.UnitRegistry`` object to use. If ``None``, then any registry
        is allowed but an exception is raised if `obj` is a string because
        there is no registry in which to look it up. If ``Ellipsis`` (the
        default), then the ``immlib.units`` registry is required.

    Returns
    -------
    bool
        ``True`` if `obj` is a ``pint.Unit`` or a string naming such a unit and
        ``False`` otherwise.
    """
    if isinstance(obj, pint.Unit):
        return True
    if ureg is Ellipsis:
        ureg = _default_ureg()
    if is_str(obj):
        if ureg is None:
            raise ValueError(
                "cannot determine if str is unit-like when ureg is None")
        return hasattr(ureg, obj) and isinstance(getattr(ureg, obj), pint.Unit)
    else:
        return False
def unit(obj, /, ureg=None):
    """Converts the argument into a a ``pint.Unit`` object.

    ``unit(obj)`` returns the ``immlib``-library unit object for the given unit
    object `obj` (which may be from a separate ``pint.UnitRegistry`` instance).

    ``unit(unitname)`` returns the unit object for the given unit name string
    ``unitname``.

    ``unit(q)`` returns the unit of the given quantity object ``q``.

    .. Note:: ``immlib`` considers an object to be "unit-like" if ``unit(obj)``
        returns a valid ``pint.Unit`` object.
    
    Parameters
    ----------
    obj : object
        The object that is to be converted to a unit.
    ureg : ping.UnitRegistry, None, or Ellipsis, optional
        The unit registry to convert the object into. If ``Ellipsis``, then
        ``immlib.units`` is used. If ``None`` (the default), then the unit
        registry for `obj` is used if `obj` is a quantity or unit already, and
        ``immlib.units`` is used if not. Otherwise, must be a unit registry.

    Returns
    -------
    pint.Unit
        The ``Unit`` object associated with the given argument.

    Raises
    ------
    TypeError
        When the argument cannot be converted to a ``pint.Unit`` object.
    """
    if obj is None:
        raise ValueError("cannot create a unit for None; use 'dimensionless'")
    if ureg is Ellipsis:
        ureg = _default_ureg()
    if is_quant(obj):
        obj = obj.u
    if is_unit(obj):
        if ureg is None or ureg is unitregistry(obj):
            return obj
        else:
            return getattr(ureg, str(obj))
    elif is_str(obj):
        if ureg is None:
            ureg = _default_ureg()
        return getattr(ureg, obj)
    else:
        raise ValueError(f'unrecognized unit argument: {obj}')
_unitlike_types = (str, pint.Unit, pint.Quantity)
def alike_units(a, b, /, *, ureg=None):
    """Returns ``True`` if the arguments are alike units, otherwise ``False``.

    ``alike_units(a, b)`` returns ``True`` if `a` and `b` can be cast to
    each other in terms of units and ``False`` otherwise. Both `a` and `b`
    can either be units, unit names, or quantities with units. If either `a`
    or `b` is neither a unit nor a quantity, or is a quantity whose units are
    ``None``, then it is treated as a bare number, which (as in Pint's own
    arithmetic) is alike only to dimensionless units.

    Parameters
    ----------
    a : unit-like
        A unit object or the name of a unit or a quantity.
    b : unit-like
        A unit object or the name of a unit or a quantity.
    ureg : pint.UnitRegistry, None, Ellipsis, optional
        The ``pint.UnitRegistry`` object to use. If ``Ellipsis``, then the
        ``immlib.units`` registry is used. If ``None``, then the registry of
        object `a` is used if available or that of object `b` if not. If
        neither `a` nor `b` has an available registry, then ``immlib.units`` is
        used.

    Returns
    -------
    bool
        ``True`` if the units `a` and `b` are alike and ``False`` otherwise.
    """
    if ureg is Ellipsis:
        ureg = _default_ureg()
    if ureg is None:
        ureg = unitregistry(a, None)
        if ureg is None:
            ureg = unitregistry(b, Ellipsis)
    # A non-unit-like object, or a quantity whose units are None, is treated
    # like a bare number (i.e., as dimensionless), as in Pint's arithmetic.
    if not isinstance(a, _unitlike_types) or (
            isinstance(a, pint.Quantity) and a._units is None):
        a = 'dimensionless'
    if not isinstance(b, _unitlike_types) or (
            isinstance(b, pint.Quantity) and b._units is None):
        b = 'dimensionless'
    return ureg.is_compatible_with(a, b)
def _quant_magnitude(mag):
    """Returns the magnitude that ``quant()`` stores for the non-quantity
    `mag`, or raises ``TypeError`` if `mag` is not numerical.

    PyTorch tensors, NumPy arrays, and SciPy sparse arrays/matrices are kept
    as they are (so ``quant()`` never changes an existing array's or tensor's
    backend or device). Anything else (Python and NumPy scalars, lists,
    tuples, and other array-like values) is converted with
    ``numpy.asarray``, so scalars become 0-dimensional arrays. Arrays must
    have a numeric (or boolean) dtype.
    """
    if torch.is_tensor(mag):
        return mag
    if isinstance(mag, (str, bytes)):
        raise TypeError(
            f"quant: magnitude must be numerical, not {type(mag).__name__}"
            " (to parse a quantity from a string, use the unit registry,"
            " e.g. immlib.units.Quantity('5 mm'))")
    if isinstance(mag, np.ndarray) or scipy__is_sparse(mag):
        arr = mag
    else:
        try:
            arr = np.asarray(mag)
        except Exception as e:
            raise TypeError(
                f"quant: magnitude of type {type(mag).__name__} could not be"
                " converted into a numerical array") from e
    dt = arr.dtype
    if not (np.issubdtype(dt, np.number) or np.issubdtype(dt, np.bool_)):
        raise TypeError(
            f"quant: magnitude must be numerical, but it has dtype {dt}"
            f" (type {type(mag).__name__})")
    return arr
def quant(mag, /, unit=Ellipsis, *, ureg=None):
    """Returns a ``pint.Quantity`` object with the given magnitude and unit.

    ``quant(mag, unit)`` returns a ``pint.Quantity`` object with the given
    magnitude `mag` and `unit`. If `mag` is alreaady a ``pint.Quantity``, then
    it is converted into the given units and returned (a copy of `mag` is made
    only if necessary); if the units of `mag` in this case are not compatible
    with `unit`, then an error is raised. If `mag` is not a quantity, then the
    given `unit` is used to create the quantity.

    If `mag` is not a quantity, it must be numerical. NumPy arrays, PyTorch
    tensors, and SciPy sparse arrays/matrices are used as the magnitude
    as-is (``quant()`` never converts an array into a tensor or vice versa,
    and never moves a tensor between devices; see ``immlib.to_array`` and
    ``immlib.to_tensor`` for that). Other numerical values, including Python
    and NumPy scalars and lists of numbers, are converted into NumPy arrays;
    scalars become 0-dimensional arrays.

    ``quant(mag)`` is equivalent to ``quant(mag, Ellipsis)``. If `mag` is
    already a ``pint.Quantity`` object (of any kind, ``immlib.Quantity`` or
    not), it is returned unchanged, in its own unit registry: because `ureg`
    defaults to ``None``, which requests no particular registry, a plain
    ``pint.Quantity`` is not promoted to an ``immlib.Quantity``. Naming a
    registry asks for one, so ``quant(mag, ureg=Ellipsis)`` returns an
    ``immlib.Quantity`` in ``immlib.units`` whatever kind of quantity `mag`
    is; ``immlib.ilquant`` is this function with that default. If `mag` is
    not a quantity, a unit-less (``units=None``) quantity is created by
    default (see the ``unit=None`` note below).

    .. Note:: ``unit=None`` is not equivalent to ``unit='dimensionless'``.
        ``immlib`` uses ``None`` to represent having no units at all--as
        opposed to Pint's own ``dimensionless``, a real, measurable unit--and
        ``quant(mag, None)`` returns a quantity whose ``units`` is ``None``
        and whose magnitude is ``mag``'s bare magnitude, unconverted (see
        ``immlib.Quantity``); likewise, a non-quantity `mag` with no `unit`
        given (``quant(mag)``) becomes a unit-less quantity. This is only
        possible when the relevant unit registry (`mag`'s own registry, or
        `ureg`, if given) is an ``immlib.UnitRegistry``: unit-less quantities
        cannot belong to a plain ``pint.UnitRegistry``, so requesting one
        (explicitly via ``unit=None``, or implicitly via ``quant(mag)`` on a
        non-quantity `mag`) raises a ``ValueError`` when the relevant
        registry is a plain ``pint.UnitRegistry``.

    Parameters
    ----------
    mag : object
        The magnitude to be given a unit.
    unit : unit-like, None, Ellipsis, optional
        The units to use in the returned quantity. If ``Ellipsis`` is given
        (the default), then `mag`'s own units are used if `mag` is already a
        quantity, and ``None`` (immlib's "no units") is used otherwise. If
        ``None`` is given explicitly, the returned quantity has no units,
        whatever units `mag` may have had; this requires an
        ``immlib.UnitRegistry`` (see above).
    ureg : pint.UnitRegistry, None, Ellipsis, optional
        The ``pint.UnitRegistry`` object to use for units. If `ureg` is
        ``Ellipsis``, then ``immlib.units`` is used. If `ureg` is ``None``
        (the default), then `mag`'s own unit registry is used when `mag` is
        already a quantity, and ``immlib.units`` is used otherwise.

    Returns
    -------
    pint.Quantity
        A quantity object representing the given magnitude and unit.

    Raises
    ------
    TypeError
        If `mag` is not a quantity and is not numerical: strings, and arrays
        whose dtype is neither numeric nor boolean (such as object arrays),
        are rejected.
    ValueError
        If a unit-less (``unit=None``) quantity is requested using a unit
        registry that is not an ``immlib.UnitRegistry``.

    """
    if ureg is Ellipsis:
        ureg = _default_ureg()
    if is_quant(mag):
        if ureg is None:
            ureg = unitregistry(mag)
        qcls = ureg.Quantity
        if unit is None:
            if not issubclass(qcls, Quantity):
                raise ValueError(
                    "quant: unit=None requires an immlib.UnitRegistry; the"
                    " given (or mag's own) unit registry is a plain"
                    " pint.UnitRegistry, which cannot represent a"
                    " unit-less quantity")
            # Strip mag down to its bare magnitude and attach no units,
            # regardless of what units mag previously had--there is no
            # meaningful "conversion" from a real unit (or from no unit) to
            # having no units at all, so we never invoke mag's own to().
            return qcls(_quant_magnitude(mag.magnitude), None)
        elif unit is Ellipsis:
            # mag keeps its own units. If the caller named a registry, the
            # re-homing at the end of this function moves mag into it (and
            # so promotes a plain pint.Quantity into an immlib.Quantity);
            # if the caller named none, ureg is mag's own registry, above,
            # and mag is returned exactly as given.
            q = mag
        else:
            # mag.to() stays within mag's own unit registry; if the caller
            # also passed an explicit, different `ureg`, that is handled
            # below.
            q = mag.to(unit)
    else:
        if ureg is None:
            ureg = _default_ureg()
        qcls = ureg.Quantity
        if unit is Ellipsis:
            # A non-quantity defaults to immlib's "no units" (None), not to
            # pint's dimensionless; see immlib.Quantity.
            unit = None
        if unit is None and not issubclass(qcls, Quantity):
            raise ValueError(
                "quant: unit=None requires an immlib.UnitRegistry; the given"
                " (or default) unit registry is a plain pint.UnitRegistry,"
                " which cannot represent a unit-less quantity")
        q = qcls(_quant_magnitude(mag), unit)
    q_ureg = unitregistry(q)
    if q_ureg is not ureg:
        # The caller explicitly requested a different registry than the one
        # q ended up in (e.g., mag belonged to a different registry than an
        # explicitly-given ureg); re-home q into the requested registry.
        if isinstance(q, Quantity) and q.units is None:
            # A unit-less quantity cannot be represented outside of an
            # immlib.UnitRegistry; moving it into a plain pint registry
            # would silently turn "no units" into dimensionless.
            if not issubclass(ureg.Quantity, Quantity):
                raise ValueError(
                    "quant: a unit-less quantity cannot be moved into a"
                    " plain pint.UnitRegistry, which cannot represent one")
            return ureg.Quantity(q._magnitude, None)
        return ureg.Quantity(q._magnitude, q._units)
    else:
        return q
@docwrap(format='numpy', inheritparams=quant, inheritraises=quant)
def ilquant(mag, /, unit=Ellipsis, *, ureg=Ellipsis):
    """Returns an ``immlib.Quantity`` with the given magnitude and unit.

    ``ilquant`` is ``immlib.quant`` with one difference: `ureg` defaults to
    ``Ellipsis`` instead of to ``None``, so a unit registry is always
    requested and the result is always an ``immlib.Quantity``.

    The difference shows only for a `mag` that is already a quantity.
    ``quant(q)`` requests no particular registry, so it returns `q`
    untouched, including a plain ``pint.Quantity``; ``ilquant(q)`` requests
    ``immlib.units``, so it returns an ``immlib.Quantity`` holding `q`'s
    magnitude and units. For every other argument, and for a `mag` that is
    not a quantity, the two functions are the same.

    Parameters
    ----------
    ureg : pint.UnitRegistry, None, Ellipsis, optional
        The ``pint.UnitRegistry`` object to use for units. If `ureg` is
        ``Ellipsis`` (the default), then ``immlib.units`` is used. If `ureg`
        is ``None``, then `mag`'s own unit registry is used when `mag` is
        already a quantity, and ``immlib.units`` is used otherwise--i.e.,
        ``ilquant(mag, ureg=None)`` is ``quant(mag)``.

    Returns
    -------
    immlib.Quantity
        A quantity object representing the given magnitude and unit, in the
        requested unit registry.

    See Also
    --------
    quant
    """
    return quant(mag, unit, ureg=ureg)
def mag(obj, /, unit=Ellipsis, *, strict=False):
    """Returns the magnitude of the given object.

    ``mag(quantity)`` returns the magnitude of the given quantity, regardless
    of the quantity's unit.

    ``mag(obj)``, for a non-quantity object `obj`, simply returns `obj`.

    ``mag(arg, unit)`` returns ``arg.m_as(unit)`` if ``arg`` is a quantity and
    returns ``arg`` itself if ``arg`` is not a quantity.

    ``mag(arg, Ellipsis)`` is equivalent to ``mag(arg)``.

    ``mag(obj, None)`` returns `obj` if it is not a ``pint.Quantity`` and
    raises an exception if `obj` is a ``pint.Quantity``.

    If ``mag(quantity, unit)`` is given a quantity not compatible with the
    given unit, then an error is raised.

    Note that if the first argument to ``mag()`` is not a quantity, then the
    `unit` argument is always ignored, and the first argument is returned
    as-is. This behavior can be changed using the `strict` option.

    Parameters
    ----------
    obj : object
        The object that is to be converted into a magnitude.
    unit : unit-like, None, or Ellipsis, optional
        The unit in which the magnitude of the argument `obj` should be
        returned. The default argument of ``Ellipsis`` indicates that the
        value's native unit, if any, should be used. A value of ``None``
        indicates that `obj` must have no units at all--either because it is
        not a quantity, or because it is an ``immlib.Quantity`` whose own
        ``units`` is ``None`` (see ``immlib.Quantity``)--otherwise an
        exception is raised; note that this is a stricter check than
        ``quant(obj, None)``, which always succeeds by stripping whatever
        units `obj` has.
    strict : bool, optional
        Whether strict matching of the unit is performed. If ``False`` (the
        default), then a non-quantity (such as a plain NumPy array) is treated
        as a quantity whose unit is the type passed in the `unit` parameter; if
        ``True``, then `obj` must be compatible with the `unit` parameter or an
        error is raised.

    Returns
    -------
    object
        The magnitude of `obj` in the requested unit, if `obj` is a quantity,
        or `obj` itself, if it is not a quantity.

    Raises
    ------
    DimensionalityError
        If the given `obj` is a quantity whose unit is not compatible with the
        `unit` parameter.
    ValueError
        If `unit` is None but `obj` is a quantity or if a unit is requested of
        a non-quantity with the `strict` option enabled.
    """
    if is_quant(obj):
        if unit is None:
            if obj.units is None:
                return obj.m
            raise ValueError(
                "unit=None requested of a quantity with real units; to"
                " strip a quantity's units regardless of what they are, use"
                " quant(obj, None) instead")
        elif unit is Ellipsis:
            return obj.m
        else:
            return obj.m_as(unit)
    elif strict is True:
        if unit is not None:
            raise ValueError(
                f"unit '{unit}' does not strictly match non-quantity")
    return obj


# Promotion ###################################################################

def _array_promote(*args, ureg=None):
    return [to_array(el, ureg=ureg) for el in args]
@alttorch(_array_promote)
def promote(*args, ureg=None):
    """Promotes all arguments into quantities with compatible magnitudes.

    ``promote(a, b, c...)`` converts all of the passed arguments into numerical
    quantity objects and returns them as a list. The returned arguments will
    all have compatible (promoted) types or magnitude types.

    Promotion is determined based on the object type. If any of the objects are
    PyTorch tensors or quantities with tensor magnitudes, then all of the
    returned quantities will have for their magnitudes PyTorch tensors with the
    same profile (e.g, device) as the tensor argument(s). Otherwise, the
    returned quantities will be converted into array types. The purpose of this
    promotion is to ensure that all arguments can be combined into an
    expression (PyTorch tensor operations generally require that all arguments
    are tensors).

    Parameters
    ----------
    args
        The arguments that are to be promoted.
    ureg : pint.UnitRegistry or None, optional
        The ``pint.UnitRegistry`` object to use for units. If `ureg` is
        ``Ellipsis``, then ``immlib.units`` is used. If `ureg` is ``None`` (the
        default), then no specific coersion to a ``pint.UnitRegistry`` is
        performed.

    Returns
    -------
    list of tensors or arrays
        A list of the arguments after each has been promoted.
    """
    if ureg is None:
        ureg = _default_ureg()
    # We can start by making sure that the quants in the args use ureg
    if ureg is not None:
        args = [
            (quant(arg, ureg=ureg) if is_quant(arg, ureg=ureg) else
             arg                   if torch.is_tensor(arg)     else
             arg                   if scipy__is_sparse(arg)    else
             np.asarray(arg))
            for arg in args]
    # Basic question: are any of them tensors?
    first_tensor = next(
        (a for a in args
         if (is_quant(a) and torch.is_tensor(a.m)) or torch.is_tensor(a)),
        None)
    # If there aren't any tensors then args is fine as-is.
    if first_tensor is None:
        return args
    device = first_tensor.device
    # Otherwise, we need to turn them all into tensors like this one.
    for (ii,q) in enumerate(args):
        if q is first_tensor:
            continue
        if is_quant(q):
            mag = q.m
            mag = to_tensor(mag, device=device)
            if mag is not q.m:
                args[ii] = q.__class__(mag, q.u)
        else:
            mag = to_tensor(q, device=device)
            if mag is not q:
                args[ii] = mag
    # That's all that is needed.
    return args
