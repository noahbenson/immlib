# -*- coding: utf-8 -*-
###############################################################################
# immlib/util/_quantity.py


# Dependencies ################################################################

import inspect
import operator
import warnings

import pint
import numpy as np
import scipy.sparse as sps

from ..doc import docwrap
from ._core import (is_set, is_str, unitregistry)
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
@docwrap('immlib.is_ureg')
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
@docwrap('immlib.is_unit')
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
        from immlib import units
        return isinstance(q, units.Unit)
    elif is_ureg(ureg):
        return isinstance(q, ureg.Unit)
    else:
        raise TypeError("parameter ureg must be a UnitRegistry")
@docwrap('immlib.is_quant')
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
    %(immlib.is_unit.raises)s

    """
    if ureg is None:
        if not isinstance(obj, pint.Quantity):
            return False
    else:
        if ureg is Ellipsis:
            from immlib import units
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
@docwrap('immlib.default_ureg')
class default_ureg:
    """Context manager for setting the default ``immlib`` unit registry.

    The following code-block can be used to evaluate the code represented by
    ``...`` using the unit-registry ``ureg`` as the default ``immlib.units``
    registry:

    .. code-block:: python
    
       with immlib.default_ureg(ureg):
           ...
    """
    def __init__(self, ureg):
        if not is_ureg(ureg):
            raise TypeError("ureg must be a pint.UnitRegistry")
        object.__setattr__(self, 'original', None)
        object.__setattr__(self, 'ureg', ureg)
    def __enter__(self):
        import immlib
        object.__setattr__(self, 'original', immlib.units)
        immlib.units = self.ureg
        return self.ureg
    def __exit__(self, exc_type, exc_val, exc_tb):
        import immlib
        immlib.units = self.original
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
    PyTorch tensor magnitude.

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
    else:
        # The caller guarantees that at least one side is a tensor, so
        # b_mag must be one here.
        a_mag = to_tensor(a_mag, device=b_mag.device, dtype=b_mag.dtype)
    result_mag = torch.matmul(a_mag, b_mag)
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
# (var/prod) or whose PyTorch-native keyword arguments ('dim', 'keepdim',
# 'correction') differ in name from immlib.math's own NumPy-style ones
# ('axis', 'keepdims', 'ddof')--std/var/prod all fall in the latter
# category (see _TORCH_KWARG_TRANSLATION below), so dispatching these
# through __torch_function__ must translate PyTorch's own kwargs before
# delegating to immlib.math, keeping torch.var(q, dim=0, correction=0)
# behaving like ordinary PyTorch (just unit-aware) rather than switching
# callers over to immlib.math's own argument names.
#
# torch.min/torch.max are deliberately NOT included here (or anywhere in
# __torch_function__): called with a `dim` argument, they return a
# PyTorch-native ``(values, indices)`` named tuple, a result shape that
# immlib.math.min/max (which, matching numpy.min/max, return only the
# values--see its docstring) cannot reproduce without silently changing
# torch.min/max's own documented contract for a Quantity input, which is
# exactly the kind of surprising, backend-specific behavior the design
# spec asks to avoid (section 9.3); they are left unhandled, so PyTorch
# raises its own error for a Quantity argument, rather than immlib
# guessing at a translation.
_TORCH_MATH_REDUCTION_FUNCS = {'std': 'std', 'var': 'var', 'prod': 'prod'}
_TORCH_KWARG_TRANSLATION = {
    'dim': 'axis', 'keepdim': 'keepdims', 'correction': 'ddof'}
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
    # Backend ------------------------------------------------------------
    @property
    def backend(self):
        """``numpy`` if this quantity's magnitude is a NumPy array or a
        SciPy sparse array/matrix, or ``torch`` if it is a PyTorch tensor.
        """
        return torch if torch.is_tensor(self._magnitude) else np
    # Units of None ------------------------------------------------------
    @property
    def units(self):
        if self._units is None:
            return None
        return self._REGISTRY.Unit(self._units)
    @property
    def u(self):
        return self.units
    @property
    def dimensionless(self):
        if self._units is None:
            return False
        return super().dimensionless
    @property
    def dimensionality(self):
        if self._units is None:
            raise TypeError(
                "quantity has no units (units is None); dimensionality is"
                " undefined for a unitless immlib quantity")
        return super().dimensionality
    def check(self, dimension):
        if self._units is None:
            return False
        return super().check(dimension)
    def to(self, other=None, *contexts, **ctx_kwargs):
        if self._units is None or other is None:
            if contexts or ctx_kwargs:
                raise ValueError(
                    "to: contexts are not supported when converting to or"
                    " from a unitless (units=None) quantity")
            return self.__class__(self._magnitude, other)
        return super().to(other, *contexts, **ctx_kwargs)
    def ito(self, other=None, *contexts, **ctx_kwargs):
        if self._units is None or other is None:
            new = self.to(other, *contexts, **ctx_kwargs)
            self._magnitude = new._magnitude
            self._units = new._units
            return None
        return super().ito(other, *contexts, **ctx_kwargs)
    def __str__(self):
        if self._units is None:
            return str(self._magnitude)
        return super().__str__()
    def __format__(self, spec):
        if self._units is None:
            return format(self._magnitude, spec)
        return super().__format__(spec)
    def m_as(self, units):
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
    def _binop_none(self, other, op):
        self_none = self._units is None
        other_is_q = isinstance(other, pint.Quantity)
        other_none = other_is_q and other._units is None
        self_val = self._magnitude if self_none else self
        other_val = other._magnitude if other_none else other
        result = op(self_val, other_val)
        if isinstance(result, pint.Quantity):
            if isinstance(result, type(self)):
                return result
            qcls = result._REGISTRY.Quantity
            return qcls(result._magnitude, result._units)
        return self.__class__(result, None)
    def _add_sub(self, other, op):
        if self._units is None or (
                isinstance(other, pint.Quantity) and other._units is None):
            return self._binop_none(other, op)
        return super()._add_sub(other, op)
    def _iadd_sub(self, other, op):
        if self._units is None or (
                isinstance(other, pint.Quantity) and other._units is None):
            result = self._binop_none(other, op)
            self._magnitude = result._magnitude
            self._units = result._units
            return self
        return super()._iadd_sub(other, op)
    def _mul_div(self, other, magnitude_op, units_op=None):
        if self._units is None or (
                isinstance(other, pint.Quantity) and other._units is None):
            return self._binop_none(other, magnitude_op)
        return super()._mul_div(other, magnitude_op, units_op)
    def _imul_div(self, other, magnitude_op, units_op=None):
        if self._units is None or (
                isinstance(other, pint.Quantity) and other._units is None):
            result = self._binop_none(other, magnitude_op)
            self._magnitude = result._magnitude
            self._units = result._units
            return self
        return super()._imul_div(other, magnitude_op, units_op)
    def __pow__(self, other):
        if self._units is None or (
                isinstance(other, pint.Quantity) and other._units is None):
            return self._binop_none(other, operator.pow)
        return super().__pow__(other)
    def __ipow__(self, other):
        if self._units is None or (
                isinstance(other, pint.Quantity) and other._units is None):
            result = self._binop_none(other, operator.pow)
            self._magnitude = result._magnitude
            self._units = result._units
            return self
        return super().__ipow__(other)
    def __rpow__(self, other):
        if self._units is None:
            return other ** self._magnitude
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
        self_none = self._units is None
        other_is_q = isinstance(other, pint.Quantity)
        other_none = other_is_q and other._units is None
        self_val = self._magnitude if self_none else self
        other_val = other._magnitude if other_none else other
        return op(self_val, other_val)
    def __eq__(self, other):
        if self._units is None or (
                isinstance(other, pint.Quantity) and other._units is None):
            return self._binop_none_bool(other, operator.eq)
        return super().__eq__(other)
    def __ne__(self, other):
        if self._units is None or (
                isinstance(other, pint.Quantity) and other._units is None):
            return self._binop_none_bool(other, operator.ne)
        return super().__ne__(other)
    def __hash__(self):
        return super().__hash__()
    def compare(self, other, op):
        if self._units is None or (
                isinstance(other, pint.Quantity) and other._units is None):
            return self._binop_none_bool(other, op)
        return super().compare(other, op)
    def __matmul__(self, other):
        other_mag = (
            other._magnitude if isinstance(other, pint.Quantity) else other)
        if torch.is_tensor(self._magnitude) or torch.is_tensor(other_mag):
            return _immlib_matmul(self, other)
        return super().__matmul__(other)
    def __rmatmul__(self, other):
        # Computes other @ self; note the operand order relative to
        # __matmul__ above.
        other_mag = (
            other._magnitude if isinstance(other, pint.Quantity) else other)
        if torch.is_tensor(self._magnitude) or torch.is_tensor(other_mag):
            return _immlib_matmul(other, self)
        return super().__rmatmul__(other)
    # NumPy dispatch ----------------------------------------------------------
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Extends Pint's own NumPy ufunc dispatch with the curated,
        None-units-aware ``immlib.math``-backed subset described by
        ``_numpy_math_dispatch``, falling back to Pint's ordinary
        (real-units-only) behavior--unchanged--for anything outside that
        subset, or for any ufunc method other than a plain call (e.g.
        ``np.add.reduce``, which is not handled here).

        Operator-mirroring and comparison ufuncs (see
        _NUMPY_OPERATOR_METHODS/_NUMPY_UNARY_OPERATOR_METHODS) are
        handled first, and separately from ``_numpy_math_dispatch``, by
        calling the Quantity operand's own dunder method directly--see
        the comment above _NUMPY_MATH_DELEGATE_FUNCS for why.
        """
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
        return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
    def __array_function__(self, func, types, args, kwargs):
        """Extends Pint's own NumPy function dispatch (``np.concatenate``,
        ``np.sum``, etc.) the same way as ``__array_ufunc__`` above.
        """
        result = _numpy_math_dispatch(func.__name__, args, kwargs)
        if result is not NotImplemented:
            return result
        return super().__array_function__(func, types, args, kwargs)
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
            if rest:
                # A positional dim (torch.var(q, 0), say) isn't handled;
                # use the dim= keyword instead.
                return NotImplemented
            translated = {
                _TORCH_KWARG_TRANSLATION.get(k, k): v
                for (k, v) in kwargs.items()}
            from ..math import _core as _immath
            fn = getattr(_immath, _TORCH_MATH_REDUCTION_FUNCS[name])
            return fn(self, **translated)
        elif name in _TORCH_BOOL_REDUCTION_FUNCS:
            if not args or not isinstance(args[0], pint.Quantity):
                return NotImplemented
            (self, rest) = (args[0], args[1:])
            return func(self._magnitude, *rest, **kwargs)
        return NotImplemented
    @classmethod
    def _torch_concat(cls, func, args, kwargs):
        """Implements ``torch.cat``/``torch.stack``-style dispatch: every
        quantity in the input sequence must share compatible units (a
        non-quantity element is only accepted when the result is
        unit-less); see ``__torch_function__``.
        """
        if not args:
            return NotImplemented
        (seq, rest) = (args[0], args[1:])
        quants = [a for a in seq if isinstance(a, pint.Quantity)]
        if not quants:
            return NotImplemented
        unit0 = quants[0]._units
        mags = []
        for a in seq:
            if isinstance(a, pint.Quantity):
                if a._units is None and unit0 is None:
                    mags.append(a._magnitude)
                elif a._units is None or unit0 is None:
                    return NotImplemented
                elif a._units == unit0:
                    mags.append(a._magnitude)
                else:
                    mags.append(a.m_as(unit0))
            elif unit0 is None:
                mags.append(a)
            else:
                return NotImplemented
        result_mag = func(mags, *rest, **kwargs)
        return quants[0].__class__(result_mag, unit0)


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

    .. Note:: Because an ``immlib.UnitRegistry``'s ``Quantity`` type accepts
        ``units=None`` to mean "no units" rather than pint's own
        "dimensionless", a handful of pint's own internal code paths that
        construct a quantity without specifying units at all--such as
        parsing the literal word ``"dimensionless"`` or an empty expression
        string--are affected by this reinterpretation too: on an
        ``immlib.UnitRegistry``, such a quantity comes back with ``units``
        of ``None`` rather than pint's real ``dimensionless`` unit. This is
        a deliberate, documented consequence of ``immlib.UnitRegistry``'s
        design, and it is scoped to registries of this type; a plain
        ``pint.UnitRegistry`` is entirely unaffected.
    """
    Quantity = type('ImmlibQuantity', (Quantity, pint.UnitRegistry.Quantity), {})


_initial_global_ureg = UnitRegistry()
# We want to disable the awful pint warning for numpy if it's present:
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _initial_global_ureg.Quantity([])
# Make sure there's a pixel unit
if not hasattr(_initial_global_ureg, 'pixels'):
    _initial_global_ureg.define('pixel = [image_length] = px')
@docwrap('immlib.like_unit')
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
        from immlib import units as ureg
    if is_str(obj):
        if ureg is None:
            raise ValueError(
                "cannot determine if str is unit-like when ureg is None")
        return hasattr(ureg, obj) and isinstance(getattr(ureg, obj), pint.Unit)
    else:
        return False
@docwrap('immlib.unit')
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
        from immlib import units as ureg
    if is_quant(obj):
        obj = obj.u
    if is_unit(obj):
        if ureg is None or ureg is unitregistry(obj):
            return obj
        else:
            return getattr(ureg, str(obj))
    elif is_str(obj):
        if ureg is None:
            from immlib import units as ureg
        return getattr(ureg, obj)
    else:
        raise ValueError(f'unrecognized unit argument: {obj}')
_unitlike_types = (str, pint.Unit, pint.Quantity)
@docwrap('immlib.alike_units')
def alike_units(a, b, /, *, ureg=None):
    """Returns ``True`` if the arguments are alike units, otherwise ``False``.

    ``alike_units(a, b)`` returns ``True`` if `a` and `b` can be cast to
    each other in terms of units and ``False`` otherwise. Both `a` and `b`
    can either be units, unit names, or quantities with units. If either `a`
    or `b` is neither a unit nor a quantity, then it is considered equivalent
    to having units of ``None``, i.e., no units.

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
        from immlib import units as ureg
    if ureg is None:
        ureg = unitregistry(a, None)
        if ureg is None:
            ureg = unitregistry(b, Ellipsis)
    if not isinstance(a, _unitlike_types):
        a = 'dimensionless'
    if not isinstance(b, _unitlike_types):
        b = 'dimensionless'
    return ureg.is_compatible_with(a, b)
@docwrap('immlib.quant')
def quant(mag, /, unit=Ellipsis, *, ureg=None):
    """Returns a ``pint.Quantity`` object with the given magnitude and unit.

    ``quant(mag, unit)`` returns a ``pint.Quantity`` object with the given
    magnitude `mag` and `unit`. If `mag` is alreaady a ``pint.Quantity``, then
    it is converted into the given units and returned (a copy of `mag` is made
    only if necessary); if the units of `mag` in this case are not compatible
    with `unit`, then an error is raised. If `mag` is not a quantity, then the
    given `unit` is used to create the quantity.

    ``quant(mag)`` is equivalent to ``quant(mag, Ellipsis)``. If `mag` is
    already a ``pint.Quantity`` object (of any kind, ``immlib.Quantity`` or
    not), it is returned unchanged, in its own unit registry; a plain
    ``pint.Quantity`` is not promoted to an ``immlib.Quantity`` by this
    function. If `mag` is not a quantity, a unit-less (``units=None``)
    quantity is created by default (see the ``unit=None`` note below).

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
    ValueError
        If a unit-less (``unit=None``) quantity is requested using a unit
        registry that is not an ``immlib.UnitRegistry``.

    """
    if ureg is Ellipsis:
        from immlib import units as ureg
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
            return qcls(mag.magnitude, None)
        elif unit is Ellipsis:
            # mag is returned exactly as given, in its own unit registry;
            # a plain pint.Quantity is not promoted to immlib.Quantity here.
            return mag
        else:
            # mag.to() stays within mag's own unit registry; if the caller
            # also passed an explicit, different `ureg`, that is handled
            # below.
            q = mag.to(unit)
    else:
        if ureg is None:
            from immlib import units as ureg
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
        q = qcls(mag, unit)
    q_ureg = unitregistry(q)
    if q_ureg is not ureg:
        # The caller explicitly requested a different registry than the one
        # q ended up in (e.g., mag belonged to a different registry than an
        # explicitly-given ureg); re-home q into the requested registry.
        return ureg.Quantity(q._magnitude, q._units)
    else:
        return q
@docwrap('immlib.mag')
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
@docwrap('immlib.promote')
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
        from immlib import units as ureg
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
