# -*- coding: utf-8 -*-
################################################################################
# immlib/util/_core.py


# Dependencies #################################################################

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


# Units and Quantities #########################################################

# Units are fundamentally treated as part of the immlib type-system. Immlib
# functios that deal with an object's type typically take an option `unit` that
# can be used to change the function's behavior depending on the units attached
# to an object.
# Setup pint / units:
from pint import UnitRegistry
@docwrap
def is_ureg(obj):
    """Returns `True` if an object is a `ping.UnitRegistry` object.

    `is_ureg(obj)` returns `True` if the given object `obj` is an instance
    of the `pint.UnitRegistry` type.

    Parameters
    ----------
    obj : object
        The object whose quality as an `UnitRegistry` object is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an instance of `UnitRegistry`, otherwise `False`.
    """
    return isinstance(obj, UnitRegistry)
from pint import Unit
@docwrap
def is_unit(q, ureg=None):
    """Returns `True` if `q` is a `pint` unit and `False` otherwise.

    `is_unit(q)` returns `True` if `q` is a `pint` unit and `False` otherwise.

    Parameters
    ----------
    q : object
        The object whose quality as a `pint` unit is to be assessed.
    ureg : UnitRegistry or None, optional
        The `pint` `UnitRegistry` object that the given unit object must belong
        to. If `None` (the default), then any unit registry is allowed. If
        `Ellipsis`, then the `immlib.units` registry is used. Otherwise, this
        must be a specific `UnitRegistry` object.

    Returns
    -------
    boolean
        `True` if `q` is a `pint` unit and `False` otherwise.

    Raises
    ------
    TypeError
        If the `ureg` parameter is not a `UnitRegistry`, `Ellipsis`, or `None`.
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
@docwrap
def is_quant(obj, unit=Ellipsis, ureg=None):
    """Returns `True` if given a `pint` quantity and `False` otherwise.

    `is_quant(q)` returns `True` if `q` is a `pint` quantity and `False`
    otherwise. The optional parameter `unit` may additionally specify a unit
    that `obj` must be compatible with.

    Parameters
    ----------
    obj : object
        The object whose quality as a quantity is to be assessed.
    unit : UnitLike or None, optional
        The unit that the object must have in order to be considered valid. This
        may be a `pint` unit or unit-name (see also `immlib.unit`), a list or
        tuple of such units/unit-names, or `None`. If `Ellipsis` (the default),
        then the object must be a `Quantity` object, but it doesn't matter what
        the unit of the object is. Otherwise, the object must have a unit
        equivalent to the unit or to one of the units given (`unit` may be a
        tuple of possible units). The `UnitRegistry` objects for the units given
        via this parameter are ignored; only the `ureg` parameter influences the
        `UnitRegistry` requirements. Note that the value `None` for a unit type
        indicates a scalar without a unit (i.e., an object that is not a
        quantity), and so, while `None` is a valid value, this function will
        always return `False` when it is passed.
    ureg : pint.UnitRegistry or Ellipsis or None, optional
        The `pint` `UnitRegistry` object to use for units. If `Ellipsis`, then
        `immlib.units` is used. If `ureg` is `None` (the default), then a
        specific unit registry is not checked.

    Returns
    -------
    boolean
        `True` if `obj` is a `pint` quantity whose unit is compatible with the
        requested `unit` and `False` otherwise.

    Raises
    ------
    %(immlib.util._quantity.is_unit.raises)s
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
    return (True  if unit is Ellipsis else
            False if unit is None     else
            obj.is_compatible_with(unit))
class default_ureg(object):
    """Context manager for setting the default `immlib` unit registry.

    The following code-block can be used to evaluate the code represented by
    `...` using the unit-registry `ureg` as the default `immlib.units` registry:

    ```python
    with immlib.default_ureg(ureg):
        ...
    ```
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
_initial_global_ureg = pint.UnitRegistry()
# We want to disable the awful pint warning for numpy if it's present:
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _initial_global_ureg.Quantity([])
# Make sure there's a pixel unit
if not hasattr(_initial_global_ureg, 'pixels'):
    _initial_global_ureg.define('pixel = [image_length] = px')
@docwrap
def like_unit(q, ureg=Ellipsis):
    """Returns `True` if `q` is or names a `pint` unit and `False` otherwise.

    `like_unit(q)` returns `True` if `q` is a `pint` unit or a string that names
    a `pint` unit and `False` otherwise.

    Parameters
    ----------
    q : object
        The object whose quality as a `pint` unit is to be assessed.
    ureg : UnitRegistry or Ellipsis or None, optional
        The `pint` `UnitRegistry` object to use. If `None`, then any registry is
        allowed but an exception is raised if `q` is a string because there is
        no registry in which to look it up. If `Ellipsis` (the default), then
        the `immlib.units` registry is required.

    Returns
    -------
    boolean
        `True` if `q` is a `pint` unit or a string naming a `pint` unit and
        `False` otherwise.

    """
    if isinstance(q, pint.Unit):
        return True
    if ureg is Ellipsis:
        from immlib import units as ureg
    if is_str(q):
        if ureg is None:
            raise ValueError(
                "cannot determine if str is unit-like when ureg is None")
        return hasattr(ureg, q) and isinstance(getattr(ureg, q), pint.Unit)
    else:
        return False
@docwrap
def unit(obj, ureg=None):
    """Converts the argument into a a `Unit` object.

    `unit(u)` returns the `immlib`-library unit object for the given unit object
    `u` (which may be from a separate `pint.UnitRegistry` instance).

    `unit(uname)` returns the unit object for the given unit name string
    `uname`.

    `unit(q) returns the unit of the given quantity object `q`.

    Parameters
    ----------
    obj : object
        The object that is to be converted to a unit.
    ureg : UnitRegistry or None or Ellipsis, optional
        The unit registry to convert the object into. If `Ellipsis`, then
        `immlib.units` is used. If `None` (the default), then the unit registry
        for `obj` is used if `obj` is a quantity or unit already, and
        `immlib.units` is used if not. Otherwise, must be a unit registry.

    Returns
    -------
    pint.Unit
        The `Unit` object associated with the given argument.

    Raises
    ------
    TypeError
        When the argument cannot be converted to a `Unit` object.
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
@docwrap
def alike_units(a, b, ureg=None):
    """Returns `True` if the arguments are alike units, otherwise `False`.

    `alike_units(a,b)` returns `True` if `a` and `b` can be cast to each other
    in terms of units and `False` otherwise. Both `a` and `b` can either be
    units, unit names, or quantities with units. If either `a` or `b` is neither
    a unit nor a quantity, then it is considered equivalent to having units of
    `None`, i.e., no units. `None` is compatible with dimensionless units but is
    considered incompatible with all other units.

    Parameters
    ----------
    a : UnitLike
        A unit object or the name of a unit or a quantity.
    b : UnitLike
        A unit object or the name of a unit or a quantity.
    ureg : UnitRegistry or None or Ellipsis, optional
        The `pint` `UnitRegistry` object to use. If `Ellipsis`, then the
        `immlib.units` registry is used. If `None`, then the registry of object
        `a` is used if available or that of object `b` if not. If neither `a`
        nor `b` has an available registry, then `immlib.units` is used.

    Returns
    -------
    boolean
        `True` if the units `a` and `b` are alike and `False` otherwise.
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
@docwrap
def quant(mag, unit=Ellipsis, ureg=None):
    """Returns a `Quantity` object with the given magnitude and unit.

    `quant(mag, unit)` returns a `Quantity` object with the given magnitude and
    unit. If `mag` is alreaady a `Quantity`, then it is converted into the given
    units and returned (a copy of `mag` is made only if necessary); if the units
    of `mag` in this case are not compatible with `unit`, then an error is
    raised. If `mag` is not a quantity, then the given `unit` is used to create
    the quantity.

    The value `unit=None` is not equivalent to `unit='dimensionless'`; rather,
    `unit=None` us used throughout immlib to indicate a non-quantity such as a
    plain PyTorch tensor or a NumPy array. Accordingly, an exception is raised
    when `unit=None` is given.

    `quant(mag)` is equivalent to `quant(mag, Ellipsis)`. Both return `mag` if
    `mag` is already a `Quantity`; otherwise they return a quantity with
    dimensionless units.

    Parameters
    ----------
    mag : object
        The magnitude to be given a unit.
    unit : unit-like or None or Ellipsis, optional
        The units to use in the returned quantity. If `Ellipsis` (the default),
        then dimensionless units are assumed unless the `mag` argument already
        is a quantity with its own units.
    ureg : UnitRegistry or None or Ellipsis, optional
        The `pint` `UnitRegistry` object to use for units. If `ureg` is
        `Ellipsis`, then `immlib.units` is used. If `ureg` is `None` (the
        default), then no specific coersion to a `UnitRegistry` is performed.

    Returns
    -------
    pint.Quantity
        A quantity object representing the given magnitude and units.

    Raises
    ------
    ValueError
        If `unit` is `None`.
    """
    if ureg is Ellipsis:
        from immlib import units as ureg
    if unit is None:
        raise ValueError(
            "quant cannot create a quantity with a unit of None;"
            " use 'dimensionless' instead")
    if is_quant(mag):
        if ureg is None:
            ureg = unitregistry(mag)
        q = mag if unit is Ellipsis else mag.to(unit)
    else:
        if ureg is None:
            from immlib import units as ureg
        if unit is Ellipsis:
            unit = ureg.dimensionless
        q = ureg.Quantity(mag, unit)
    q_ureg = unitregistry(q)
    if q_ureg is not ureg:
        return ureg.Quantity(q.m, q.u)
    else:
        return q
@docwrap
def mag(val, unit=Ellipsis, strict=False):
    """Returns the magnitude of the given object.

    `mag(quantity)` returns the magnitude of the given quantity, regardless of
    the quantity's unit.

    `mag(obj)`, for a non-quantity object `obj`, simply returns `obj`.

    `mag(arg, unit)` returns `arg.m_as(unit)` if `arg` is a quantity and returns
    `arg` itself if `arg` is not a quantity.

    `mag(arg, Ellipsis)` is equivalent to `mag(arg)`.

    If `mag(quantity, unit)` is given a quantity not compatible with the given
    unit, then an error is raised.

    Note that if the first argument to `mag()` is not a quantity, then the
    `unit` argument is always ignored, and the first argument is returned as-is.

    Parameters
    ----------
    val : object
        The object that is to be converted into a magnitude.
    unit : unit-like or None or Ellipsis, optional
        The unit in which the magnitude of the argument `val` should be
        returned. The default argument of `Ellipsis` indicates that the value's
        native unit, if any, should be used. A value of `None` indicates that
        the `val` must have no units (i.e., not be a quantity), otherwise an
        exception is raised.
    strict : boolean, optional
        Whether strict matching of the unit is performed. If `False` (the
        default), then a non-quantity (such as a plain numpy array) is treated
        as a quantity whose unit is the type passed in the `unit` parameter; if
        `True`, then `val` must be compatible with the `unit` parameter or an
        error is raised.

    Returns
    -------
    object
        The magnitude of `val` in the requested unit, if `val` is a quantity,
        or `val` itself, if it is not a quantity.

    Raises
    ------
    DimensionalityError
        If the given `val` is a quantity whose unit is not compatible with the
        `unit` parameter.
    ValueError
        If `unit` is None but `val` is a quantity or if a unit is requested of a
        non-quantity with the `strict` option enabled.
    """
    if is_quant(val):
        if unit is None:
            raise ValueError("unit=None requested of quantity")
        elif unit is Ellipsis:
            return val.m
        else:
            return val.m_as(unit)
    elif strict is True:
        if unit is not None:
            raise ValueError(
                f"unit '{unit}' does not strictly match non-quantity")
    return val


# Promotion ####################################################################

def _array_promote(*args, ureg=None):
    return [to_array(el, ureg=ureg) for el in args]
@alttorch(_array_promote)
@docwrap
def promote(*args, ureg=None):
    """Promotes all arguments into quantities with compatible magnitudes.

    `promote(a, b, c...)` converts all of the passed arguments into numerical
    quantity objects and returns them as a list. The returned arguments will all
    have compatible (promoted) types or magnitude types.

    Promotion is determined based on the object type. If any of the objects are
    PyTorch tensors or quantities with tensor magnitudes, then all of the
    returned quantities will have for their magnitudes PyTorch tensors with the
    same profile (e.g, device) as the tensor argument(s). Otherwise, the
    returned quantities will converted into array types. The purpose of this
    promotion is to ensure that all arguments can be combined into an expression
    (PyTorch tensor operations generally require that all arguments are
    tensors).

    Parameters
    ----------
    *args
        The arguments that are to be promoted.
    ureg : pint.UnitRegistry or None, optional
        The `pint` `UnitRegistry` object to use for units. If `ureg` is
        `Ellipsis`, then `immlib.units` is used. If `ureg` is `None` (the
        default), then no specific coersion to a `UnitRegistry` is performed.

    Returns
    -------
    list of numeric objects
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
