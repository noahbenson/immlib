---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# Utilities for Physical Units

`immlib` uses the [`pint`](https://pint.readthedocs.io/) library to represent
physical quantities such as `10 * il.units.mm` or `[1.0, 2.0, 3.0] *
il.units.seconds`. Quantities with units are generally handled seamlessly by
`immlib`'s functions, including the numerical utilities described in
[Numerical Utilities](/user-guide/utils/numerical.md) and the backend-aware
math functions described in [The `immlib.math` Module](/user-guide/math.md),
but the library also provides a number of helper functions for creating,
inspecting, and converting quantities directly.

`immlib`'s `Quantity` type is a subclass of `pint.Quantity` that adds one
important feature beyond what `pint` provides on its own: the ability for a
quantity to have *no units at all*, as distinct from `pint`'s own
`dimensionless`. This is covered in detail below, in
[Quantities Without Units](#quantities-without-units).


## The Unit Registry

Like `pint`, `immlib` tracks units using a `pint.UnitRegistry` object. The
global registry that `immlib`'s own functions use by default is available as
`immlib.units`:

```{code-cell}
import immlib as il

il.units
```

```{code-cell}
# Units and quantities can be created from this registry directly, as with
# any pint.UnitRegistry.
10 * il.units.mm
```

The default registry can be temporarily replaced within a `with` block using
`immlib.default_ureg`:

```{code-cell}
import pint

other_registry = pint.UnitRegistry()
with il.default_ureg(other_registry):
    print(il.units is other_registry)
print(il.units is other_registry)
```

Most `immlib` functions that work with units accept a `ureg` parameter that
lets a specific unit registry be used for a single call instead of changing
the default.


## Creating and Converting Quantities

### `quant`

`il.quant(magnitude, unit)` is the primary way to create a quantity. If
`magnitude` is already a quantity, it is converted to `unit` (a copy is made
only if necessary); if `magnitude` is not a quantity, a new one is created.

```{code-cell}
il.quant(3, 'mm')
```

```{code-cell}
import numpy as np

il.quant(np.array([1.0, 2.0, 3.0]), 'seconds')
```

```{code-cell}
# A quantity can be converted to a different (compatible) unit by passing it
# back through quant.
il.quant(il.quant(1, 'm'), 'mm')
```

Calling `il.quant(magnitude)` with no unit at all (or `il.quant(magnitude,
Ellipsis)`, the default) returns `magnitude` unchanged if it is already a
quantity, and otherwise creates a *unit-less* quantity--see
[Quantities Without Units](#quantities-without-units) below.

### `mag` and `unit`

`il.mag(obj, unit)` is the inverse of `quant`: it extracts a plain magnitude
(a NumPy array, PyTorch tensor, or Python number) from a quantity, optionally
converting to `unit` first. If `obj` is not a quantity, it is returned as-is
(the `unit` argument is ignored unless `strict=True` is passed).

```{code-cell}
q = il.quant(1.0, 'm')
(il.mag(q), il.mag(q, 'cm'), il.mag(q, 'mm'))
```

`il.unit(obj)` converts its argument into a `pint.Unit` object; `obj` can be
a unit, a unit name string, or a quantity (in which case the quantity's own
unit is returned).

```{code-cell}
(il.unit('mm'), il.unit(q))
```


## Querying Quantities and Units

A handful of predicate functions let you test whether an object is a
quantity, a unit, or a unit registry, and whether two units are compatible
with one another.

* `il.is_quant(obj)` returns `True` if `obj` is a `pint.Quantity` (of any
  kind--`immlib.Quantity` or a plain `pint.Quantity`); `il.is_quant(obj,
  unit)` additionally requires that `obj`'s unit be compatible with `unit`.
* `il.is_unit(obj)` returns `True` if `obj` is a `pint.Unit` object.
* `il.like_unit(obj)` returns `True` if `obj` is a `pint.Unit` *or* a string
  that names one (useful for validating a "unit-like" argument before
  passing it to something that requires an actual `pint.Unit`).
* `il.is_ureg(obj)` returns `True` if `obj` is a `pint.UnitRegistry`.
* `il.alike_units(a, b)` returns `True` if the units `a` and `b` (each of
  which may be a unit, a unit name, or a quantity) are dimensionally
  compatible--e.g., millimeters and feet, but not millimeters and seconds.

```{code-cell}
(il.is_quant(q), il.is_quant(q, 'mm'), il.is_quant(q, 's'))
```

```{code-cell}
(il.alike_units('mm', 'ft'), il.alike_units('mm', 's'))
```


(quantities-without-units)=
## Quantities Without Units

`pint` requires every quantity to have *some* unit, even if that unit is
`dimensionless`--a real, measurable unit (radians and steradians, for
example, are dimensionless). `immlib` extends `pint.Quantity` with a second,
distinct concept: a quantity whose `units` is `None`, meaning that it isn't a
physical quantity with units at all--just a NumPy array, PyTorch tensor, or
Python number with some `Quantity` bookkeeping attached to it. This is
*not* the same as `dimensionless`:

```{code-cell}
none_q = il.quant([1.0, 2.0, 3.0])
dimensionless_q = il.quant([1.0, 2.0, 3.0], 'dimensionless')
(none_q.units, dimensionless_q.units)
```

A non-quantity value passed to `il.quant()` with no `unit` argument becomes a
unit-less (`units=None`) quantity by default--*not* a `dimensionless` one.
The same is true of `il.quant(obj, None)`, which strips whatever units `obj`
previously had.

```{code-cell}
none_q
```

```{note}
A unit-less quantity can only be created using an `immlib.UnitRegistry` (such
as the default `immlib.units`); a plain `pint.UnitRegistry` has no way to
represent "no units" at all, and `il.quant(x, None)` raises a `ValueError` if
the relevant registry is a plain `pint.UnitRegistry`.
```

For essentially every purpose--arithmetic, comparisons, conversion to
`int`/`float`/`complex`, string formatting--a unit-less quantity behaves
exactly like its bare magnitude. Combining it with another unit-less
quantity always produces another unit-less quantity:

```{code-cell}
none_q + il.quant([1.0, 1.0, 1.0])
```

Combining a unit-less quantity with a quantity that *does* have real units
defers entirely to the real-unit quantity's own rules. This means addition
and subtraction, which require compatible units, raise an error, while
multiplication and division, which do not, succeed by treating the
unit-less side as a plain scalar or array:

```{code-cell}
real_q = il.quant([1.0, 2.0, 3.0], 'mm')
try:
    none_q + real_q
except Exception as e:
    print(f'{type(e).__name__}: {e}')
```

```{code-cell}
none_q * real_q
```

Comparisons follow the same rule: a unit-less quantity compares equal to
another unit-less quantity based on their bare magnitudes, and compares
*unequal* (rather than raising) to a quantity with real units:

```{code-cell}
(il.quant(5.0) == il.quant(5.0), il.quant(5.0) == il.quant(5.0, 'm'))
```

`str()`, `format()`, `int()`, `float()`, and `complex()` on a unit-less
quantity all pass straight through to the underlying magnitude:

```{code-cell}
print(none_q)
print(f'{il.quant(3.5)}')
```

```{note}
Because an `immlib.UnitRegistry`'s `Quantity` type interprets `units=None` as
"no units" rather than `pint`'s own `dimensionless`, a few of `pint`'s own
internal code paths that construct a quantity without specifying any unit at
all--such as parsing the literal string `"dimensionless"` or an empty unit
expression--are affected by this too: on an `immlib.UnitRegistry`, such a
quantity comes back with `units` of `None` rather than `pint`'s real
`dimensionless` unit. This is scoped to `immlib.UnitRegistry`; a plain
`pint.UnitRegistry` is entirely unaffected.
```


## Combining Arrays and Tensors: `promote`

`il.promote(*args)` converts a collection of arguments--quantities, arrays,
tensors, or plain numbers--into a list of magnitudes with compatible types:
if any argument is (or wraps) a PyTorch tensor, everything is converted to a
tensor with a matching device; otherwise everything is converted to a NumPy
array. This is useful when writing a function that needs to combine several
arguments in an expression that requires them all to share a type (PyTorch
operations, in particular, generally require every operand to be a tensor).

```{code-cell}
import torch

il.promote(np.array([1.0, 2.0]), 3.0, torch.tensor([4.0, 5.0]))
```


## Math on Quantities: `immlib.math`

Ordinary Python arithmetic operators (`+`, `-`, `*`, `/`, `**`, comparisons,
and `@` for matrix multiplication) all work directly on `immlib.Quantity`
objects, including unit-less ones, as shown above. For everything else--
elementary functions, reductions, shape manipulation--`immlib` provides a
dedicated namespace, `immlib.math`, that works uniformly across
`immlib.Quantity` objects, plain NumPy arrays, and plain PyTorch tensors,
automatically selecting the right backend. See
[The `immlib.math` Module](/user-guide/math.md) for the full reference.

```{code-cell}
import immlib.math as im

im.sqrt(il.quant([4.0, 9.0, 16.0], 'm**2'))
```


(using-numpy-and-pytorch-functions-directly)=
## Using NumPy and PyTorch Functions Directly

In addition to `immlib.math`, a curated, tested subset of NumPy and PyTorch's
own functions can be called directly on an `immlib.Quantity`, and they do the
natural, unit-aware thing--including for unit-less quantities, which `pint`'s
own (unmodified) NumPy/PyTorch dispatch does not handle correctly. Under the
hood, this is implemented via NumPy's `__array_ufunc__`/`__array_function__`
protocols and PyTorch's `__torch_function__` protocol.

```{code-cell}
q = il.quant([1.0, 4.0, 9.0], 'm**2')
np.sqrt(q)
```

```{code-cell}
tq = il.quant(torch.tensor([1.0, 4.0, 9.0]), 'm**2')
torch.sqrt(tq)
```

```{code-cell}
# Axis-based reductions work too, using NumPy's own argument names for the
# NumPy path...
a2 = il.quant(np.array([[1.0, 2.0], [3.0, 4.0]]), 's')
np.sum(a2, axis=0)
```

```{code-cell}
# ...and PyTorch's own argument names (dim, keepdim, correction) for the
# PyTorch path.
t2 = il.quant(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), 'm')
torch.var(t2, dim=0, correction=0)
```

This support covers only a deliberately curated, tested subset of NumPy's
and PyTorch's functions--the same set of operations `immlib.math` covers, plus
the ordinary arithmetic/comparison operators. A function that falls outside
this subset behaves exactly as it always has: it either falls back to
`pint`'s own existing (real-units-only) handling, or, if `pint` doesn't
implement it either, raises the same error it would without any of this
support (`immlib` never silently guesses at unit semantics for an
unsupported function).

```{code-cell}
try:
    torch.linalg.det(il.quant(torch.eye(2), 'm'))
except TypeError as e:
    print(f'{type(e).__name__}: {e}')
```

```{note}
`torch.min` and `torch.max` are deliberately left unsupported: called with a
`dim` argument, they return PyTorch's own `(values, indices)` tuple, which
`immlib.math.min`/`immlib.math.max` (matching NumPy's `min`/`max`, which
return only the values) cannot reproduce without silently changing
`torch.min`/`torch.max`'s own documented contract for a `Quantity`
argument. Use `immlib.math.min`/`immlib.math.max` instead if you only need
the values.
```
