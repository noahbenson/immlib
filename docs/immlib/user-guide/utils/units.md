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

The replacement applies only to the thread (or asynchronous task) that
enters the `with` block, so other threads continue to use their own default
registry. Assigning `il.units = registry` instead changes the default
registry for every thread.

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

`il.quant()` only accepts numerical values. NumPy arrays, PyTorch tensors, and
SciPy sparse arrays are used as the magnitude exactly as given (`il.quant()`
never converts between NumPy and PyTorch or moves a tensor to another
device), while Python numbers, NumPy scalars, and lists of numbers are
converted into NumPy arrays--scalars become 0-dimensional arrays. Strings and
other non-numerical values raise a `TypeError` (use
`il.units.Quantity('5 mm')` to parse a quantity from a string).

```{code-cell}
(type(il.quant(5).m), il.quant(5).m.ndim)
```

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
works exactly as if the unit-less quantity were replaced by its bare
magnitude: for any supported operator `op`, `op(il.quant(x), il.quant(y,
u))` is equivalent to `(x, y) = il.promote(x, y); op(x, il.quant(y, u))`.
`pint` treats a bare value as dimensionless, so addition and subtraction
with a quantity whose units have dimensions raise an error, while
multiplication and division succeed by treating the unit-less side as a
plain scalar or array:

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
*unequal* (an all-false result, rather than an error) to a quantity with
real dimensions--just as a quantity with real units compares unequal to one
with incompatible units. (As in `pint`, a bare value that is entirely zero
or NaN is the one exception: it is compared by magnitude alone.) Ordering
comparisons (`<`, `>=`, etc.) between a unit-less quantity and a quantity
with real dimensions raise an error. These rules are the same for NumPy
and PyTorch magnitudes.

```{code-cell}
(il.quant(5.0) == il.quant(5.0), il.quant(5.0) == il.quant(5.0, 'm'))
```

`str()`, `format()`, `int()`, `float()`, and `complex()` on a unit-less
quantity all pass straight through to the underlying magnitude:

```{code-cell}
print(none_q)
print(f'{il.quant(3.5)}')
```

Unit-less quantities can be pickled (they are restored, like every
`immlib.Quantity`, in the `immlib.units` registry, since unit registries
themselves are not pickled), and a unit-less quantity with a 0-dimensional
magnitude hashes like the number it contains.

```{note}
Only an explicit `units=None` means "no units". Quantities that `pint` itself
creates without being given units--for example, by parsing the string
`"dimensionless"`--have `pint`'s real `dimensionless` unit, even in an
`immlib.UnitRegistry`.
```


(quantity-mutability)=
## Mutability

Like `pint.Quantity`, `immlib.Quantity` is mutable, but using its mutating
features is strongly discouraged. These are:

* the in-place operators (`q += x`, `q *= x`, etc.);
* the `ito` family of methods (`ito`, `ito_base_units`, `ito_reduced_units`,
  `ito_root_units`, and `ito_preferred`);
* item assignment (`q[k] = v`) and in-place NumPy methods such as `fill` and
  `put`.

Changing a quantity changes it for every part of a program (and every
thread) that refers to it, and these changes are not thread-safe: some of
them replace a quantity's magnitude and units one after the other, so another
thread can briefly see the new magnitude with the old units. The
non-mutating forms, such as `q = q + x` and `q = q.to('mm')`, return new
quantities and are always safe to use. A quantity that is shared can be made
immutable outright with `persist`, described next.


(quantity-persist)=
### Making a Quantity Immutable: `persist`

`q.persist()` makes `q` immutable from then on, and returns it, so that a
quantity can be persisted in the expression that creates it. Quantities are
created mutable, which `pint`'s own internal operations rely on; persisting
one is a decision made after it is fully built, usually just before it is
returned or stored somewhere that other code can see it.

```{code-cell}
import numpy as np

q = il.quant(np.array([1.0, 2.0, 3.0]), 'mm').persist()
q.is_persistent
```

Everything in the list above that would change the quantity itself now raises
a `TypeError`: the `ito` family, item assignment, assigning or deleting an
attribute, and a NumPy `out=` argument naming it.

```{code-cell}
try:
    q.ito('cm')
except TypeError as e:
    print(e)
```

The in-place *operators* are the exception: they are not refused. `q += x` on
a persistent quantity computes a new quantity and rebinds the name to it,
leaving the original untouched, exactly as `n += 1` does for an `int` and for
every other immutable Python object. Code written for mutable quantities
therefore keeps working, without mutating anything another reference can see.

```{code-cell}
p = q
p += il.quant(1.0, 'mm')

(p.m, q.m)
```

Reading and computing are unaffected, including gradient tracking for a
tensor magnitude: persistence is a rule about the quantity object, not about
the values inside it.

```{code-cell}
(q + q).to('cm')
```

```{note}
This method is named `persist` rather than `persistent`, which is what
`pcollections` calls the corresponding method, because it does something
different. A transient collection's `d.persistent()` returns a *new*,
persistent collection and leaves `d` alone; a quantity's `q.persist()` changes
`q` itself into a persistent quantity, and returns it only so that it can be
called in an expression.
```

What is frozen is the quantity: its units, and which array or tensor is its
magnitude. The *contents* of that magnitude are not, since a quantity does not
own them--`q.m[0] = 5` still works, just as mutating a list stored in a
`pcollections.pdict` still works. Freeze a NumPy magnitude with
`il.freezearray` before persisting the quantity if the values must not change
either; PyTorch has no equivalent, which is the other reason `persist` does
not attempt this itself.

```{code-cell}
arr = np.array([1.0, 2.0, 3.0])
il.freezearray(arr)
qf = il.quant(arr, 'mm').persist()

try:
    qf.m[0] = 99.0
except ValueError as e:
    print(e)
```

Copying a persistent quantity, with `copy.copy`, `copy.deepcopy`, or
`pickle`, gives a persistent quantity; `il.quant(q, persist=False)` gives an
equal quantity that is mutable. A quantity can never be made mutable again in
place.

(quantity-thread-safety)=
### Persistence and Threads

A persistent quantity can be read from any number of threads at once,
including in a free-threaded interpreter, without a lock. This is the main
reason to persist one.

The race it removes is not a stale read, which would be harmless, but a
*wrong* one. `ito` and its relatives assign the magnitude and then the
units, in two separate statements:

```python
self._magnitude = self._convert_magnitude(other, *contexts, **ctx_kwargs)
self._units = other
```

While that is in progress, another thread reading the quantity can see the
converted magnitude together with the units it had before the conversion --
a value wrong by whatever the conversion factor was. With eight threads
reading a millimeter quantity while one thread flips it between millimeters
and centimeters, every reader sees such a pair within a few hundred
iterations. Persisting the quantity removes the window by removing the
assignments; `immlib`'s own test suite checks both halves of this, in
`immlib/test/concurrency/test_threads.py`.

Persisting also settles the one write that a quantity would otherwise still
make on its own: `pint` computes `dimensionality` the first time it is asked
for and caches it on the quantity. `persist` computes it up front, so a
persistent quantity's `__dict__` stops changing at the moment it becomes
persistent, rather than on whichever thread happens to ask first.

Two things remain the caller's responsibility.

* **The contents of the magnitude.** A NumPy array that another thread
  writes to is a data race whoever holds it, and persisting the quantity
  that wraps it changes nothing about that. Use `il.freezearray` on the
  magnitude before persisting, as above. PyTorch has no equivalent, so a
  tensor magnitude that is shared between threads must simply not be
  written to.
* **`persist` itself**, which is a write like any other. Persist a quantity
  before sharing it with other threads, not afterwards.

```{note}
Thread safety here is a property of the quantity, not of the unit registry.
A `pint.UnitRegistry` is mutable -- `ureg.define(...)` adds units at
runtime -- and defining new units while other threads are parsing or
converting is a separate concern that persistence does not address. The
usual advice applies: finish setting up the registry before the threads
start.
```


## Writing Functions That Take Quantities: `quantwrap`

A function that should accept either a quantity or a plain number usually
ends up asking, argument by argument, which one it got. The `@il.quantwrap`
decorator does that once: it converts the arguments into quantities before
the function runs, so the body can assume it is always working with
quantities, and it decides what the units of the return value are afterward.
It is the quantity analogue of `il.tensor_args` and its relatives.

```{code-cell}
@il.quantwrap
def midpoint(a, b):
    # a and b are always quantities here.
    return (a + b) / 2

midpoint(il.quant(1.0, 'mm'), il.quant(3.0, 'mm'))
```

An argument that is already a quantity keeps its units; one that is not gets
units of `None`. The most important part of the design is what comes back:
if the caller passed no quantities at all, the magnitude is returned rather
than a quantity, so a caller who spoke in plain numbers is answered in plain
numbers.

```{code-cell}
midpoint(1.0, 3.0)
```

```{note}
Units of `None` are not `dimensionless`, so wrapping an argument does not
make a bare number interchangeable with a dimensional one. `midpoint(quant(1,
'mm'), 3.0)` still raises, because adding a length to a bare number is still
meaningless. What the decorator removes is the need to *ask* whether each
argument is a quantity, not the arithmetic of units.
```

### Naming Units for Arguments

`unit` converts named arguments into the units given for them. This makes no
requirement of the caller: a bare number is taken to be in that unit already,
and a quantity in a compatible unit is converted.

```{code-cell}
@il.quantwrap(unit={'dist': 'mm'})
def describe(dist):
    return f"{float(dist.m):.1f} {dist.units}"

(describe(10), describe(il.quant(1.0, 'm')))
```

`require_unit` has the same form but *requires* the caller to pass a
quantity in a compatible unit. A compatible but different unit satisfies the
requirement and is converted, so a function that requires millimeters always
sees millimeters.

```{code-cell}
@il.quantwrap(require_unit={'dist': 'mm'})
def strict(dist):
    return float(dist.m)

try:
    strict(10)
except TypeError as e:
    print(e)
```

For a function with a `*args` or `**kwargs` parameter, naming that parameter
applies its unit to every one of those arguments.

### Naming Units for the Return Value

`runit` says what unit to return the result in. It also says that the caller
cares about units, so it turns off the plain-numbers rule above.

```{code-cell}
@il.quantwrap(runit='m')
def perimeter(w, h):
    return 2 * (w + h)

perimeter(il.quant(30.0, 'cm'), il.quant(20.0, 'cm'))
```

`require_runit` is the other half of the pair, and runs first: it is a check
on the *decorated* function, which must return a quantity whose units are
compatible with what is required (a unit-less quantity is not enough). It
converts as well as checking, so that a function which sometimes answers in
metres and sometimes in millimetres cannot leave the caller guessing. `runit`
then says what the *composed* function returns, so `quantwrap(require_runit=
'm', runit='mm')` requires metres and returns millimetres.

```{warning}
`runit` does not check the decorated function. If the function returns a bare
magnitude, `runit='mm'` assumes that magnitude is already in millimeters and
labels it so, rather than raising. Only a return value that is already a
quantity in an incompatible unit is an error. Use `require_runit` when the
function must return a quantity.
```

`return_quant` settles the question outright: `True` always returns
quantities and `False` always returns magnitudes. It is applied after
`runit`, so the two compose.

```{code-cell}
@il.quantwrap(runit='m', return_quant=False)
def perimeter_m(w, h):
    return 2 * (w + h)

perimeter_m(il.quant(30.0, 'cm'), il.quant(20.0, 'cm'))
```

All of this applies to a returned tuple element by element, and to a returned
mapping value by value, without recursing further; a tuple or a mapping of
units gives one per element or key. A returned mapping keeps its own type, so
a `dict`, a `pdict` and an `ldict` all survive the trip.

```{code-cell}
from pcollections import pdict

@il.quantwrap(runit={'w': 'm', 'h': 'cm'})
def sides(w, h):
    return pdict(w=w, h=h)

sides(il.quant(30.0, 'cm'), il.quant(20.0, 'cm'))
```

`persist` settles whether the quantities that come back are persistent, and
is applied last of all. Its default, `None`, leaves each one as it is; it
says nothing about the arguments, which follow `il.quant`'s own rule.

```{code-cell}
@il.quantwrap(persist=False)
def mutable_result(a):
    return a * 2

mutable_result(il.quant(1.0, 'mm')).is_persistent
```

### Unit Registries

`quantwrap` requires an `immlib.UnitRegistry`, because it gives unit-less
arguments units of `None`, which a plain `pint.UnitRegistry` cannot
represent. By default every argument is re-homed into immlib's own registry,
as `il.ilquant` does. Pass `ureg` to name a different one; `ureg=None`, which
means "no particular registry" to `il.quant`, is an error here. When the
default is in use and the arguments disagree about which registry they belong
to, that is an error too, since there is then no unambiguous answer -- naming
a registry resolves it.

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

For NumPy, every other function that `pint` supports also works with
unit-less quantities: if all of a function's quantity arguments are
unit-less, `pint` computes the result as though they were dimensionless and
the result is returned without units (`pint` still decides which results are
quantities, so `np.argmax` returns a plain index); if some arguments have real
units, the unit-less ones are treated as bare values. NumPy ufunc methods
such as `np.add.reduce` and `np.multiply.outer`, and the `out=` argument, are
supported when every quantity argument is unit-less (`pint` does not support
them for quantities with real units).

```{code-cell}
(np.cumsum(il.quant([1.0, 2.0, 3.0])), np.add.reduce(il.quant([1.0, 2.0, 3.0])))
```

For PyTorch, support covers a deliberately curated, tested subset of
functions--the same set of operations `immlib.math` covers, plus the ordinary
arithmetic/comparison operators. A function that falls outside this subset,
or outside what `pint` supports for NumPy, raises the same error it would
without any of this support (`immlib` never silently guesses at unit
semantics for an unsupported function).

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
