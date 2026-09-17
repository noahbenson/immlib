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
# The `immlib.math` Module

`immlib.math` is a small, deliberately common numerical namespace that sits
on top of NumPy and PyTorch. It provides elementwise arithmetic, comparisons,
elementary functions, reductions, shape/combination operations, and matrix
multiplication, all of which operate uniformly on `immlib.Quantity` objects,
plain NumPy arrays, plain PyTorch tensors, and plain Python numbers.

```{code-cell}
import immlib as il
import immlib.math as im

q = il.quant([1.0, 2.0, 3.0], 'm')
im.sum(q)
```

```{code-cell}
im.exp(il.quant([1.0, 2.0, 3.0]))
```

The functions in `immlib.math` exist mainly so that code written against
`immlib.Quantity` objects doesn't need to choose between NumPy and PyTorch
(or between `np.foo`'s and `torch.foo`'s occasionally-different argument
names and conventions) at every call site--`immlib.math` picks the backend
and the translation for you. Ordinary Python operators (`+`, `-`, `*`, `/`,
`**`, comparisons, `@`) and the curated subset of NumPy/PyTorch functions
described in [Using NumPy and PyTorch Functions Directly](#using-numpy-and-pytorch-functions-directly)
remain the more natural choice for anything they already cover.


## Writing Functions with `immlib.math`

`immlib.math` also provides `quant`, `mag`, `promote`, `to_array`, and
`to_tensor` (the same functions as `il.quant`, etc.), so a numerical function
can be written with only `import immlib.math as im`. Such a function can
accept quantities, NumPy arrays, and PyTorch tensors alike: convert each
argument with `im.quant` (arguments that aren't quantities are assumed to be
in the given units), compute the result, and return
`result.as_input_type(*args)`, which returns the result as a quantity if any
argument was a quantity, and returns its magnitude (an array or tensor)
otherwise.

```{code-cell}
import torch

def hypot(a, b):
    qa = im.quant(a, 'mm')
    qb = im.quant(b, 'mm')
    return im.sqrt(qa**2 + qb**2).as_input_type(a, b)

(hypot(3.0, 4.0),
 hypot(torch.tensor([3.0]), torch.tensor([4.0])),
 hypot(il.quant(3.0, 'cm'), 4.0))
```


(design-principles)=
## Design Principles

A few rules apply consistently across every function in this module:

* **Backend selection is automatic and per-call.** If any argument's
  magnitude is a PyTorch tensor, the PyTorch backend is used; otherwise the
  NumPy backend is used. A plain (non-quantity) tensor or array argument
  works the same way as a quantity whose magnitude is a tensor or array.
* **Every function returns an `immlib.Quantity`**, with one exception: a
  function whose natural result is a boolean array (the comparisons `equal`,
  `not_equal`, `less`, `less_equal`, `greater`, `greater_equal`, and the
  reductions `any`/`all`) returns a plain NumPy array or PyTorch tensor of
  `bool` instead--matching ordinary NumPy/PyTorch ergonomics for masks and
  indexing, rather than forcing every mask to be unwrapped before use.
* **Units are computed from each function's mathematical meaning**, not by
  delegating a whole `Quantity` to `np.foo`/`torch.foo`'s own dispatch.
  Functions like `add` and `multiply` preserve or combine units the way `+`
  and `*` already do; `sqrt` raises the unit to the 1/2 power; `var` squares
  it; functions like `exp`, `log`, and `sin` require a unit-less
  (`units=None`) input and raise a `TypeError` otherwise, since "the sine of
  3 meters" isn't a meaningful physical quantity.
* **NumPy's own semantics and argument conventions are the reference.**
  Where PyTorch's same-named function differs--argument names (`dim`/
  `keepdim` vs. `axis`/`keepdims`), differing defaults (see the `std`/`var`
  note below), or a differing result shape (`torch.min`/`torch.max`'s
  `(values, indices)` tuple)--the PyTorch side is implemented, per function,
  to match NumPy's own behavior instead.
* **Tensor computations stay on PyTorch's own differentiable operations.**
  No `immlib.math` function round-trips a tensor's magnitude through NumPy,
  so gradients flow through `immlib.math` calls exactly as they would
  through the equivalent raw PyTorch code, for any function whose
  underlying operation is itself differentiable.
* **Sparse data stays sparse when it can.** SciPy sparse arrays and matrices
  remain sparse through operations that preserve sparsity (arithmetic,
  `abs`, `sqrt`, `floor`, `ceil`, `round`, `sin`, `tan`, `arcsin`, `arctan`,
  `maximum`, `minimum`, `reshape`, `transpose`, 2-D `concatenate`, and
  `matmul`). Reductions (`sum`, `mean`, `min`, `max`, `std`, `var`, `prod`,
  `any`, `all`) are computed without densifying the input and return a dense
  NumPy array (or a scalar, for a full reduction); reductions of sparse
  PyTorch tensors likewise return dense tensors. A function whose result
  would be dense (`exp`, `log`, `log10`, `cos`, `arccos`, `where`) raises a
  `TypeError` rather than silently allocating a dense array; convert the
  input with `il.to_dense` first if that is what you want.
* **A function whose NumPy and PyTorch semantics differ too materially to
  unify is simply not provided.** `numpy.dot`, for example, behaves like
  broadcasting matrix multiplication for 2-D-and-higher input, while
  `torch.dot` is restricted to a 1-D inner product; rather than picking one
  behavior and surprising callers expecting the other, `immlib.math` has no
  `dot` function at all--use `immlib.math.matmul` (or `@`) instead.


## Elementwise Arithmetic

`abs`, `add`, `subtract`, `multiply`, `divide` (aliased as `true_divide`),
`power`, `negative`, and `positive` all behave exactly like the
corresponding Python operator or builtin applied to `immlib.Quantity`
arguments--`im.add(a, b)` is exactly `a + b`, `im.abs(a)` is exactly
`abs(a)`, and so on. They exist mainly for use as ordinary function
references (e.g. passing `im.multiply` to `functools.reduce`).

```{code-cell}
im.multiply(il.quant(2.0, 'm'), il.quant(3.0, 's'))
```


## Comparisons

`equal`, `not_equal`, `less`, `less_equal`, `greater`, and `greater_equal`
each behave like the corresponding operator, but--like `any`/`all` in
[Reductions](#reductions) below--return a plain boolean NumPy array or
PyTorch tensor rather than a `Quantity`, so the result can be used directly
as a mask or in `if`/`np.where`/boolean indexing without unwrapping it
first.

```{code-cell}
im.less(il.quant([1.0, 2.0, 3.0], 'm'), il.quant([150.0, 150.0, 150.0], 'cm'))
```

`maximum`, `minimum`, and `where` are also comparison-adjacent, but return a
`Quantity` (an elementwise choice between two quantities, not a boolean
result). Their arguments must have compatible units: if either argument has
real units, the result has the units of the first such argument, and a
unit-less argument is treated as a bare, dimensionless value (so, exactly as
with `np.maximum(x, il.quant(y, 'm'))`, combining a unit-less value with a
length raises `pint.DimensionalityError`).

The comparisons compare unit-less values the same way: a unit-less value is
unequal to a quantity with real dimensions, and quantities with
incompatible units are unequal (the result is all `False`) rather than an
error, on both backends.

```{code-cell}
im.maximum(il.quant([1.0, 5.0], 'm'), il.quant([300.0, 300.0], 'cm'))
```


## Elementary Functions

`sqrt` raises a quantity's unit to the 1/2 power along with taking the
square root of its magnitude:

```{code-cell}
im.sqrt(il.quant([4.0, 9.0, 16.0], 'm**2'))
```

`exp`, `log`, `log10`, `sin`, `cos`, `tan`, `arcsin`, `arccos`, and `arctan`
each require a unit-less (`units=None`) argument, and raise `TypeError`
otherwise--these functions aren't meaningful on a quantity with real,
physical units:

```{code-cell}
try:
    im.exp(il.quant(1.0, 'm'))
except TypeError as e:
    print(f'{type(e).__name__}: {e}')
```

`arctan2(y, x)` (matching NumPy's own `y, x` argument order) aligns the
units of `y` and `x` as `maximum` does and always returns a unit-less result,
since an angle has no unit of its own here.

`floor`, `ceil`, and `round` all preserve the input's units. `round` takes
its digit-count argument as `ndigits`, matching Python's builtin `round`
(rather than NumPy's `decimals` or PyTorch's `decimals`):

```{code-cell}
im.round(il.quant([1.234, 5.678], 'm'), ndigits=1)
```


(reductions)=
## Reductions

`sum`, `mean`, `min`, `max`, `std`, `var`, and `prod` all accept NumPy-style
`axis`/`keepdims` keywords (translated to PyTorch's `dim`/`keepdim` on the
tensor backend). `sum`, `mean`, `min`, and `max` preserve the input's units;
`var` squares them; `prod` raises them to the power of however many elements
were multiplied together.

```{code-cell}
im.sum(il.quant([[1.0, 2.0], [3.0, 4.0]], 's'), axis=0)
```

Reductions of SciPy sparse arrays return dense NumPy arrays:

```{code-cell}
import scipy.sparse as sps

im.sum(il.quant(sps.eye(3, format='csr'), 'm'), axis=0)
```

```{important}
`std` and `var` both default to `ddof=0` (NumPy's own default, a *population*
standard deviation/variance) on *both* backends--including the PyTorch
backend, where `torch.std`/`torch.var` called directly default to
`correction=1` (a *sample* standard deviation/variance) instead. Pass
`ddof=1` explicitly if you want PyTorch's default behavior.
```

`min` and `max` return only the reduced values, matching NumPy's
`min`/`max`--*not* PyTorch's own `torch.min`/`torch.max`, which additionally
return the indices of the selected elements as a `(values, indices)` tuple
when called with a `dim` argument. If you need the indices, call
`torch.min`/`torch.max` (or `.m.min(...)`/`.m.max(...)`) directly on the
magnitude instead.

`any` and `all` return a plain boolean NumPy array or PyTorch tensor, like
the comparison functions above, rather than a unit-less `Quantity`.


## Shape and Combination

`reshape`, `transpose`, and `squeeze` all preserve units and accept NumPy's
own argument conventions (e.g. `transpose(a, axes=None)` reverses all axes
by default on both backends, matching `np.transpose`'s default--even though
that requires an explicit `Tensor.permute` call on the PyTorch side, since
PyTorch has no direct equivalent to NumPy's axis-reversal default for
`transpose`).

`stack` and `concatenate` combine a sequence of quantities along a new or
existing axis respectively. If any element has real units, the result has
the units of the first such element, and every element is converted into
them (unit-less elements are treated as dimensionless values, as in
`maximum`); if every element is unit-less, so is the result.

```{code-cell}
im.stack([il.quant([1.0, 2.0], 'm'), il.quant([300.0, 400.0], 'cm')])
```


## Linear Algebra

`matmul(a, b)` is exactly `a @ b`, including its tensor-safe handling for
PyTorch magnitudes (in particular, it never risks silently converting a
gradient-tracking PyTorch tensor to NumPy). There is no `dot` function; see
[Design Principles](#design-principles) above for why.
