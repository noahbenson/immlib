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
# Numerical Utilities

The `immlib` library contains a number of utilities that are intended to make
it easier to write scientific libraries that nicely handle numerical data,
specifically NumPy arrays and PyTorch tensors. Although PyTorch and NumPy
occasionally work together and though functions can often be written to handle
either kind of input, a frequent scenario when writing scientific tools is the
choice between writing (and supporting) a NumPy version of the tool, a PyTorch
version of the tool, or both. The tools described here aim to make supporting a
library that handles both much easier.

```{note}
PyTorch is not an explicit dependency of `immlib` because the tools in `immlib`
are useful even if one does not use PyTorch. Most of the tools related to
PyTorch described below will still function correctly if you do not have
PyTorch installed (because, for example, they are written to fail when
non-PyTorch inputs are provided). If you write an API with `immlib`'s tools, it
should work for NumPy, and it should work for PyTorch when it is installed.
```


## NumPy-specific Tools

A few `immlib` tools are specific to NumPy arrays. The most critical of these
are the `is_array` and `to_array` functions. These functions provide the simple
utilities of testing whether something is a NumPy array or not and of
converting something into a NumPy array, but with many additional options and
features.

### The `is_array` Function

`is_array(x)` fundamentally tests whether the object `x` is either a NumPy
array or a SciPy sprase array, but it allows `pint` quantities whose magnitudes
are arrays to be considered arrays also (this behavior can be changed with the
`quant` optional parameter). The optional parameters to `is_array` allow one to
test for many additional constraints on the array object simultaneously. When
the following descriptions refer to "the array" they are referring to `x` when
`x` is an array and `x`'s magnitude when `x` is a `pint` quantity whose
magnitude is an array.

* **Testing whether the object is a quantity**.
  * `is_array(x, quant=True)` requires that `x` be a `pint` quantity with an array
    magnitude.
  * `is_array(x, quant=False)` requires that `x` be *not* be a `pint` quantity
    (instead it must be a plain NumPy array).
  * `is_array(x, quant=None)` (the default value) allows that `x` be either a
    NumPy array or a `pint` quantity with a NumPy array as its magnitude.
* **Testing the object's physical units**.
  * `is_array(x, quant=True, unit='mm')` requires that `x` be a `pint` quantity
    with a physical unit that is *compatible with* millimeters. Any length-based
    unit is compatible with millimeters, so a quantity in feet, for example,
    would be allowed.
  * `is_array(x, unit=None)` requires that `x` not have a unit, meaning that
    `x` must not be a `pint` quantity.
  * `is_array(x, unit=...)` (the default value) does not apply any requirements
    to the object's units.
* **Testing the dtype**.
  * `is_array(x, dtype=dt)` requires that the dtype of the array represented by
    `x` have a dtype that is a sub-dtype of `dt`. For example `is_array(x,
    dtype=np.number)` will require that `x` have a numerical dtype, but would
    allow integers, floating point, or complex dtypes.
  * `is_array(x, dtype=(dt1, dt2))` then the dtype of the array must be equal
    to one of the listed dtypes.
  * `is_array(x, dtype=None)` (the default value) does not apply any
    restrictions to the array's dtype.
* **Testing the shape**.
  * `is_array(x, ndim=d)` requires that the number of dimensions of the array
    be equal to `d`.
  * `is_array(x, ndim=(d1, d2))` requires that the number of dimensions of the array
    be equal to one of the listed dimensionalities.
  * `is_array(x, shape=sh)` requires that the shape of `x` match `sh`. When
    specifying the shape that the array must match, the object `sh` can take a
    few forms. First, it can be either a tuple of numbers of an integer. When
    an integer `n` is given, this is considered equivalent to
    `(n,)`. Additionally, the shape pattern can contain entries equal to `-1`,
    which indicates that the relevant entry of the shape tuple can have any
    size. Finally, an `Ellipsis` object (`...`) can appear in at most one entry
    of the tuple. This entry matches any number of shape dimensions. For
    example, `is_array(x, shape=(-1, 2, 3, ..., 4))` would return `True` for
    NumPy arrays with any of the following shapes: `(100, 2, 3, 4)`, `(10, 2,
    3, 1, 4)`, `(1, 2, 3, 1, 0, 4)`. It would not, however, match any of these
    shapes: `(2, 3, 4)`, `(10, 2, 3, 4, 5)`, `(10, 2, 3)`.
  * `is_array(x, numel=n)` requires that the total number of elements in the
    array (after flattening) be equal to `n`.
  * Note that `ndim`, `shape`, and `numel` can be used simultaneously.
* **Testing immutability**.
  * `is_array(x, frozen=False)` requires that `x` not be a read-only array.
  * `is_array(x, frozen=True)` requires that `x` be a read-only array.
  * `is_array(x, frozen=None)` does not apply any restrictions to the array's
    immutability.
* **Testing sparsity**.
  * `is_array(x, sparse=True)` requires that `x` be a SciPy sparse array (or a
    quantity whose magnitude is a sparse array).
  * `is_array(x, sparse=False)` requires that `x` not be a sparse array.
  * `is_array(x, sparse=None)` (the default value) does not require `x` to
    either be sparse or dense.

All of the above tests can be combined simultaneously in a single function
call. Here are a few examples:

```{code-cell}
import immlib as il, numpy as np

# Make a NumPy array we can query.
arr = np.random.randn(100, 5, 2, 1, 8)

# Is this an array?
il.is_array(arr)
```

Is `arr` an array with exactly 8000 elements?

```{code-cell}
il.is_array(arr, numel=8000)
```

Is `arr` an array that has a quantity and exactly 8000 elements?

```{code-cell}
il.is_array(arr, numel=8000, quant=True)
```

Is `arr` an array that has a quantity, exactly 8000 elements, and an shape that
ends with `5, 2, _, 8`?

```{code-cell}
il.is_array(arr, numel=8000, shape=(..., 5, 2, -1, 8))
```


### The `to_array` Function

The `to_array` function is at its code similar to NumPy's `asarray` function in
that it attempts to convert the argument into a NumPy array, if possible
without copying it. Like `is_array`, the `to_array` function considers SciPy
sprase arrays and `pint` quantities whose magnitudes are arrays to be arrays
(though this behavior can be modified with options). Overall, the `to_array`
has many of the same options as the `is_array` function, though the options are
applied to the returned array rather than used as a query.

* **Creating a quantity**.
  * `to_array(x, quant=True)` will raise an error if `x` is not already a
    quantity (because it does not know what unit to give `x`).
  * `is_array(x, quant=False)` will return an array only, stripping the unit
    off of `x` if it is a quantity object.
  * `to_array(x, quant=None)` (the default value) does not attemt to change
    whether `x` is a quantity or not and will return a quantity if `x` is
    already a quantity (but it will always ensure that the quantity's magnitude
    is an array).
* **Changing the array's physical units**.
  * `to_array(x, quant=True, unit='mm')` returns a quantity with the unit of
    millimeters. If `x` is a quantity whose unit is not compatible with
    millimeters, then an error is raised. Otherwise, the unit and magnitude are
    converted.
  * `to_array(x, unit=None)` and `to_array(x, quant=False)` will both return an
    array that is not a quantity.
  * to
  * `to_array(x, unit=...)` and `to_array(x, quant=None)` (the default values)
    do not attept to change whether the object is a quantity or its unit.
* **Setting the dtype**.
  * `to_array(x, dtype=dt)` returns an array with the given dtype `dt`.
* **Immutability**.
  * `to_array(x, frozen=False)` returns an array that is not read-only.
  * `to_array(x, frozen=True)` returns a copy of `x` that is read-only unless
    `x` is already a read-only array (in which case it is returned).
  * `to_array(x, frozen=None)` does not apply any restrictions to the array's
    immutability, leaving it unchanged.
* **Creating sparse or dense arrays**.
  * `to_array(x, sparse=True)` converts `x` into a SciPy sparse array and
    returs it.
  * `to_array(x, sparse=False)` converts `x` into a dense array and returns it.
  * `to_array(x, sparse=None)` (the default value) does not change the sparsity
    of `x`.
* **Additional options**.
  * `copy=True` can be passed to `to_array()` to force the data to be copied
    into the returned array. The default, `copy=False` only copies the data
    when needed.
  * The `order` parameter is passed along to the `numpy.array` or
    `numpy.asarray` function.
  * `detach` (default: `True`) specifies that when a PyTorch tensor that
    requires gradient tracking is passed to `to_array`, `immlib` should detach
    the tensor and return its NumPy representation. If this is set to `False`,
    then an error is raised whenever such a tensor is encountered.


## PyTorch Tensor Functions

In addition to the `is_array()` and `to_array()` functions described above,
there are two very similar functions for PyTorch tensors: `is_tensor()` and
`to_tensor()`. These two functions are largely the same as their NumPy
equivalents and thus are not documented again. They do, however have a few
differences. Primarily, both `is_tensor` and `to_tensor` accept options for
`requires_grad` and `device` that allow querying or manipulating the PyTorch
device and gradient tracking requirement. Additionally, because `PyTorch` does
not support read-only tensors, the `frozen` options are not available.


## Generic NumPy/PyTorch Functions

Finally, the functions `is_numeric()` and `to_numeric()` are generic versions
of `is_array()` and `to_array()` that work for either PyTorch tensors or NumPy
arrays. The idea behind these functions is that sometimes all that matters is
that an input variable be a numeric collection with a certain shape and type,
and that can be easily detected with a call to `is_numeric(x, dtype=int,
shape=(2,-1))` (potentially with different options). The `to_numeric()`
function always returns either a tensor or an array (or a quantity whose
magnitude is a tensor or array, keeping in mind that SciPy sprase arrays are
considered arrays).

Additionally, `immlib` defines a few utility functions related to the above
that should behave similarly for array and tensor inputs. All of these
functions accept a subset of the options that the `is_numeric` and `to_numeric`
functions accept.
* `is_sparse(x)` detects whether a tensor or array is sparse.
* `to_sparse(x)` converts `x` into a sparse array or tensor.
* `is_dense(x)` detects whether a tensor or array is dense.
* `to_dense(x)` converts `x` into a dense array or tensor.


## The `@numapi` Decorator

A common issue when designing scientific libraries is the need to choose
between NumPy and PyTorch. Because the two libraries use slightly different
APIs and have slightly different ways of expressing the same formulae, the two
can't always be programmed using the same code. To simplify the job of handling
both cases, a function can be decorated using the `@numapi` decorator, then two
different versions of the function can be declared: one for NumPy arrays and
one for PyTorch tensors.

```{code-cell}
import immlib as il

# Declare a function that uses different code for numpy arrays and pytorch
# tensors:
@il.numapi
def betafn(alpha, beta):
   """Calculates and returns the Beta function B(a, b).
   
   If any of the parameters are PyTorch tensors, then a PyTorch tensor is
   returned. Otherwise, a NumPy array is returned.
   """
   pass

# The NumPy version of this function:
@betafn.array
def _(a, b):
    from scipy.special import beta
    return beta(a, b)

# The PyTorch version of this function:
@betafn.tensor
def _(a, b):
    import torch
    la = torch.lgamma(a)
    lb = torch.lgamma(b)
    lab = torch.lgamma(a + b)
    return torch.exp(la + lb - lab)

# When we give this function a PyTorch tensor, it returns the same type:
betafn(1.0, il.to_tensor([2.5, 3.2]))
```

```{code-cell}
# But if we give it any other data, it returns NumPy arrays:
betafn(1.0, [2.5, 3.2])
```


## The `@array_args`, `@tensor_args`, and `@numeric_args` Decorators

Finally, the `@array_args`, `@tensor_args`, and `@numeric_args` decorators
ensure that any arguments to the decorated function are automatically promoted
to the associated type. Optionally, a list of the names of the arguments that
should be promoted can be given, causing the decorator to skip any argument
that isn't listed. The `numeric_args` decorator promotes all arguments (or all
listed arguments) to arrays or all arguments to tensors but it will never
promote some to arrays and others to tensors; tensors are used if any of the
listed arguments are tensors.

The `@tensor_args` deocrator has an extra option, `keep_arrays`, whose default
value is `False`. If this value is set to `True`, then all returned tensors are
converted back into NumPy arrays *if* none of the arguments were tensors. This
effectively allows one to write a PyTorch implementation of a numerical
function that can then operate over PyTorch tensors or NumPy arrays.

The following example demonstrates this method of writing a PyTorch
implementation only:

```{code-cell}
import immlib as il

@il.tensor_args(keep_arrays=True)
def betafn(a, b):
    import torch
    print('Type of a:', type(a))
    print('Type of b:', type(a))
    la = torch.lgamma(a)
    lb = torch.lgamma(b)
    lab = torch.lgamma(a + b)
    return torch.exp(la + lb - lab)
 
# Call the function using PyTorch tensors:
betafn(1.0, il.to_tensor([2.5, 3.2]))
```

```{code-cell}
# Call the function without PyTorch tensors:
betafn(1.0, [2.5, 3.2])
```
