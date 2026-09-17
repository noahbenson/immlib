# -*- coding: utf-8 -*-
###############################################################################
# immlib/math/__init__.py

"""``immlib.math``: a common numerical namespace over NumPy and PyTorch.

``immlib.math`` provides a small, intentionally common subset of
NumPy/PyTorch functionality (elementwise arithmetic, comparisons, elementary
functions, reductions, shape/combination operations, and matrix
multiplication) that operates uniformly on ``immlib.Quantity`` objects, plain
NumPy arrays, plain PyTorch tensors, and plain Python numbers.

For convenience, ``immlib.math`` also provides ``quant``, ``mag``,
``promote``, ``to_array``, and ``to_tensor`` (the same functions as
``immlib.quant``, etc.), so that a numerical function can be written using
only ``import immlib.math as im``; see also ``immlib.Quantity.as_input_type``.

For any call, the backend is selected automatically: if any argument's
magnitude is a PyTorch tensor, PyTorch is used; otherwise NumPy is used. Every
function returns an ``immlib.Quantity``, except for the comparisons and the
``any``/``all`` reductions, which return a plain boolean NumPy array or
PyTorch tensor (matching ordinary NumPy/PyTorch ergonomics for masks and
indexing) rather than a unit-less ``Quantity``.

See the ``immlib.math._core`` module docstring for the full set of governing
design principles, and the design spec (section 9-11) for the original
rationale.

Examples
--------
>>> import immlib as il
>>> import immlib.math as im
>>> q = il.quant([1.0, 2.0, 3.0], 'm')
>>> im.sum(q)
<Quantity(6.0, 'meter')>
>>> im.exp(il.quant([1.0, 2.0, 3.0]))
<Quantity([ 2.71828183  7.3890561  20.08553692], None)>
"""

from ..util import (quant, mag, promote, to_array, to_tensor)
from ._core import (
    abs,
    add,
    subtract,
    multiply,
    divide,
    true_divide,
    power,
    negative,
    positive,
    equal,
    not_equal,
    less,
    less_equal,
    greater,
    greater_equal,
    maximum,
    minimum,
    where,
    sqrt,
    exp,
    log,
    log10,
    sin,
    cos,
    tan,
    arcsin,
    arccos,
    arctan,
    arctan2,
    floor,
    ceil,
    round,
    sum,
    prod,
    mean,
    min,
    max,
    any,
    all,
    std,
    var,
    reshape,
    transpose,
    squeeze,
    stack,
    concatenate,
    matmul)

__all__ = (
    "quant",
    "mag",
    "promote",
    "to_array",
    "to_tensor",
    "abs",
    "add",
    "subtract",
    "multiply",
    "divide",
    "true_divide",
    "power",
    "negative",
    "positive",
    "equal",
    "not_equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    "maximum",
    "minimum",
    "where",
    "sqrt",
    "exp",
    "log",
    "log10",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "arctan2",
    "floor",
    "ceil",
    "round",
    "sum",
    "prod",
    "mean",
    "min",
    "max",
    "any",
    "all",
    "std",
    "var",
    "reshape",
    "transpose",
    "squeeze",
    "stack",
    "concatenate",
    "matmul")
