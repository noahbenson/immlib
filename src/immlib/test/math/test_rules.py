# -*- coding: utf-8 -*-
################################################################################
# immlib/test/math/test_rules.py
#
# Tests of the two rules that govern immlib.math and immlib.Quantity's
# numerical methods (see the immlib.math._core module docstring):
#
#   Rule 1. A function gives equal results for a NumPy array and a PyTorch
#           tensor that are equal; only the type of the result differs,
#           following the type of the input.
#   Rule 2. Nothing breaks PyTorch's gradient tracking: a tensor argument is
#           computed on with PyTorch operations, never through NumPy.
#
# These are checked here over the whole namespace at once, rather than
# function by function, so that a function added later is covered by the
# same rules without anyone having to remember them.


# Dependencies #################################################################

import warnings
from unittest import TestCase

import numpy as np
import pint
import torch

import immlib as il
import immlib.math as im
from immlib import quant


# The calls under test #########################################################

# (name, args, kwargs, units): `units` is the unit given to every quantity
# argument, and is None for the functions that require a unit-less argument.
_ARRAY = np.array([[1.0, 2.0, 3.0], [4.0, 0.5, 6.0]])
_VEC = np.array([1.0, 2.0, 3.0])

UNARY_CALLS = [
    ('abs', (), {}, 'mm'),
    ('negative', (), {}, 'mm'),
    ('positive', (), {}, 'mm'),
    ('sqrt', (), {}, 'mm'),
    ('exp', (), {}, None),
    ('log', (), {}, None),
    ('log10', (), {}, None),
    ('sin', (), {}, None),
    ('cos', (), {}, None),
    ('tan', (), {}, None),
    ('arctan', (), {}, None),
    ('asin', (), {}, None),
    ('arccos', (), {}, None),
    ('floor', (), {}, 'mm'),
    ('ceil', (), {}, 'mm'),
    ('round', (), {}, 'mm'),
    ('round', (1,), {}, 'mm'),
    ('conj', (), {}, 'mm'),
    ('sum', (), {}, 'mm'),
    ('sum', (1,), {}, 'mm'),
    ('sum', (), {'dim': 1, 'keepdim': True}, 'mm'),
    ('sum', (), {'axis': 1, 'keepdims': True}, 'mm'),
    ('prod', (), {}, 'mm'),
    ('prod', (1,), {}, 'mm'),
    ('mean', (), {}, 'mm'),
    ('mean', (0,), {}, 'mm'),
    ('std', (), {}, 'mm'),
    ('std', (), {'correction': 0}, 'mm'),
    ('var', (), {}, 'mm'),
    ('var', (), {'dim': 0}, 'mm'),
    ('min', (), {}, 'mm'),
    ('min', (1,), {}, 'mm'),
    ('max', (), {}, 'mm'),
    ('max', (), {'dim': 0, 'keepdim': True}, 'mm'),
    ('amin', (), {}, 'mm'),
    ('amin', (1,), {}, 'mm'),
    ('amax', (), {}, 'mm'),
    ('any', (), {}, 'mm'),
    ('all', (), {}, 'mm'),
    ('cumsum', (0,), {}, 'mm'),
    ('reshape', ((3, 2),), {}, 'mm'),
    ('reshape', (3, 2), {}, 'mm'),
    ('transpose', (0, 1), {}, 'mm'),
    ('permute', (), {}, 'mm'),
    ('permute', ((1, 0),), {}, 'mm'),
    ('squeeze', (), {}, 'mm'),
    ('squeeze', (1,), {}, 'mm'),      # a dimension that is not of size 1
    ('unsqueeze', (1,), {}, 'mm'),
    ('ravel', (), {}, 'mm'),
    ('flatten', (), {}, 'mm'),
    ('flatten', (0, 1), {}, 'mm'),
]

BINARY_CALLS = [
    ('add', 'mm'), ('subtract', 'mm'), ('multiply', 'mm'), ('divide', 'mm'),
    ('true_divide', 'mm'), ('pow', None), ('maximum', 'mm'),
    ('minimum', 'mm'), ('eq', 'mm'), ('equal', 'mm'), ('not_equal', 'mm'),
    ('less', 'mm'), ('less_equal', 'mm'), ('greater', 'mm'),
    ('greater_equal', 'mm'), ('arctan2', 'mm'),
]

# (name, args, kwargs, units) for the methods of Quantity.
METHOD_CALLS = [
    ('sum', (), {}), ('sum', (1,), {}), ('sum', (), {'axis': 0}),
    ('prod', (), {}), ('mean', (), {}), ('mean', (0,), {}),
    ('std', (), {}), ('std', (), {'correction': 0}), ('var', (), {}),
    ('min', (), {}), ('min', (1,), {}), ('max', (), {}),
    ('amin', (), {}), ('amax', (1,), {}),
    ('any', (), {}), ('all', (), {}),
    ('cumsum', (0,), {}),
    ('round', (), {}), ('round', (1,), {}),
    ('conj', (), {}), ('conjugate', (), {}),
    ('reshape', (3, 2), {}), ('reshape', ((3, 2),), {}),
    ('transpose', (), {}), ('transpose', (0, 1), {}),
    ('permute', ((1, 0),), {}),
    ('squeeze', (), {}), ('squeeze', (1,), {}),
    ('unsqueeze', (1,), {}),
    ('ravel', (), {}), ('flatten', (), {}),
    ('astype', ('float32',), {}),
]

# Operators, as (label, function of two operands).
OPERATOR_CALLS = [
    ('a + b', lambda a, b: a + b),
    ('a - b', lambda a, b: a - b),
    ('a * b', lambda a, b: a * b),
    ('a / b', lambda a, b: a / b),
    ('a // b', lambda a, b: a // b),
    ('a % b', lambda a, b: a % b),
    ('divmod(a, b)', lambda a, b: divmod(a, b)),
    ('a == b', lambda a, b: a == b),
    ('a != b', lambda a, b: a != b),
    ('a < b', lambda a, b: a < b),
    ('a >= b', lambda a, b: a >= b),
    ('a @ b.T', lambda a, b: a @ im.permute(b)),
    ('abs(a)', lambda a, b: abs(a)),
    ('-a', lambda a, b: -a),
    ('a ** 2', lambda a, b: a ** 2),
    ('a // 2', lambda a, b: a // 2),
    ('2 // a', lambda a, b: 2 // a),
    ('a % 2', lambda a, b: a % 2),
    ('2 % a', lambda a, b: 2 % a),
    ('divmod(a, 2)', lambda a, b: divmod(a, 2)),
]

# NumPy dispatch (__array_ufunc__/__array_function__) on a unit-less
# quantity, where immlib must stay in the argument's own backend.
UFUNC_CALLS = [
    ('np.exp', lambda q: np.exp(q)),
    ('np.sin', lambda q: np.sin(q)),
    ('np.sqrt', lambda q: np.sqrt(q)),
    ('np.add', lambda q: np.add(q, q)),
    ('np.multiply', lambda q: np.multiply(q, 2)),
    ('np.power', lambda q: np.power(q, 2)),
    ('np.equal', lambda q: np.equal(q, q)),
    ('np.sum', lambda q: np.sum(q)),
]


def _asarray(x):
    """Returns `x`'s values as a NumPy array, whatever it is."""
    if isinstance(x, pint.Quantity):
        x = x.m
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x)

def _tracks_grad(x):
    """Returns whether `x` is a tensor that carries gradient information."""
    return torch.is_tensor(x) and (x.requires_grad or x.grad_fn is not None)


class TestRules(TestCase):
    """Tests immlib.math's and Quantity's two governing rules."""

    def setUp(self):
        warnings.simplefilter('ignore')

    # Helpers ------------------------------------------------------------
    def _quants(self, units, grad=False):
        """Returns equal (NumPy-backed, PyTorch-backed) quantities."""
        a = quant(_ARRAY.copy(), units) if units else quant(_ARRAY.copy())
        t = torch.tensor(_ARRAY, dtype=torch.float64, requires_grad=grad)
        b = quant(t, units) if units else quant(t)
        return (a, b)

    def _call(self, fn, *args, **kwargs):
        """Returns ``('ok', result)`` or ``('err', ExceptionType)``."""
        try:
            return ('ok', fn(*args, **kwargs))
        except Exception as e:
            return ('err', type(e))

    def _assert_same(self, label, npr, tcr):
        """Asserts that a NumPy-backed result and a PyTorch-backed one are
        the same result in the two backends (Rule 1)."""
        (nkind, nval) = npr
        (tkind, tval) = tcr
        self.assertEqual(nkind, tkind, f"{label}: {nval} vs {tval}")
        if nkind == 'err':
            self.assertIs(nval, tval, f"{label}: differing exception types")
            return
        # Both succeeded: they must have the same shape of result.
        self.assertEqual(isinstance(nval, tuple), isinstance(tval, tuple),
                         f"{label}: one result is a tuple")
        if isinstance(nval, tuple):
            self.assertEqual(len(nval), len(tval), f"{label}: tuple length")
            for (i, (x, y)) in enumerate(zip(nval, tval)):
                self._assert_same(f"{label}[{i}]", ('ok', x), ('ok', y))
            return
        self.assertEqual(isinstance(nval, pint.Quantity),
                         isinstance(tval, pint.Quantity),
                         f"{label}: one result is a Quantity")
        if isinstance(nval, pint.Quantity):
            self.assertEqual(nval.units, tval.units, f"{label}: units")
        # The result follows the type of the input (Rule 1's proviso).
        nmag = nval.m if isinstance(nval, pint.Quantity) else nval
        tmag = tval.m if isinstance(tval, pint.Quantity) else tval
        if isinstance(nmag, (np.ndarray, np.generic)):
            self.assertTrue(
                torch.is_tensor(tmag),
                f"{label}: the tensor-backed result is not a tensor"
                f" ({type(tmag).__name__})")
        # And the values agree.
        (na, ta) = (_asarray(nmag), _asarray(tmag))
        self.assertEqual(na.shape, ta.shape, f"{label}: shape")
        self.assertTrue(
            np.allclose(na.astype(float), ta.astype(float), equal_nan=True),
            f"{label}: values differ ({na} vs {ta})")

    # Rule 1 -------------------------------------------------------------
    def test_rule1_math_unary(self):
        "Every one-argument immlib.math call agrees across the backends."
        for (name, args, kwargs, units) in UNARY_CALLS:
            with self.subTest(fn=name, args=args, kwargs=kwargs):
                fn = getattr(im, name)
                (a, b) = self._quants(units)
                self._assert_same(f"im.{name}",
                                  self._call(fn, a, *args, **kwargs),
                                  self._call(fn, b, *args, **kwargs))

    def test_rule1_math_binary(self):
        "Every two-argument immlib.math call agrees across the backends."
        for (name, units) in BINARY_CALLS:
            with self.subTest(fn=name):
                fn = getattr(im, name)
                (a, b) = self._quants(units)
                self._assert_same(f"im.{name}",
                                  self._call(fn, a, a),
                                  self._call(fn, b, b))

    def test_rule1_math_sequence(self):
        "stack/cat agree across the backends."
        for name in ('stack', 'cat', 'concatenate', 'concat'):
            for dim in (0, 1):
                with self.subTest(fn=name, dim=dim):
                    fn = getattr(im, name)
                    (a, b) = self._quants('mm')
                    self._assert_same(f"im.{name}",
                                      self._call(fn, [a, a], dim),
                                      self._call(fn, [b, b], dim))

    def test_rule1_methods(self):
        "Every Quantity method agrees across the backends."
        for units in ('mm', None):
            for (name, args, kwargs) in METHOD_CALLS:
                with self.subTest(units=units, method=name, args=args):
                    (a, b) = self._quants(units)
                    self._assert_same(
                        f"Quantity.{name}",
                        self._call(lambda q: getattr(q, name)(*args,
                                                              **kwargs), a),
                        self._call(lambda q: getattr(q, name)(*args,
                                                              **kwargs), b))

    def test_rule1_operators(self):
        "Every operator agrees across the backends."
        for units in ('mm', None):
            for (label, op) in OPERATOR_CALLS:
                with self.subTest(units=units, op=label):
                    (a, b) = self._quants(units)
                    self._assert_same(label,
                                      self._call(op, a, a),
                                      self._call(op, b, b))

    def test_rule1_setitem(self):
        "Item assignment agrees across the backends."
        cases = [('bare', 5.0), ('unit-less quantity', quant(5.0)),
                 ('mm quantity', quant(5.0, 'mm')),
                 ('cm quantity', quant(0.5, 'cm'))]
        for units in ('mm', None):
            for (label, value) in cases:
                with self.subTest(units=units, value=label):
                    def assign(q):
                        q[0, 0] = value
                        return q
                    (a, b) = self._quants(units)
                    self._assert_same(f"q[0,0] = {label}",
                                      self._call(assign, a),
                                      self._call(assign, b))

    def test_rule1_numpy_dispatch(self):
        "NumPy dispatch on a unit-less quantity stays in its own backend."
        for (label, fn) in UFUNC_CALLS:
            with self.subTest(call=label):
                (a, b) = self._quants(None)
                self._assert_same(label, self._call(fn, a),
                                  self._call(fn, b))

    # Rule 2 -------------------------------------------------------------
    def test_rule2_math(self):
        "immlib.math keeps a tensor's gradient tracking."
        skip = ('floor', 'ceil', 'round', 'any', 'all', 'eq', 'equal',
                'not_equal', 'less', 'less_equal', 'greater',
                'greater_equal')
        for (name, args, kwargs, units) in UNARY_CALLS:
            if name in skip:
                continue
            with self.subTest(fn=name, args=args, kwargs=kwargs):
                (_, b) = self._quants(units, grad=True)
                r = getattr(im, name)(b, *args, **kwargs)
                for part in (r if isinstance(r, tuple) else (r,)):
                    m = part.m if isinstance(part, pint.Quantity) else part
                    if torch.is_tensor(m) and not m.dtype.is_floating_point:
                        continue    # an index result carries no gradient
                    self.assertTrue(
                        _tracks_grad(m),
                        f"im.{name} lost gradient tracking ({type(m)})")

    def test_rule2_methods(self):
        "Quantity's methods keep a tensor's gradient tracking."
        skip = ('round', 'any', 'all', 'astype')
        for units in ('mm', None):
            for (name, args, kwargs) in METHOD_CALLS:
                if name in skip:
                    continue
                with self.subTest(units=units, method=name, args=args):
                    (_, b) = self._quants(units, grad=True)
                    r = getattr(b, name)(*args, **kwargs)
                    for part in (r if isinstance(r, tuple) else (r,)):
                        m = part.m if isinstance(part, pint.Quantity) \
                            else part
                        if torch.is_tensor(m) and \
                                not m.dtype.is_floating_point:
                            continue
                        self.assertTrue(
                            _tracks_grad(m),
                            f"Quantity.{name} lost gradient tracking")

    def test_rule2_numpy_dispatch(self):
        """NumPy dispatch keeps gradient tracking: a NumPy ufunc applied to a
        tensor would convert it (and raises outright for a tensor that
        requires grad), so immlib calls PyTorch's own function instead."""
        for (label, fn) in UFUNC_CALLS:
            if label == 'np.equal':
                continue
            with self.subTest(call=label):
                (_, b) = self._quants(None, grad=True)
                r = fn(b)
                m = r.m if isinstance(r, pint.Quantity) else r
                self.assertTrue(_tracks_grad(m),
                                f"{label} lost gradient tracking")

    def test_rule2_backward(self):
        "A gradient computed through immlib reaches the original tensor."
        for units in ('mm', None):
            with self.subTest(units=units, path='immlib.math'):
                (_, b) = self._quants(units, grad=True)
                im.sum(im.multiply(b, b)).m.backward()
                self.assertIsNotNone(b.m.grad)
            with self.subTest(units=units, path='methods'):
                (_, b) = self._quants(units, grad=True)
                (b * b).sum().m.backward()
                self.assertIsNotNone(b.m.grad)
            with self.subTest(units=units, path='numpy dispatch'):
                (_, b) = self._quants(None, grad=True)
                np.sum(np.exp(b)).m.backward()
                self.assertIsNotNone(b.m.grad)

    # The rules apply to whatever immlib.math holds ----------------------
    def test_every_public_function_is_covered(self):
        """Every public immlib.math function appears in the tables above, so
        that a function added later is not silently left untested."""
        covered = {name for (name, _, _, _) in UNARY_CALLS}
        covered |= {name for (name, _) in BINARY_CALLS}
        covered |= {'stack', 'cat', 'concatenate', 'concat'}
        # Functions tested elsewhere, or not of the shape tested here.
        covered |= {'quant', 'ilquant', 'mag', 'promote', 'to_array',
                    'to_tensor', 'matmul', 'where', 'min_result',
                    'max_result'}
        # Aliases of a covered function are covered by it.
        aliases = {'swapaxes': 'transpose', 'swapdims': 'transpose',
                   'acos': 'arccos', 'atan': 'arctan', 'atan2': 'arctan2',
                   'arcsin': 'asin', 'ne': 'not_equal',
                   'lt': 'less', 'le': 'less_equal', 'gt': 'greater',
                   'ge': 'greater_equal', 'absolute': 'abs',
                   'acos': 'arccos'}
        missing = []
        for name in im.__all__:
            if name in covered or aliases.get(name) in covered:
                continue
            missing.append(name)
        self.assertEqual(
            missing, [],
            f"these immlib.math functions are not covered by the Rule 1/2"
            f" tables in this module: {missing}")
