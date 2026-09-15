# -*- coding: utf-8 -*-
################################################################################
# immlib/test/math/test_math.py
#
# Tests of the immlib.math module.


# Dependencies #################################################################

from unittest import TestCase

class TestMath(TestCase):
    """Tests the immlib.math module."""

    # Elementwise arithmetic ###################################################
    def test_arithmetic_array(self):
        import immlib as il
        import immlib.math as im
        import numpy as np
        a = il.quant(np.array([1.0, 2.0, 3.0]), 'm')
        b = il.quant(np.array([10.0, 20.0, 30.0]), 'cm')
        r = im.add(a, b)
        self.assertEqual(str(r.units), 'meter')
        self.assertTrue(np.allclose(r.m, [1.1, 2.2, 3.3]))
        r = im.subtract(a, b)
        self.assertTrue(np.allclose(r.m, [0.9, 1.8, 2.7]))
        r = im.multiply(a, il.quant(2.0, None))
        self.assertEqual(str(r.units), 'meter')
        self.assertTrue(np.allclose(r.m, [2.0, 4.0, 6.0]))
        r = im.divide(a, il.quant(2.0, 's'))
        self.assertEqual(str(r.units), 'meter / second')
        r = im.true_divide(a, il.quant(2.0, 's'))
        self.assertEqual(str(r.units), 'meter / second')
        r = im.power(il.quant(np.array([2.0, 3.0]), 'm'), 2)
        self.assertEqual(str(r.units), 'meter ** 2')
        self.assertTrue(np.allclose(r.m, [4.0, 9.0]))
        self.assertTrue(np.allclose(im.negative(a).m, [-1, -2, -3]))
        self.assertTrue(np.allclose(im.positive(a).m, [1, 2, 3]))
        r = im.abs(il.quant(np.array([-1.0, 2.0, -3.0]), 'm'))
        self.assertEqual(str(r.units), 'meter')
        self.assertTrue(np.allclose(r.m, [1, 2, 3]))
        # Every immlib.math function also accepts raw (non-Quantity) values,
        # treating them as unit-less quantities.
        r = im.add(np.array([1.0, 2.0]), np.array([1.0, 1.0]))
        self.assertIsNone(r.units)
        self.assertTrue(np.allclose(r.m, [2.0, 3.0]))

    def test_arithmetic_tensor(self):
        import immlib as il
        import immlib.math as im
        import torch
        a = il.quant(torch.tensor([1.0, 2.0, 3.0]), 'm')
        b = il.quant(torch.tensor([10.0, 20.0, 30.0]), 'cm')
        r = im.add(a, b)
        self.assertEqual(str(r.units), 'meter')
        self.assertTrue(torch.is_tensor(r.m))
        self.assertTrue(torch.allclose(r.m, torch.tensor([1.1, 2.2, 3.3])))
        # Gradients must flow through immlib.math exactly as through the
        # equivalent raw PyTorch arithmetic.
        ta = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        qa = il.quant(ta, 'm')
        qb = il.quant(torch.tensor([1.0, 1.0, 1.0]), 'm')
        r = im.add(qa, qb)
        loss = im.sum(r).m
        loss.backward()
        self.assertTrue(torch.allclose(ta.grad, torch.tensor([1.0, 1.0, 1.0])))

    # Comparisons ###############################################################
    def test_comparisons(self):
        import immlib as il
        import immlib.math as im
        import numpy as np
        a = il.quant(np.array([1.0, 2.0, 3.0]), 'm')
        same = il.quant(np.array([1.0, 2.0, 3.0]), 'm')
        r = im.equal(a, same)
        # Comparisons return a plain bool array/tensor, never a Quantity.
        self.assertIsInstance(r, np.ndarray)
        self.assertTrue(r.all())
        self.assertTrue(im.not_equal(a, il.quant(np.array([9.0,9.0,9.0]),'m')).all())
        self.assertTrue(im.less(a, il.quant(np.array([150.0,250.0,350.0]),'cm')).all())
        self.assertTrue(im.less_equal(a, same).all())
        self.assertTrue(im.greater(il.quant(np.array([2.0]),'m'), il.quant(np.array([100.0]),'cm')).all())
        self.assertTrue(im.greater_equal(a, same).all())

    def test_comparisons_tensor(self):
        import immlib as il
        import immlib.math as im
        import torch
        a = il.quant(torch.tensor([1.0, 2.0, 3.0]), 'm')
        b = il.quant(torch.tensor([1.0, 2.0, 3.0]), 'm')
        r = im.equal(a, b)
        self.assertTrue(torch.is_tensor(r))
        self.assertTrue(bool(r.all()))

    def test_maximum_minimum_where(self):
        import immlib as il
        import immlib.math as im
        import numpy as np
        a = il.quant(np.array([1.0, 2.0, 3.0]), 'm')
        b = il.quant(np.array([10.0, 20.0, 30.0]), 'cm')
        r = im.maximum(a, b)
        self.assertEqual(str(r.units), 'meter')
        self.assertTrue(np.allclose(r.m, [1, 2, 3]))
        r = im.minimum(a, b)
        self.assertTrue(np.allclose(r.m, [0.1, 0.2, 0.3]))
        cond = np.array([True, False, True])
        r = im.where(cond, a, b)
        self.assertTrue(np.allclose(r.m, [1.0, 0.2, 3.0]))
        # Mixing a unit-less quantity with a real-unit one is an error.
        with self.assertRaises(TypeError):
            im.maximum(il.quant(1.0), il.quant(1.0, 'm'))
        # Dimensionally-incompatible real units raise, not silently combine.
        import pint
        with self.assertRaises(pint.DimensionalityError):
            im.maximum(il.quant(1.0, 'm'), il.quant(1.0, 's'))

    # Elementary functions ######################################################
    def test_sqrt(self):
        import immlib as il
        import immlib.math as im
        import numpy as np
        r = im.sqrt(il.quant(np.array([4.0, 9.0]), 'm**2'))
        self.assertEqual(str(r.units), 'meter')
        self.assertTrue(np.allclose(r.m, [2, 3]))
        r = im.sqrt(il.quant(np.array([4.0, 9.0])))
        self.assertIsNone(r.units)

    def test_unitless_functions(self):
        import immlib as il
        import immlib.math as im
        import numpy as np
        r = im.exp(il.quant(np.array([0.0, 1.0])))
        self.assertIsNone(r.units)
        self.assertTrue(np.allclose(r.m, [1.0, np.e]))
        with self.assertRaises(TypeError):
            im.exp(il.quant(1.0, 'm'))
        self.assertTrue(np.allclose(im.log(il.quant(np.array([1.0, np.e]))).m, [0, 1]))
        self.assertTrue(np.allclose(im.log10(il.quant(np.array([1.0, 100.0]))).m, [0, 2]))
        self.assertTrue(np.allclose(im.sin(il.quant(np.array([0.0, np.pi/2]))).m, [0, 1]))
        self.assertTrue(np.allclose(im.cos(il.quant(np.array([0.0]))).m, [1]))
        self.assertTrue(np.allclose(im.tan(il.quant(np.array([0.0]))).m, [0]))
        self.assertTrue(np.allclose(im.arcsin(il.quant(np.array([0.0, 1.0]))).m, [0, np.pi/2]))
        self.assertTrue(np.allclose(im.arccos(il.quant(np.array([1.0]))).m, [0]))
        self.assertTrue(np.allclose(im.arctan(il.quant(np.array([0.0]))).m, [0]))
        r = im.arctan2(il.quant(np.array([1.0])), il.quant(np.array([1.0])))
        self.assertIsNone(r.units)
        self.assertTrue(np.allclose(r.m, [np.pi/4]))

    def test_unitless_functions_tensor(self):
        import immlib as il
        import immlib.math as im
        import torch
        r = im.exp(il.quant(torch.tensor([0.0, 1.0])))
        self.assertTrue(torch.is_tensor(r.m))
        self.assertTrue(torch.allclose(r.m, torch.tensor([1.0, float(torch.e)])))
        with self.assertRaises(TypeError):
            im.exp(il.quant(torch.tensor([1.0]), 'm'))
        r = im.arcsin(il.quant(torch.tensor([0.0, 1.0])))
        self.assertTrue(torch.allclose(r.m, torch.asin(torch.tensor([0.0, 1.0]))))
        # Gradients flow through unitless elementary functions.
        t = torch.tensor([0.5], requires_grad=True)
        q = il.quant(t)
        im.exp(q).m.backward()
        self.assertTrue(torch.allclose(t.grad, torch.exp(torch.tensor([0.5]))))

    def test_floor_ceil_round(self):
        import immlib as il
        import immlib.math as im
        import numpy as np
        r = im.floor(il.quant(np.array([1.7, 2.3]), 'm'))
        self.assertEqual(str(r.units), 'meter')
        self.assertTrue(np.allclose(r.m, [1, 2]))
        r = im.ceil(il.quant(np.array([1.2, 2.8]), 'm'))
        self.assertTrue(np.allclose(r.m, [2, 3]))
        r = im.round(il.quant(np.array([1.234, 2.345]), 'm'), 1)
        self.assertTrue(np.allclose(r.m, [1.2, 2.3]))

    # Reductions ################################################################
    def test_reductions_array(self):
        import immlib as il
        import immlib.math as im
        import numpy as np
        q = il.quant(np.array([[1.0, 2.0], [3.0, 4.0]]), 'm')
        r = im.sum(q)
        self.assertEqual(r.m, 10.0)
        self.assertEqual(str(r.units), 'meter')
        self.assertTrue(np.allclose(im.sum(q, axis=0).m, [4, 6]))
        self.assertEqual(im.sum(q, axis=0, keepdims=True).m.shape, (1, 2))
        self.assertEqual(im.mean(q).m, 2.5)
        self.assertTrue(np.allclose(im.min(q, axis=1).m, [1, 3]))
        self.assertTrue(np.allclose(im.max(q, axis=1).m, [2, 4]))
        self.assertTrue(bool(im.any(il.quant(np.array([0, 0, 1])))))
        self.assertFalse(bool(im.all(il.quant(np.array([1, 1, 0])))))
        self.assertTrue(np.isclose(im.std(q, ddof=0).m, np.std(q.m)))
        self.assertEqual(str(im.var(q).units), 'meter ** 2')
        self.assertTrue(np.isclose(im.var(q).m, np.var(q.m)))
        r = im.prod(il.quant(np.array([2.0, 3.0, 4.0]), 'm'))
        self.assertEqual(str(r.units), 'meter ** 3')
        self.assertEqual(r.m, 24.0)
        r = im.prod(il.quant(np.array([[2.0, 3.0], [4.0, 5.0]]), 'm'), axis=0)
        self.assertEqual(str(r.units), 'meter ** 2')
        self.assertTrue(np.allclose(r.m, [8, 15]))

    def test_reductions_tensor(self):
        import immlib as il
        import immlib.math as im
        import torch
        q = il.quant(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), 'm')
        r = im.sum(q)
        self.assertTrue(torch.is_tensor(r.m))
        self.assertEqual(float(r.m), 10.0)
        self.assertTrue(torch.allclose(im.sum(q, axis=0).m, torch.tensor([4.0, 6.0])))
        self.assertEqual(tuple(im.sum(q, axis=0, keepdims=True).m.shape), (1, 2))
        self.assertEqual(tuple(im.sum(q, keepdims=True).m.shape), (1, 1))
        # min/max must return only values (matching numpy.min/max), never
        # torch.min/torch.max's (values, indices) tuple.
        r = im.min(q, axis=1)
        self.assertTrue(torch.is_tensor(r.m))
        self.assertTrue(torch.allclose(r.m, torch.tensor([1.0, 3.0])))
        r = im.max(q, axis=1)
        self.assertTrue(torch.allclose(r.m, torch.tensor([2.0, 4.0])))
        self.assertTrue(bool(im.any(il.quant(torch.tensor([0, 0, 1])))))
        self.assertFalse(bool(im.all(il.quant(torch.tensor([1, 1, 0])))))
        # std/var default to ddof=0 (numpy's population default), not
        # PyTorch's own default of a Bessel-corrected (ddof=1) sample stat.
        r = im.std(q)
        self.assertTrue(torch.allclose(r.m, torch.std(q.m, correction=0)))
        r = im.var(q, ddof=1)
        self.assertTrue(torch.allclose(r.m, torch.var(q.m, correction=1)))
        self.assertEqual(str(r.units), 'meter ** 2')
        # prod supports a single axis for tensors, but not a tuple of axes.
        r = im.prod(il.quant(torch.tensor([2.0, 3.0, 4.0]), 'm'))
        self.assertEqual(str(r.units), 'meter ** 3')
        self.assertEqual(float(r.m), 24.0)
        with self.assertRaises(TypeError):
            im.prod(il.quant(torch.tensor([[2.0,3.0],[4.0,5.0]]), 'm'), axis=(0,1))
        # Gradients flow through reductions.
        t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        im.sum(il.quant(t)).m.backward()
        self.assertTrue(torch.allclose(t.grad, torch.ones(3)))

    # Shape / combination #######################################################
    def test_shape_combination_array(self):
        import immlib as il
        import immlib.math as im
        import numpy as np
        q = il.quant(np.arange(6.0).reshape(2, 3), 'm')
        r = im.reshape(q, (3, 2))
        self.assertEqual(r.m.shape, (3, 2))
        self.assertEqual(str(r.units), 'meter')
        r = im.transpose(q)
        self.assertTrue(np.array_equal(r.m, q.m.T))
        r = im.transpose(q, axes=(1, 0))
        self.assertTrue(np.array_equal(r.m, q.m.T))
        q4 = il.quant(np.array([[[1.0, 2.0]]]), 'm')
        r = im.squeeze(q4)
        self.assertEqual(r.m.shape, (2,))
        r = im.stack([
            il.quant(np.array([1.0, 2.0]), 'm'),
            il.quant(np.array([100.0, 200.0]), 'cm')])
        self.assertEqual(str(r.units), 'meter')
        self.assertTrue(np.allclose(r.m, [[1, 2], [1, 2]]))
        r = im.concatenate([
            il.quant(np.array([1.0, 2.0]), 'm'),
            il.quant(np.array([100.0, 200.0]), 'cm')])
        self.assertTrue(np.allclose(r.m, [1, 2, 1, 2]))
        with self.assertRaises(TypeError):
            im.stack([il.quant(np.array([1.0])), il.quant(np.array([1.0]), 'm')])

    def test_shape_combination_tensor(self):
        import immlib as il
        import immlib.math as im
        import torch
        q = il.quant(torch.arange(6.0).reshape(2, 3), 'm')
        r = im.transpose(q)
        # transpose must match numpy.transpose's full-axis-reversal default,
        # not torch.transpose's single-pair-swap-only behavior.
        self.assertEqual(tuple(r.m.shape), (3, 2))
        self.assertTrue(torch.equal(r.m, q.m.permute(1, 0)))
        r = im.reshape(q, (3, 2))
        self.assertEqual(tuple(r.m.shape), (3, 2))
        r = im.squeeze(il.quant(torch.tensor([[[1.0, 2.0]]]), 'm'))
        self.assertEqual(tuple(r.m.shape), (2,))
        # Mixing an array-backed and a tensor-backed quantity in stack must
        # promote to tensor-backed, not silently densify/convert away from
        # the tensor.
        import numpy as np
        r = im.stack([
            il.quant(np.array([1.0, 2.0]), 'm'),
            il.quant(torch.tensor([100.0, 200.0]), 'cm')])
        self.assertTrue(torch.is_tensor(r.m))
        # The NumPy array (float64) and the tensor (float32) legitimately
        # promote to float64 (PyTorch's own type-promotion rule for
        # stack/cat, per the "PyTorch controls dtype promotion" design
        # invariant), so the expected tensor must match r.m's dtype rather
        # than assume either input's original dtype.
        self.assertTrue(torch.allclose(
            r.m, torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=r.m.dtype)))
        r = im.concatenate([
            il.quant(np.array([1.0, 2.0]), 'm'),
            il.quant(torch.tensor([100.0, 200.0]), 'cm')])
        self.assertTrue(torch.is_tensor(r.m))
        self.assertTrue(torch.allclose(
            r.m, torch.tensor([1.0, 2.0, 1.0, 2.0], dtype=r.m.dtype)))

    # Linear algebra #############################################################
    def test_matmul(self):
        import immlib as il
        import immlib.math as im
        import numpy as np
        r = im.matmul(
            il.quant(np.eye(2), 'm'), il.quant(np.array([[1.0], [2.0]]), 's'))
        self.assertEqual(str(r.units), 'meter * second')
        self.assertTrue(np.allclose(r.m, [[1], [2]]))
        # immlib.math.dot is intentionally not provided (np.dot and
        # torch.dot have materially different N-D semantics); matmul is the
        # supported replacement.
        self.assertFalse(hasattr(im, 'dot'))

    def test_matmul_tensor_grad(self):
        import immlib as il
        import immlib.math as im
        import torch
        ta = torch.eye(2, requires_grad=True)
        qb = il.quant(torch.tensor([[1.0], [2.0]]), 's')
        r = im.matmul(il.quant(ta, 'm'), qb)
        self.assertTrue(torch.is_tensor(r.m))
        r.m.sum().backward()
        self.assertIsNotNone(ta.grad)
