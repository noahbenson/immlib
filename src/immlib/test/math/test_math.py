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
        r = im.pow(il.quant(np.array([2.0, 3.0]), 'm'), 2)
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
        r = im.eq(a, same)
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
        r = im.eq(a, b)
        self.assertTrue(torch.is_tensor(r))
        self.assertTrue(bool(r.all()))
        # Incompatible units, or a unit-less value compared with a quantity
        # that has real dimensions, give an all-false result on both
        # backends.
        import numpy as np
        s = il.quant(torch.tensor([1.0, 2.0, 3.0]), 's')
        n = il.quant(torch.tensor([1.0, 2.0, 3.0]))
        for other in (s, n):
            r = im.eq(a, other)
            self.assertTrue(torch.equal(r, torch.zeros(3, dtype=torch.bool)))
            r = im.eq(other, a)
            self.assertTrue(torch.equal(r, torch.zeros(3, dtype=torch.bool)))
            r = im.not_equal(a, other)
            self.assertTrue(bool(r.all()))
        anp = il.quant(np.array([1.0, 2.0, 3.0]), 'm')
        for other in (il.quant(np.array([1.0, 2.0, 3.0]), 's'),
                      il.quant(np.array([1.0, 2.0, 3.0]))):
            r = im.eq(anp, other)
            self.assertTrue(np.array_equal(r, [False, False, False]))

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
        # A unit-less quantity is treated like a bare (dimensionless) value,
        # as numpy.maximum(x, quant(y, 'm')) treats x.
        import pint
        with self.assertRaises(pint.DimensionalityError):
            im.maximum(il.quant(1.0), il.quant(1.0, 'm'))
        with self.assertRaises(pint.DimensionalityError):
            np.maximum(il.quant(1.0), il.quant(1.0, 'm'))
        with self.assertRaises(pint.DimensionalityError):
            im.where(cond, il.quant(np.ones(3)), a)
        r = im.maximum(il.quant(np.array([1.0, 2000.0])),
                       il.quant(np.array([3.0, 1.0]), 'm/km'))
        self.assertEqual(r.units, il.unit('m/km'))
        self.assertTrue(np.allclose(r.m, [1000.0, 2000000.0]))
        r = im.minimum(il.quant(np.array([3.0, 1.0]), 'm/km'),
                       il.quant(np.array([1.0, 2000.0])))
        self.assertEqual(r.units, il.unit('m/km'))
        self.assertTrue(np.allclose(r.m, [3.0, 1.0]))
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
        self.assertTrue(np.allclose(im.min(q, dim=1).values.m, [1, 3]))
        self.assertTrue(np.allclose(im.max(q, dim=1).values.m, [2, 4]))
        self.assertTrue(bool(im.any(il.quant(np.array([0, 0, 1])))))
        self.assertFalse(bool(im.all(il.quant(np.array([1, 1, 0])))))
        self.assertTrue(np.isclose(im.std(q, correction=0).m, np.std(q.m)))
        self.assertEqual(str(im.var(q).units), 'meter ** 2')
        # correction defaults to 1 (torch's default), not numpy's ddof=0.
        self.assertTrue(np.isclose(im.var(q).m, np.var(q.m, ddof=1)))
        self.assertTrue(np.isclose(im.var(q, correction=0).m, np.var(q.m)))
        # min/max return (values, indices), as torch.min/torch.max do;
        # amin/amax return the values alone.
        r = im.min(q, dim=1)
        self.assertTrue(np.allclose(r.values.m, [1, 3]))
        self.assertTrue(np.array_equal(r.indices, [0, 0]))
        self.assertTrue(np.allclose(im.amin(q, dim=1).m, [1, 3]))
        self.assertTrue(np.allclose(im.amax(q, dim=1).m, [2, 4]))
        # A whole-quantity min/max is just the value.
        self.assertEqual(im.min(q).m, 1.0)
        self.assertEqual(im.max(q).m, 4.0)
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
        # min/max return torch's (values, indices) tuple for both backends.
        r = im.min(q, dim=1)
        self.assertTrue(torch.is_tensor(r.values.m))
        self.assertTrue(torch.allclose(r.values.m, torch.tensor([1.0, 3.0])))
        self.assertTrue(torch.equal(r.indices, torch.tensor([0, 0])))
        r = im.max(q, dim=1)
        self.assertTrue(torch.allclose(r.values.m, torch.tensor([2.0, 4.0])))
        self.assertTrue(torch.is_tensor(im.amin(q, dim=1).m))
        self.assertTrue(bool(im.any(il.quant(torch.tensor([0, 0, 1])))))
        self.assertFalse(bool(im.all(il.quant(torch.tensor([1, 1, 0])))))
        # std/var default to correction=1, PyTorch's own default (a
        # Bessel-corrected sample statistic), not numpy's ddof=0.
        r = im.std(q)
        self.assertTrue(torch.allclose(r.m, torch.std(q.m, correction=1)))
        self.assertTrue(
            torch.allclose(im.std(q, correction=0).m,
                           torch.std(q.m, correction=0)))
        r = im.var(q, correction=1)
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
        r = im.permute(q)
        self.assertTrue(np.array_equal(r.m, q.m.T))
        r = im.permute(q, (1, 0))
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
        import pint
        with self.assertRaises(pint.DimensionalityError):
            im.stack([il.quant(np.array([1.0])), il.quant(np.array([1.0]), 'm')])
        # The first real units win; unit-less elements are dimensionless.
        r = im.concatenate([il.quant(np.array([1.0])),
                            il.quant(np.array([2.0]), 'm/km')])
        self.assertEqual(r.units, il.unit('m/km'))
        self.assertTrue(np.allclose(r.m, [1000.0, 2.0]))

    def test_shape_combination_tensor(self):
        import immlib as il
        import immlib.math as im
        import torch
        q = il.quant(torch.arange(6.0).reshape(2, 3), 'm')
        r = im.permute(q)
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
        # immlib.math.dot means what torch.dot means: a 1-D inner product,
        # and an error for anything else. numpy.dot's wider meaning, matrix
        # multiplication with broadcasting, is matmul.
        v = il.quant(np.array([1.0, 2.0, 3.0]), 'm')
        r = im.dot(v, v)
        self.assertEqual(str(r.units), 'meter ** 2')
        self.assertEqual(float(r.m), 14.0)
        with self.assertRaises(ValueError):
            im.dot(il.quant(np.eye(2), 'm'), il.quant(np.eye(2), 'm'))
        with self.assertRaises(ValueError):
            im.dot(v, il.quant(np.array([1.0, 2.0]), 'm'))

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

    def test_sparse_reductions(self):
        """Reductions of SciPy sparse arrays return dense NumPy results that
        match the reductions of the equivalent dense arrays."""
        import immlib as il
        import immlib.math as im
        import numpy as np
        import scipy.sparse as sps
        dense = np.array([[1.0, 0.0, 2.0], [3.0, 4.0, 5.0], [0.0, 0.0, 0.0]])
        reductions = [('sum', {}), ('mean', {}), ('amin', {}), ('amax', {}),
                      ('any', {}), ('all', {}), ('std', {}), ('var', {}),
                      ('std', {'correction': 1}), ('prod', {})]
        for fmt in (sps.csr_matrix, sps.csr_array, sps.coo_array):
            sp = il.quant(fmt(dense), 'm')
            dn = il.quant(dense, 'm')
            for (name, kw) in reductions:
                # prod reduces one dimension at a time, as torch.prod does.
                axes = (None, 0, 1, -1) if name == 'prod' else \
                       (None, 0, 1, -1, (0, 1))
                for axis in axes:
                    for keepdims in (False, True):
                        with self.subTest(fmt=fmt.__name__, fn=name,
                                          axis=axis, keepdims=keepdims):
                            fn = getattr(im, name)
                            r = fn(sp, axis=axis, keepdims=keepdims, **kw)
                            e = fn(dn, axis=axis, keepdims=keepdims, **kw)
                            if name in ('any', 'all'):
                                (rm, em) = (r, e)
                            else:
                                self.assertEqual(r.units, e.units)
                                (rm, em) = (r.m, e.m)
                            self.assertFalse(sps.issparse(rm))
                            self.assertNotIsInstance(rm, np.matrix)
                            self.assertIsInstance(rm, (np.ndarray, np.generic))
                            self.assertEqual(np.shape(rm), np.shape(em))
                            self.assertTrue(np.allclose(rm, em))
        # Sparse PyTorch reductions are also returned dense.
        import torch
        t = torch.tensor(dense).to_sparse()
        r = im.sum(il.quant(t, 'm'), axis=0)
        self.assertEqual(r.m.layout, torch.strided)
        self.assertTrue(torch.allclose(r.m, torch.tensor(dense.sum(axis=0))))

    def test_sparse_elementwise(self):
        """Operations that preserve sparsity keep SciPy sparse arrays sparse;
        others raise TypeError instead of densifying."""
        import immlib as il
        import immlib.math as im
        import numpy as np
        import scipy.sparse as sps
        dense = np.array([[1.26, 0.0, 2.0], [3.0, 4.0, 0.0]])
        sp = sps.csr_array(dense)
        q = il.quant(sp, 'm')
        for r in (im.abs(q), im.negative(q), im.add(q, q), im.multiply(q, q),
                  im.sqrt(q), im.floor(q), im.ceil(q), im.round(q, 1),
                  im.maximum(q, q), im.minimum(q, q),
                  im.sin(sp), im.tan(sp), im.arcsin(sp / 10), im.arctan(sp),
                  im.reshape(q, (3, 2)), im.permute(q),
                  im.concatenate([q, q]), im.concatenate([q, q], axis=1)):
            self.assertTrue(sps.issparse(r.m))
        self.assertTrue(np.allclose(im.round(q, 1).m.toarray(),
                                    np.round(dense, 1)))
        self.assertTrue(np.allclose(im.floor(q).m.toarray(), np.floor(dense)))
        self.assertEqual(im.sqrt(q).units, il.unit('m') ** 0.5)
        self.assertEqual(im.permute(q).m.shape, (3, 2))
        r = im.concatenate([q, il.quant(dense, 'cm')])
        self.assertTrue(np.allclose(r.m.toarray(),
                                    np.vstack([dense, dense / 100])))
        r = im.matmul(q, il.quant(sp.T, 's'))
        self.assertTrue(sps.issparse(r.m))
        self.assertTrue(np.allclose(r.m.toarray(), dense @ dense.T))
        self.assertEqual(r.units, il.unit('m*s'))
        for f in (lambda: im.exp(sp), lambda: im.cos(sp), lambda: im.log(sp),
                  lambda: im.log10(sp), lambda: im.arccos(sp),
                  lambda: im.where(sp, sp, sp), lambda: im.squeeze(sp),
                  lambda: im.stack([sp, sp])):
            with self.assertRaises(TypeError):
                f()

    def test_helpers_and_as_input_type(self):
        """immlib.math re-exports quant, mag, promote, to_array, to_tensor,
        and Quantity.as_input_type matches the quantity-ness of inputs."""
        import immlib as il
        import immlib.math as im
        import numpy as np
        import torch
        for name in ('quant', 'mag', 'promote', 'to_array', 'to_tensor'):
            self.assertIs(getattr(im, name), getattr(il, name))
            self.assertIn(name, im.__all__)
        def example(a, b):
            qa = im.quant(a, 'mm')
            qb = im.quant(b, 'mm')
            result = im.sqrt((qa + 1 * qa.u) * (qb - 1 * qb.u))
            return result.as_input_type(a, b)
        r = example(3.0, 5.0)
        self.assertIsInstance(r, np.ndarray)
        self.assertEqual(float(r), 4.0)
        r = example(np.array([3.0]), [5.0])
        self.assertIsInstance(r, np.ndarray)
        r = example(torch.tensor([3.0]), torch.tensor([5.0]))
        self.assertTrue(torch.is_tensor(r))
        self.assertTrue(torch.allclose(r, torch.tensor([4.0])))
        r = example(il.quant(0.3, 'cm'), 5.0)
        self.assertIsInstance(r, il.Quantity)
        self.assertEqual(str(r.units), 'millimeter')
        self.assertAlmostEqual(float(r.m), 4.0)
        # Plain pint quantities and unit-less quantities count as quantities.
        import pint
        q = il.quant([1.0, 2.0], 'm')
        self.assertIs(q.as_input_type(1, pint.UnitRegistry().Quantity(1)), q)
        self.assertIs(q.as_input_type(np.ones(2), il.quant(1.0)), q)
        self.assertIs(q.as_input_type(), q.m)
        self.assertIs(q.as_input_type(np.ones(2), 5), q.m)

    def test_order_statistics(self):
        """sort, argsort, median, quantile, percentile, ptp and average."""
        import immlib as il
        import immlib.math as im
        import numpy as np
        a = il.quant(np.array([[3.0, 1.0, 2.0], [6.0, 5.0, 4.0]]), 'm')
        v = il.quant(np.array([1.0, 2.0, 3.0, 4.0]), 'm')
        # sort returns (values, indices), as torch.sort does.
        r = im.sort(a, 1)
        self.assertTrue(np.allclose(r.values.m, [[1, 2, 3], [4, 5, 6]]))
        self.assertTrue(np.array_equal(r.indices, [[1, 2, 0], [2, 1, 0]]))
        self.assertEqual(r.values.units, a.units)
        r = im.sort(a, 1, descending=True)
        self.assertTrue(np.allclose(r.values.m, [[3, 2, 1], [6, 5, 4]]))
        self.assertTrue(np.array_equal(im.argsort(a, 1), [[1, 2, 0],
                                                          [2, 1, 0]]))
        # argmin/argmax give a flat index when no dimension is given.
        self.assertEqual(int(im.argmin(a)), 1)
        self.assertTrue(np.array_equal(im.argmax(a, 1), [0, 0]))
        # The median is the lower of the two middle values (torch.median's
        # rule), not numpy.median's interpolated 2.5.
        self.assertEqual(float(im.median(v).m), 2.0)
        self.assertEqual(im.median(v).units, v.units)
        r = im.median(a, 1)
        self.assertTrue(np.allclose(r.values.m, [2.0, 5.0]))
        self.assertTrue(np.array_equal(r.indices, [2, 1]))
        # quantile and percentile are the same scale apart.
        self.assertTrue(np.isclose(float(im.quantile(v, 0.25).m), 1.75))
        self.assertTrue(np.isclose(float(im.percentile(v, 25).m), 1.75))
        self.assertEqual(im.quantile(v, 0.25).units, v.units)
        # ptp is the range; average is the weighted mean.
        self.assertEqual(float(im.ptp(a).m), 5.0)
        self.assertTrue(np.allclose(im.ptp(a, 1).m, [2.0, 2.0]))
        self.assertEqual(im.ptp(a).units, a.units)
        self.assertEqual(float(im.average(a).m), 3.5)
        w = np.array([1.0, 2.0, 3.0])
        self.assertTrue(
            np.allclose(im.average(a, 1, weights=w).m,
                        np.average(a.m, axis=1, weights=w)))
        # The weights must be unit-less.
        with self.assertRaises(TypeError):
            im.average(a, 1, weights=il.quant(w, 's'))

    def test_set_operations(self):
        """unique and the set operations, including their units."""
        import immlib as il
        import immlib.math as im
        import numpy as np
        a = il.quant(np.array([3.0, 1.0, 2.0, 1.0]), 'm')
        b = il.quant(np.array([200.0, 400.0]), 'cm')
        self.assertTrue(np.allclose(im.unique(a).m, [1.0, 2.0, 3.0]))
        self.assertEqual(im.unique(a).units, a.units)
        (vals, counts) = im.unique(a, return_counts=True)
        self.assertTrue(np.array_equal(counts, [2, 1, 1]))
        self.assertFalse(isinstance(counts, il.Quantity))
        # The second argument is converted into the first's units.
        self.assertTrue(np.allclose(im.union1d(a, b).m, [1, 2, 3, 4]))
        self.assertEqual(im.union1d(a, b).units, a.units)
        self.assertTrue(np.allclose(im.intersect1d(a, b).m, [2.0]))
        self.assertTrue(np.allclose(im.setdiff1d(a, b).m, [1.0, 3.0]))
        self.assertTrue(np.allclose(im.setxor1d(a, b).m, [1.0, 3.0, 4.0]))
        self.assertTrue(np.array_equal(im.isin(a, b),
                                       [False, False, True, False]))
        # Incompatible units are an error, as in maximum.
        import pint
        with self.assertRaises(pint.DimensionalityError):
            im.union1d(a, il.quant(np.array([1.0]), 's'))

    def test_indexing_functions(self):
        """gather, index_select, take, masked_select, flip, roll,
        repeat_interleave and tile."""
        import immlib as il
        import immlib.math as im
        import numpy as np
        a = il.quant(np.array([[3.0, 1.0, 2.0], [6.0, 5.0, 4.0]]), 'm')
        self.assertTrue(
            np.allclose(im.gather(a, 1, [[0, 2], [1, 0]]).m,
                        [[3.0, 2.0], [5.0, 6.0]]))
        self.assertTrue(
            np.allclose(im.index_select(a, 1, [0, 2]).m,
                        [[3.0, 2.0], [6.0, 4.0]]))
        self.assertTrue(np.allclose(im.take(a, [0, 4]).m, [3.0, 5.0]))
        mask = im.greater(a, il.quant(2.0, 'm'))
        self.assertTrue(
            np.allclose(im.masked_select(a, mask).m, [3.0, 6.0, 5.0, 4.0]))
        self.assertTrue(
            np.allclose(im.flip(a, 1).m, [[2.0, 1.0, 3.0], [4.0, 5.0, 6.0]]))
        self.assertTrue(
            np.allclose(im.roll(a, 1, 1).m, [[2.0, 3.0, 1.0],
                                             [4.0, 6.0, 5.0]]))
        self.assertTrue(
            np.allclose(im.repeat_interleave(a, 2, 1).m,
                        np.repeat(a.m, 2, axis=1)))
        self.assertTrue(np.allclose(im.tile(a, (1, 2)).m, np.tile(a.m, (1, 2))))
        # Units are preserved throughout, and an index must be unit-less.
        for r in (im.gather(a, 1, [[0], [1]]), im.take(a, [0]),
                  im.flip(a, 1), im.tile(a, (1, 2))):
            self.assertEqual(r.units, a.units)
        with self.assertRaises(TypeError):
            im.take(a, il.quant(np.array([0]), 'm'))

    def test_predicates_and_bounds(self):
        """nonzero, isnan/isinf/isfinite, and clamp/clip."""
        import immlib as il
        import immlib.math as im
        import numpy as np
        a = il.quant(np.array([[3.0, 0.0, 2.0], [np.nan, np.inf, 4.0]]), 'm')
        # nonzero gives torch's (n, ndim) rows by default and numpy's
        # tuple-of-indices form with as_tuple.
        r = im.nonzero(il.quant(np.array([[1.0, 0.0], [0.0, 2.0]])))
        self.assertTrue(np.array_equal(r, [[0, 0], [1, 1]]))
        r = im.nonzero(il.quant(np.array([[1.0, 0.0], [0.0, 2.0]])),
                       as_tuple=True)
        self.assertIsInstance(r, tuple)
        self.assertTrue(np.array_equal(r[0], [0, 1]))
        # The predicates ignore units and return plain bool arrays.
        self.assertTrue(np.array_equal(im.isnan(a),
                                       [[False, False, False],
                                        [True, False, False]]))
        self.assertTrue(np.array_equal(im.isinf(a),
                                       [[False, False, False],
                                        [False, True, False]]))
        self.assertTrue(np.array_equal(im.isfinite(a),
                                       [[True, True, True],
                                        [False, False, True]]))
        self.assertNotIsInstance(im.isnan(a), il.Quantity)
        # clamp bounds are unit-aligned, and clip is its alias.
        b = il.quant(np.array([1.0, 5.0, 9.0]), 'm')
        r = im.clamp(b, il.quant(200.0, 'cm'), il.quant(800.0, 'cm'))
        self.assertEqual(r.units, b.units)
        self.assertTrue(np.allclose(r.m, [2.0, 5.0, 8.0]))
        self.assertTrue(np.allclose(im.clip(b, il.quant(2.0, 'm')).m,
                                    [2.0, 5.0, 9.0]))
        self.assertIs(im.clip, im.clamp)
        # At least one bound is required, as torch.clamp requires.
        with self.assertRaises(ValueError):
            im.clamp(b)

    def test_pad(self):
        """pad follows torch.nn.functional.pad, including its argument
        order and its rank restriction for the non-constant modes."""
        import immlib as il
        import immlib.math as im
        import numpy as np
        a = il.quant(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), 'm')
        # The amounts are given for the LAST dimension first, as in torch.
        r = im.pad(a, (1, 0))
        self.assertEqual(r.m.shape, (2, 4))
        self.assertTrue(np.allclose(r.m[:, 0], [0.0, 0.0]))
        self.assertEqual(r.units, a.units)
        r = im.pad(a, (1, 0, 2, 0))
        self.assertEqual(r.m.shape, (4, 4))
        # A bare fill value is in a's own units.
        r = im.pad(a, (1, 0), value=9)
        self.assertTrue(np.allclose(r.m[:, 0], [9.0, 9.0]))
        r = im.pad(a, (1, 0), value=il.quant(900.0, 'cm'))
        self.assertTrue(np.allclose(r.m[:, 0], [9.0, 9.0]))
        # torch's mode names, including the two numpy spells differently.
        for (mode, first) in (('reflect', 2.0), ('replicate', 1.0),
                              ('circular', 3.0)):
            r = im.pad(a, (1, 0), mode=mode)
            self.assertTrue(np.isclose(r.m[0, 0], first), mode)
        with self.assertRaises(ValueError):
            im.pad(a, (1, 0), mode='edge')     # numpy's name for replicate
        # The non-constant modes pad n dimensions of an (n+1)- or
        # (n+2)-dimensional argument, as torch requires.
        with self.assertRaises(ValueError):
            im.pad(a, (1, 1, 1, 1), mode='reflect')

    def test_split_and_chunk(self):
        """split takes a size and chunk a count, as in torch; chunk's sizes
        are torch's, which differ from numpy.array_split's."""
        import immlib as il
        import immlib.math as im
        import numpy as np
        a = il.quant(np.arange(10.0), 'm')
        # split's integer is the size of each piece.
        pieces = im.split(a, 3)
        self.assertEqual([len(p.m) for p in pieces], [3, 3, 3, 1])
        self.assertEqual(pieces[0].units, a.units)
        self.assertTrue(np.allclose(pieces[0].m, [0.0, 1.0, 2.0]))
        # A sequence gives the sizes individually, and must sum correctly.
        self.assertEqual([len(p.m) for p in im.split(a, [2, 8])], [2, 8])
        with self.assertRaises(ValueError):
            im.split(a, [2, 3])
        # chunk's integer is a maximum number of pieces, each of size
        # ceil(n / chunks); numpy.array_split would give [3, 3, 2, 2].
        self.assertEqual([len(p.m) for p in im.chunk(a, 4)], [3, 3, 3, 1])
        self.assertEqual(
            [len(p.m) for p in im.chunk(a, 3)],
            [len(p) for p in np.split(np.arange(10.0), [4, 8])])
        with self.assertRaises(ValueError):
            im.chunk(a, 0)


    # Rearrangement and linear algebra ########################################
    def test_movedim(self):
        import immlib as il
        import immlib.math as im
        import numpy as np
        import torch
        import scipy.sparse as sps
        a = il.quant(np.arange(6.0).reshape(2, 3), 'm')
        r = im.movedim(a, 0, 1)
        self.assertTrue(np.array_equal(r.m, np.moveaxis(a.m, 0, 1)))
        self.assertEqual(str(r.units), 'meter')
        # moveaxis is an alias of movedim, as numpy spells it.
        self.assertIs(im.moveaxis, im.movedim)
        # Negative dimensions and sequences of dimensions are accepted.
        self.assertTrue(
            np.array_equal(im.movedim(a, -1, 0).m, np.moveaxis(a.m, -1, 0)))
        # A tensor argument keeps its backend.
        t = il.quant(torch.arange(6.0).reshape(2, 3), 'm')
        self.assertIsInstance(im.movedim(t, 0, 1).m, torch.Tensor)
        # Sparse arguments are not supported.
        with self.assertRaises(TypeError):
            im.movedim(il.quant(sps.csr_matrix(np.eye(3)), None), 0, 1)

    def test_linalg(self):
        import immlib as il
        import immlib.math as im
        import numpy as np
        import torch
        import pint
        import scipy.sparse as sps
        A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        # svd: reconstruction, shapes, and the full/false forms.
        (U, S, Vh) = im.svd(A)
        self.assertTrue(np.allclose(U[:, :len(S)] @ np.diag(S) @ Vh, A))
        self.assertEqual((U.shape, Vh.shape), ((3, 3), (2, 2)))
        (U, S, Vh) = im.svd(A, full_matrices=False)
        self.assertTrue(np.allclose(U @ np.diag(S) @ Vh, A))
        self.assertEqual(U.shape, (3, 2))
        with self.assertRaises(TypeError):
            im.svd(A, driver='gesvd')       # driver is a tensor-only option
        # The results are plain arrays/tensors, not quantities.
        self.assertFalse(isinstance(U, pint.Quantity))
        # pinv agrees with each library's own and is sign-independent.
        self.assertTrue(np.allclose(im.pinv(A), np.linalg.pinv(A, rcond=None)))
        self.assertTrue(
            torch.allclose(im.pinv(torch.tensor(A)),
                           torch.linalg.pinv(torch.tensor(A))))
        # matrix_rank counts tolerance-small singular values as zero.
        D = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]])
        self.assertEqual(int(im.matrix_rank(D)), 1)
        self.assertEqual(int(im.matrix_rank(torch.tensor(D))), 1)
        self.assertFalse(isinstance(im.matrix_rank(D), pint.Quantity))
        self.assertEqual(int(im.matrix_rank(D, rtol=2.0)), 0)
        self.assertEqual(int(im.matrix_rank(D, hermitian=True)), 1)
        # Batched arguments work.
        Ab = np.stack([A, 2.0 * A])
        (Ub, Sb, Vhb) = im.svd(Ab, full_matrices=False)
        self.assertEqual((Ub.shape, Sb.shape, Vhb.shape),
                         ((2, 3, 2), (2, 2), (2, 2, 2)))
        self.assertEqual(im.pinv(Ab).shape, (2, 2, 3))
        self.assertTrue(np.array_equal(im.matrix_rank(Ab), [2, 2]))
        # The argument must be unit-less, at least 2-dimensional, and dense.
        with self.assertRaises(TypeError):
            im.svd(il.quant(A, 'm'))
        with self.assertRaises(ValueError):
            im.pinv(np.array([1.0, 2.0, 3.0]))
        with self.assertRaises(TypeError):
            im.matrix_rank(il.quant(sps.csr_matrix(np.eye(3)), None))
        # Tensor arguments keep their gradient tracking.
        Tg = torch.tensor(A, requires_grad=True)
        im.svd(Tg, full_matrices=False)[1].sum().backward()
        self.assertIsNotNone(Tg.grad)
        Tg = torch.tensor(A, requires_grad=True)
        im.pinv(Tg).sum().backward()
        self.assertIsNotNone(Tg.grad)

    def test_einsum(self):
        import immlib as il
        import immlib.math as im
        import numpy as np
        import torch
        a = np.arange(6.0).reshape(2, 3)
        b = np.arange(6.0).reshape(3, 2)
        self.assertTrue(np.allclose(im.einsum('ij,jk->ik', a, b), a @ b))
        # The backend follows the arguments, as elsewhere in immlib.math.
        r = im.einsum('ij,jk->ik', a, torch.tensor(b))
        self.assertIsInstance(r, torch.Tensor)
        self.assertTrue(torch.allclose(r, torch.tensor(a @ b)))
        # All-array operands give an array, even as quantities.
        r = im.einsum('ij,jk->ik', il.quant(a), il.quant(b))
        self.assertIsInstance(r, np.ndarray)
        # Every operand must be unit-less.
        with self.assertRaises(TypeError):
            im.einsum('i->', il.quant(a, 'm'))
        # A tensor argument keeps its gradient tracking.
        Tg = torch.tensor(a, requires_grad=True)
        im.einsum('ij->j', Tg).sum().backward()
        self.assertIsNotNone(Tg.grad)


    def test_lstsq(self):
        import immlib as il
        import immlib.math as im
        import numpy as np
        import torch
        A = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        B = np.array([[1.0], [2.0], [2.0]])
        nx = np.linalg.lstsq(A, B, rcond=None)[0]
        # By torch's conventions, the default returns the solution and the
        # rank but empty residuals and singular values.
        (x, res, rank, sv) = im.lstsq(A, B)
        self.assertTrue(np.allclose(x, nx))
        self.assertEqual(res.shape, (0,))
        self.assertEqual(sv.shape, (0,))
        self.assertEqual(np.ndim(rank), 0)
        self.assertEqual(int(rank), 2)
        # A residual-computing driver fills in residuals and singular values.
        (x, res, rank, sv) = im.lstsq(A, B, driver='gelsd')
        self.assertTrue(np.allclose(x, nx))
        self.assertEqual(res.shape, (1,))
        self.assertEqual(sv.shape, (2,))
        self.assertEqual(int(rank), 2)
        # 'gels' computes residuals but neither rank nor singular values.
        (x, res, rank, sv) = im.lstsq(A, B, driver='gels')
        self.assertEqual(res.shape, (1,))
        self.assertEqual(rank.shape, (0,))
        self.assertEqual(sv.shape, (0,))
        # Tensors agree with arrays, part for part.
        (xt, rest, rankt, svt) = im.lstsq(torch.tensor(A), torch.tensor(B),
                                          driver='gelsd')
        self.assertTrue(torch.allclose(xt, torch.tensor(x)))
        self.assertEqual(tuple(rest.shape), (1,))
        self.assertEqual(tuple(svt.shape), (2,))
        # Batched systems work for both backends.
        Ab = np.stack([A, 2.0 * A])
        Bb = np.stack([B, B])
        (xb, resb, rankb, svb) = im.lstsq(Ab, Bb, driver='gelsd')
        self.assertEqual((xb.shape, resb.shape, rankb.shape, svb.shape),
                         ((2, 2, 1), (2, 1), (2,), (2, 2)))
        # Both arguments must be unit-less, and the driver is validated.
        with self.assertRaises(TypeError):
            im.lstsq(il.quant(A, 'm'), B)
        with self.assertRaises(TypeError):
            im.lstsq(A, il.quant(B, 'm'))
        with self.assertRaises(ValueError):
            im.lstsq(A, B, driver='bogus')
        # A gradient computed through lstsq reaches the right-hand side.
        Bt = torch.tensor(B, requires_grad=True)
        (xt, _, _, _) = im.lstsq(torch.tensor(A), Bt)
        xt.sum().backward()
        self.assertIsNotNone(Bt.grad)


    def test_like_functions(self):
        import immlib as il
        import immlib.math as im
        import numpy as np
        import torch
        import pint
        import scipy.sparse as sps
        a = il.quant(np.arange(6.0).reshape(2, 3), 'm')
        # The result has the example's shape, backend, and units.
        z = im.zeros_like(a)
        self.assertIsInstance(z.m, np.ndarray)
        self.assertEqual(z.units, a.units)
        self.assertTrue(np.array_equal(z.m, np.zeros((2, 3))))
        o = im.ones_like(a)
        self.assertEqual(o.units, a.units)
        self.assertTrue(np.allclose(o.m, 1.0))
        # A bare fill value is taken in the example's units; a quantity is
        # converted into them.
        self.assertTrue(np.allclose(im.full_like(a, 3).m, 3.0))
        self.assertTrue(np.allclose(im.full_like(a, il.quant(1.0, 'cm')).m,
                                    0.01))
        # dtype is honored.
        self.assertEqual(im.ones_like(a, dtype=np.float32).m.dtype,
                         np.dtype('float32'))
        # A tensor example gives a tensor result of the same shape.
        t = il.quant(torch.arange(6.0).reshape(2, 3), 'm')
        self.assertIsInstance(im.zeros_like(t).m, torch.Tensor)
        self.assertTrue(torch.allclose(im.full_like(t, 2.0).m,
                                       torch.full((2, 3), 2.0)))
        # A unit-less example gives a unit-less result, and a unit-bearing
        # fill value cannot be used with it.
        self.assertIsNone(im.ones_like(np.arange(3.0)).units)
        with self.assertRaises(pint.DimensionalityError):
            im.full_like(np.arange(3.0), il.quant(1.0, 'm'))
        # Sparse examples are rejected (the result would be dense).
        with self.assertRaises(TypeError):
            im.zeros_like(il.quant(sps.csr_matrix(np.eye(3)), None))


    def test_random_and_empty_like(self):
        import immlib as il
        import immlib.math as im
        import numpy as np
        import torch
        a = il.quant(np.arange(6.0).reshape(2, 3), 'm')
        # empty_like defines only shape, dtype, and units.
        e = im.empty_like(a)
        self.assertEqual(e.shape, a.shape)
        self.assertEqual(e.units, a.units)
        # Random allocators keep shape and units, and are seedable per backend.
        np.random.seed(12345)
        r1 = im.rand_like(a)
        np.random.seed(12345)
        r2 = im.rand_like(a)
        self.assertEqual(r1.units, a.units)
        self.assertTrue(np.array_equal(r1.m, r2.m))
        self.assertTrue(np.all(r1.m >= 0.0) and np.all(r1.m < 1.0))
        np.random.seed(7)
        self.assertEqual(im.randint_like(a, 0, 10).m.dtype, np.dtype('int64'))
        self.assertTrue(np.all(im.randint_like(a, 0, 3).m < 3))
        # The tensor path uses torch's allocators and is seedable with
        # torch.manual_seed.
        t = il.quant(torch.zeros(2, 3), 'm')
        torch.manual_seed(99)
        t1 = im.randn_like(t)
        torch.manual_seed(99)
        t2 = im.randn_like(t)
        self.assertIsInstance(t1.m, torch.Tensor)
        self.assertTrue(torch.equal(t1.m, t2.m))
        self.assertEqual(t1.units, t.units)

    def test_shape_helpers(self):
        import immlib as il
        import immlib.math as im
        import numpy as np
        import scipy.sparse as sps
        self.assertEqual(im.atleast_1d(il.quant(5.0, 'm')).shape, (1,))
        self.assertEqual(im.atleast_2d(il.quant(5.0, 'm')).shape, (1, 1))
        self.assertEqual(im.atleast_3d(il.quant(5.0, 'm')).shape, (1, 1, 1))
        self.assertEqual(im.atleast_2d(il.quant(5.0, 'm')).units, il.unit('m'))
        a = il.quant(np.arange(6.0).reshape(2, 3), 'm')
        b = im.broadcast_to(a, (4, 2, 3))
        self.assertEqual(b.shape, (4, 2, 3))
        self.assertEqual(b.units, a.units)
        self.assertFalse(b.m.flags['WRITEABLE'])      # a view, as in numpy
        # expand expands size-1 dimensions (and may add leading ones);
        # a -1 keeps a dimension's size.
        x = il.quant(np.array([[1.0], [2.0]]), 'm')     # shape (2, 1)
        self.assertEqual(im.expand(x, 2, 3).shape, (2, 3))
        self.assertEqual(im.expand(x, -1, 3).shape, (2, 3))
        self.assertEqual(im.expand(x, 4, 2, 3).shape, (4, 2, 3))
        # sparse is rejected
        with self.assertRaises(TypeError):
            im.broadcast_to(il.quant(sps.csr_matrix(np.eye(3)), None), (5, 3))

    def test_sequence_helpers(self):
        import immlib as il
        import immlib.math as im
        import numpy as np
        a = il.quant(np.array([[2.0, 3.0], [4.0, 5.0]]), 'm')
        p = im.cumprod(a, 1)
        self.assertTrue(np.array_equal(p.m, [[2, 6], [4, 20]]))
        self.assertEqual(p.units, a.units)
        with self.assertRaises(TypeError):
            im.cumprod(a, None)
        d = im.diff(a, 1, 1)
        self.assertTrue(np.array_equal(d.m, [[1.0], [1.0]]))
        self.assertEqual(d.units, a.units)
        self.assertTrue(np.array_equal(im.flipud(a).m, np.flipud(a.m)))
        self.assertTrue(np.array_equal(im.fliplr(a).m, np.fliplr(a.m)))

    def test_tolerance_predicates(self):
        import immlib as il
        import immlib.math as im
        import numpy as np
        import pint
        import torch
        a = il.quant(np.array([1.0, 2.0]), 'm')
        # The second argument is converted into the first's units.
        self.assertTrue(im.isclose(a, il.quant(np.array([100.0, 200.0]), 'cm')).all())
        self.assertTrue(im.allclose(a, il.quant(np.array([100.0, 200.0]), 'cm')))
        self.assertFalse(im.allclose(a, il.quant(np.array([1.0, 2.1]), 'm')))
        # A unit-less (or bare) operand is dimensionless, so comparing it with
        # a dimensional one raises, exactly as Pint does.
        with self.assertRaises(pint.DimensionalityError):
            im.isclose(a, a.m)
        with self.assertRaises(pint.DimensionalityError):
            im.isclose(a.m, a)
        with self.assertRaises(pint.DimensionalityError):
            im.allclose(a, il.quant(np.array([1.0, 2.0]), 's'))
        # Both unit-less: an ordinary comparison of the magnitudes.
        self.assertTrue(im.allclose(il.quant(np.array([1.0])), np.array([1.0])))
        # The tensor path follows the same rules.
        self.assertTrue(im.allclose(il.quant(torch.tensor([1.0, 2.0]), 'm'),
                                    il.quant(torch.tensor([100.0, 200.0]), 'cm')))
        # count_nonzero ignores units and returns a plain count.
        self.assertEqual(im.count_nonzero(a), 2)
        self.assertTrue(np.array_equal(
            im.count_nonzero(il.quant(np.zeros((2, 3)), 'm'), dim=0),
            np.zeros(3)))


    def test_unit_aware_linalg(self):
        import immlib as il
        import immlib.math as im
        import numpy as np
        import torch
        import scipy.sparse as sps
        # norm keeps the units of its argument.
        v = il.quant(np.array([3.0, 4.0]), 'm')
        self.assertAlmostEqual(im.norm(v).m, 5.0)
        self.assertEqual(im.norm(v).units, v.units)
        M = il.quant(np.array([[1.0, 2.0], [3.0, 4.0]]), 'm')
        n = im.norm(M, dim=0, keepdim=True)
        self.assertEqual(n.m.shape, (1, 2))
        self.assertEqual(n.units, M.units)
        self.assertTrue(torch.allclose(im.norm(il.quant(torch.tensor([3.0, 4.0]), 'm')).m,
                                       torch.tensor(5.0)))
        # diag/diagonal/tril/triu/trace keep the units.
        self.assertTrue(np.array_equal(im.diag(M).m, [1.0, 4.0]))
        self.assertEqual(im.diag(M).units, M.units)
        d = il.quant(np.array([1.0, 2.0, 3.0]), 'm')
        self.assertTrue(np.array_equal(im.diag(d).m, np.diag([1.0, 2.0, 3.0])))
        self.assertTrue(np.array_equal(im.diagonal(M).m, [1.0, 4.0]))
        self.assertTrue(np.array_equal(im.tril(M).m, [[1, 0], [3, 4]]))
        self.assertTrue(np.array_equal(im.triu(M).m, [[1, 2], [0, 4]]))
        self.assertAlmostEqual(im.trace(M).m, 5.0)
        self.assertEqual(im.trace(M).units, M.units)
        # Products multiply the units.
        x = il.quant(np.array([1.0, 2.0]), 'm')
        y = il.quant(np.array([3.0, 4.0]), 's')
        o = im.outer(x, y)
        self.assertTrue(np.array_equal(o.m, [[3, 4], [6, 8]]))
        self.assertEqual(o.units, il.unit('m*s'))
        self.assertAlmostEqual(im.inner(x, y).m, 11.0)
        self.assertEqual(im.inner(x, y).units, il.unit('m*s'))
        u = il.quant(np.array([1.0, 0.0, 0.0]), 'm')
        w = il.quant(np.array([0.0, 1.0, 0.0]), 's')
        c = im.cross(u, w)
        self.assertTrue(np.allclose(c.m, [0.0, 0.0, 1.0]))
        self.assertEqual(c.units, il.unit('m*s'))
        t = im.tensordot(il.quant(np.eye(2), 'm'), il.quant(np.ones((2, 2)), 's'),
                         dims=2)
        self.assertAlmostEqual(t.m, 2.0)
        self.assertEqual(t.units, il.unit('m*s'))
        # outer is 1-dimensional only, in both backends.
        with self.assertRaises(ValueError):
            im.outer(M, M)
        # The tensor path agrees.
        self.assertTrue(torch.allclose(
            im.outer(il.quant(torch.tensor([1.0, 2.0]), 'm'),
                     il.quant(torch.tensor([3.0, 4.0]), 's')).m,
            torch.tensor([[3.0, 4.0], [6.0, 8.0]])))
        # Sparse arguments are rejected.
        with self.assertRaises(TypeError):
            im.norm(il.quant(sps.csr_matrix(np.eye(3)), None))


    def test_searchsorted(self):
        import immlib as il
        import immlib.math as im
        import numpy as np
        import torch
        import pint
        a = il.quant(np.array([1.0, 2.0, 2.0, 3.0]), 'm')
        v = il.quant(np.array([0.0, 200.0, 250.0]), 'cm')   # 0, 2, 2.5 m
        # The values are converted into the sequence's units.
        self.assertTrue(np.array_equal(im.searchsorted(a, v), [0, 1, 3]))
        self.assertTrue(np.array_equal(im.searchsorted(a, v, side='right'),
                                       [0, 3, 3]))
        # 'right' is the torch spelling of side='right'.
        self.assertTrue(np.array_equal(im.searchsorted(a, v, right=True),
                                       [0, 3, 3]))
        with self.assertRaises(TypeError):
            im.searchsorted(a, v, side='right', right=True)
        with self.assertRaises(ValueError):
            im.searchsorted(a, v, side='middle')
        # A unit-less value compared with a dimensional one raises.
        with self.assertRaises(pint.DimensionalityError):
            im.searchsorted(a, a.m)
        # The sequence must be 1-dimensional, in both backends.
        with self.assertRaises(ValueError):
            im.searchsorted(il.quant(np.ones((2, 3)), 'm'),
                            il.quant(np.ones(3), 'm'))
        # The tensor path agrees and returns plain integer indices.
        r = im.searchsorted(il.quant(torch.tensor([1.0, 2.0, 2.0, 3.0]), 'm'),
                            il.quant(torch.tensor([0.0, 2.0, 2.5]), 'm'))
        self.assertIsInstance(r, torch.Tensor)
        self.assertTrue(torch.equal(r, torch.tensor([0, 1, 3])))
