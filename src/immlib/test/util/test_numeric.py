# -*- coding: utf-8 -*-
################################################################################
# immlib/test/util/test_numeric.py
#
# Tests of the numeric module in immlib: i.e., tests for the code in the
# immlib.util._numeric module.


# Dependencies #################################################################

from unittest import TestCase


# Tests ########################################################################

class TestUtilNumeric(TestCase):
    """Tests the immlib.util._numeric module."""

    # Numeric Types ############################################################
    def test_is_number(self):
        from immlib import is_number
        import torch, numpy as np
        # is_number returns True for numbers and False for non-numbers.
        self.assertTrue(is_number(0))
        self.assertTrue(is_number(5))
        self.assertTrue(is_number(10.0))
        self.assertTrue(is_number(-2.0 + 9.0j))
        self.assertTrue(is_number(True))
        self.assertFalse(is_number('abc'))
        self.assertFalse(is_number('10'))
        self.assertFalse(is_number(None))
        # Scalar arrays and tensors are counted as scalars.
        self.assertFalse(is_number(np.array(5)))
        self.assertFalse(is_number(np.array(10.0 + 2.0j)))
        self.assertFalse(is_number(torch.tensor(5)))
        self.assertFalse(is_number(torch.tensor(10.0 + 2.0j)))
        # Arrays and tensors are *not* checked for their dtype to be such--that
        # is instead performed by is_numberdata.
        self.assertFalse(is_number(torch.tensor([1,2,3])))
        self.assertFalse(is_number(np.array([[-12.0]])))
    def test_is_bool(self):
        from immlib import is_bool
        import torch, numpy as np
        # is_bool returns True for bools and False for non-bools.
        self.assertFalse(is_bool(0))
        self.assertFalse(is_bool(5))
        self.assertFalse(is_bool(10.0))
        self.assertFalse(is_bool(-2.0 + 9.0j))
        self.assertTrue(is_bool(True))
        self.assertFalse(is_bool('abc'))
        self.assertFalse(is_bool('10'))
        self.assertFalse(is_bool(None))
        # Scalar arrays and tensors are allowed.
        self.assertTrue(is_bool(np.array(False)))
        self.assertTrue(is_bool(torch.tensor(True)))
        self.assertFalse(is_bool(np.array(10.0 + 2.0j)))
        self.assertFalse(is_bool(torch.tensor(10.0 + 2.0j)))
        # is_bool does not return True for collections (use is_intdata
        # instead).
        self.assertFalse(is_bool(torch.tensor([1,2,3])))
        self.assertFalse(is_bool(np.array([[12.0]])))
    def test_is_integer(self):
        from immlib import is_integer
        import torch, numpy as np
        # is_integer returns True for integers and False for non-integers.
        self.assertTrue(is_integer(0))
        self.assertTrue(is_integer(5))
        self.assertFalse(is_integer(10.0))
        self.assertFalse(is_integer(-2.0 + 9.0j))
        self.assertTrue(is_integer(True))
        self.assertFalse(is_integer('abc'))
        self.assertFalse(is_integer('10'))
        self.assertFalse(is_integer(None))
        # Scalar arrays and tensors are allowed.
        self.assertTrue(is_integer(np.array(5)))
        self.assertTrue(is_integer(torch.tensor(5)))
        self.assertFalse(is_integer(np.array(10.0 + 2.0j)))
        self.assertFalse(is_integer(torch.tensor(10.0 + 2.0j)))
        # is_integer does not return True for collections (use is_intdata
        # instead).
        self.assertFalse(is_integer(torch.tensor([1,2,3])))
        self.assertFalse(is_integer(np.array([[12.0]])))
    def test_is_real(self):
        from immlib import is_real
        import torch, numpy as np
        # is_real returns True for reals and False for non-reals.
        self.assertTrue(is_real(0))
        self.assertTrue(is_real(5))
        self.assertTrue(is_real(10.0))
        self.assertTrue(is_real(True))
        self.assertFalse(is_real(-2.0 + 9.0j))
        self.assertFalse(is_real('abc'))
        self.assertFalse(is_real('10'))
        self.assertFalse(is_real(None))
        # Scalar arrays and tensors are allowed.
        self.assertTrue(is_real(np.array(5)))
        self.assertTrue(is_real(torch.tensor(5)))
        self.assertFalse(is_real(np.array(10.0 + 2.0j)))
        self.assertFalse(is_real(torch.tensor(10.0 + 2.0j)))
        # Arrays and tensors are checked for their dtype to be such that their
        # elements are numbers.
        self.assertFalse(is_real(torch.tensor([1,2,3])))
        self.assertFalse(is_real(np.array([[12.0]])))
    def test_is_complex(self):
        from immlib import is_complex
        import torch, numpy as np
        # is_complex returns True for complexs and False for non-complexs.
        self.assertTrue(is_complex(0))
        self.assertTrue(is_complex(5))
        self.assertTrue(is_complex(10.0))
        self.assertTrue(is_complex(True))
        self.assertTrue(is_complex(-2.0 + 9.0j))
        self.assertFalse(is_complex('abc'))
        self.assertFalse(is_complex('10'))
        self.assertFalse(is_complex(None))
        # Scalar arrays/tensors are allowed.
        self.assertTrue(is_complex(np.array(5)))
        self.assertTrue(is_complex(np.array(10.0 + 2.0j)))
        self.assertTrue(is_complex(torch.tensor(5)))
        self.assertTrue(is_complex(torch.tensor(10.0 + 2.0j)))
        # Arrays and tensors are only checked if they are scalars.
        self.assertFalse(is_complex(torch.tensor([1,2,3])))
        self.assertFalse(is_complex(np.array([[12.0]])))

    # NumPy Utilities ##########################################################
    def test_is_numpydtype(self):
        from immlib.util import is_numpydtype
        import torch, numpy as np
        # is_numpydtype returns true for dtypes and dtypes alone.
        self.assertTrue(is_numpydtype(np.dtype('int')))
        self.assertTrue(is_numpydtype(np.dtype(float)))
        self.assertTrue(is_numpydtype(np.dtype(np.bool_)))
        self.assertFalse(is_numpydtype('int'))
        self.assertFalse(is_numpydtype(float))
        self.assertFalse(is_numpydtype(np.bool_))
        self.assertFalse(is_numpydtype(torch.float))
    def test_like_numpydtype(self):
        from immlib.util import like_numpydtype
        import torch, numpy as np
        # Anything that can be converted into a numpy dtype object is considered
        # to be like a numpy dtype.
        self.assertTrue(like_numpydtype('int'))
        self.assertTrue(like_numpydtype(float))
        self.assertTrue(like_numpydtype(np.bool_))
        self.assertFalse(like_numpydtype('abc'))
        self.assertFalse(like_numpydtype(10))
        self.assertFalse(like_numpydtype(...))
        # Note that None can be converted to a numpy dtype (float64).
        self.assertTrue(like_numpydtype(None))
        # numpy dtypes themselves are like numpy dtypes, as are torch dtypes.
        self.assertTrue(like_numpydtype(np.dtype(int)))
        self.assertTrue(like_numpydtype(torch.float))
    def test_to_numpydtype(self):
        from immlib.util import to_numpydtype
        import torch, numpy as np
        # Converting a numpy dtype into a dtype results in the identical dtype.
        dt = np.dtype(int)
        self.assertIs(dt, to_numpydtype(dt))
        # Torch dtypes can be converted into a numpy dtype.
        self.assertEqual(np.dtype(np.float64), to_numpydtype(torch.float64))
        self.assertEqual(np.dtype('int32'), to_numpydtype(torch.int32))
        # Ordinary tags can be converted into dtypes as well.
        self.assertEqual(np.dtype(np.float64), to_numpydtype(np.float64))
        self.assertEqual(np.dtype('int32'), to_numpydtype('int32'))
    def test_is_array(self):
        from immlib import (is_array, quant)
        from numpy import (array, linspace, dot)
        from scipy.sparse import csr_matrix
        import torch, numpy as np
        # By default, is_array() returns True for numpy arrays, scipy sparse
        # matrices, and quantities of these.
        arr = linspace(0, 1, 25)
        mtx = dot(linspace(0, 1, 10)[:,None], linspace(1, 2, 10)[None,:])
        sp_mtx = csr_matrix(
            ([1.0, 0.5, 0.5, 0.2, 0.1],
             ([0, 0, 4, 5, 9], [4, 9, 4, 1, 8])),
            shape=(10, 10),
            dtype=float)
        q_arr = quant(arr, 'mm')
        q_mtx = quant(arr, 'seconds')
        q_sp_mtx = quant(sp_mtx, 'kg')
        self.assertTrue(is_array(arr))
        self.assertTrue(is_array(mtx))
        self.assertTrue(is_array(sp_mtx))
        self.assertTrue(is_array(q_arr))
        self.assertTrue(is_array(q_mtx))
        self.assertTrue(is_array(q_sp_mtx))
        # Things like lists, numbers, and torch tensors are not arrays.
        self.assertFalse(is_array('abc'))
        self.assertFalse(is_array(10))
        self.assertFalse(is_array([12.0, 0.5, 3.2]))
        self.assertFalse(is_array(torch.tensor([1.0, 2.0, 3.0])))
        self.assertFalse(is_array(quant(torch.tensor([1.0, 2.0, 3.0]), 'mm')))
        # We can use the dtype argument to restrict what we consider an array by
        # its dtype. The dtype of the is_array argument must be a sub-dtype of
        # the dtype parameter.
        self.assertTrue(is_array(arr, dtype=np.number))
        self.assertTrue(is_array(arr, dtype=arr.dtype))
        self.assertFalse(is_array(arr, dtype=np.str_))
        # If a tuple is passed for the dtype, the dtype must match one of the
        # tuple's members exactly.
        self.assertTrue(is_array(mtx, dtype=(mtx.dtype,)))
        self.assertTrue(is_array(mtx, dtype=(mtx.dtype,np.dtype(int),np.str_)))
        self.assertFalse(is_array(mtx, dtype=(np.dtype(int),np.str_)))
        self.assertFalse(is_array(np.array([], dtype=np.int32),
                                  dtype=(np.int64,)))
        # torch dtypes can be interpreted into numpy dtypes.
        self.assertTrue(is_array(mtx, dtype=torch.as_tensor(mtx).dtype))
        # We can use the ndim argument to restrict the number of dimensions that
        # an array can have in order to be considered a matching array.
        # Typically, this is just the number of dimensions.
        self.assertTrue(is_array(arr, ndim=1))
        self.assertTrue(is_array(mtx, ndim=2))
        self.assertFalse(is_array(arr, ndim=2))
        self.assertFalse(is_array(mtx, ndim=1))
        # Alternately, a tuple may be given, in which case any of the dimension
        # counts in the tuple are accepted.
        self.assertTrue(is_array(mtx, ndim=(1,2)))
        self.assertTrue(is_array(arr, ndim=(1,2)))
        self.assertFalse(is_array(mtx, ndim=(1,3)))
        self.assertFalse(is_array(arr, ndim=(0,2)))
        # Scalar arrays have 0 dimensions.
        self.assertTrue(is_array(array(0), ndim=0))
        # The shape option is a more specific version of the ndim parameter. It
        # lets you specify what kind of shape is required of the array. The most
        # straightforward usage is to require a specific shape.
        self.assertTrue(is_array(arr, shape=(25,)))
        self.assertTrue(is_array(mtx, shape=(10,10)))
        self.assertFalse(is_array(arr, shape=(25,25)))
        self.assertFalse(is_array(mtx, shape=(10,)))
        # A -1 value that appears in the shape option represents any size along
        # that dimension (a wildcard). Any number of -1s can be included.
        self.assertTrue(is_array(arr, shape=(-1,)))
        self.assertTrue(is_array(mtx, shape=(-1,10)))
        self.assertTrue(is_array(mtx, shape=(10,-1)))
        self.assertTrue(is_array(mtx, shape=(-1,-1)))
        self.assertFalse(is_array(mtx, shape=(1,-1)))
        # No more than 1 ellipsis may be included in the shape to indicate that
        # any number of dimensions, with any sizes, can appear in place of the
        # ellipsis.
        self.assertTrue(is_array(arr, shape=(...,25)))
        self.assertTrue(is_array(arr, shape=(25,...)))
        self.assertFalse(is_array(arr, shape=(25,...,25)))
        self.assertTrue(is_array(mtx, shape=(...,10)))
        self.assertTrue(is_array(mtx, shape=(10,...)))
        self.assertTrue(is_array(mtx, shape=(10,...,10)))
        self.assertTrue(is_array(mtx, shape=(10,10,...)))
        self.assertTrue(is_array(mtx, shape=(...,10,10)))
        self.assertFalse(is_array(mtx, shape=(10,...,10,10)))
        self.assertFalse(is_array(mtx, shape=(10,10,...,10)))
        self.assertTrue(is_array(np.zeros((1,2,3,4,5)), shape=(1,...,4,5)))
        # The numel option allows one to specify the number of elements that an
        # object must have. This does not care about dimensionality.
        self.assertTrue(is_array(arr, numel=25))
        self.assertFalse(is_array(arr, numel=26))
        self.assertTrue(is_array(np.array(0), numel=1))
        self.assertTrue(is_array(np.array([0]), numel=1))
        self.assertTrue(is_array(np.array([[0]]), numel=1))
        # The frozen option can be used to test whether an array is frozen
        # or not. This is judged by the array's 'WRITEABLE' flag.
        self.assertFalse(is_array(arr, frozen=True))
        self.assertTrue(is_array(arr, frozen=False))
        self.assertFalse(is_array(mtx, frozen=True))
        self.assertTrue(is_array(mtx, frozen=False))
        # If we change the flags of these arrays, they become frozen.
        arr.setflags(write=False)
        mtx.setflags(write=False)
        self.assertTrue(is_array(arr, frozen=True))
        self.assertFalse(is_array(arr, frozen=False))
        self.assertTrue(is_array(mtx, frozen=True))
        self.assertFalse(is_array(mtx, frozen=False))
        # The sparse option can test whether an object is a sparse matrix or
        # not. By default sparse is None, meaning that it doesn't matter whether
        # an object is sparse, but sometimes you want to check for strict
        # numpy arrays only.
        self.assertTrue(is_array(arr, sparse=False))
        self.assertTrue(is_array(mtx, sparse=False))
        self.assertFalse(is_array(sp_mtx, sparse=False))
        self.assertFalse(is_array(arr, sparse=True))
        self.assertFalse(is_array(mtx, sparse=True))
        self.assertTrue(is_array(sp_mtx, sparse=True))
        # You can also require a kind of sparse matrix.
        self.assertTrue(is_array(sp_mtx, sparse='csr'))
        self.assertFalse(is_array(sp_mtx, sparse='csc'))
        # The quant option can be used to control whether the object must or
        # must not be a quantity.
        self.assertTrue(is_array(arr, quant=False))
        self.assertTrue(is_array(mtx, quant=False))
        self.assertFalse(is_array(arr, quant=True))
        self.assertFalse(is_array(mtx, quant=True))
        self.assertTrue(is_array(q_arr, quant=True))
        self.assertTrue(is_array(q_mtx, quant=True))
        self.assertFalse(is_array(q_arr, quant=False))
        self.assertFalse(is_array(q_mtx, quant=False))
        # The units option can be used to require that either an object have
        # no units (or is not a quantity) or that it have specific units.
        self.assertTrue(is_array(arr, unit=None))
        self.assertTrue(is_array(mtx, unit=None))
        self.assertFalse(is_array(arr, unit='mm'))
        self.assertFalse(is_array(mtx, unit='s'))
        self.assertFalse(is_array(q_arr, unit=None))
        self.assertFalse(is_array(q_mtx, unit=None))
        self.assertTrue(is_array(q_arr, unit='mm'))
        self.assertTrue(is_array(q_mtx, unit='s'))
        self.assertFalse(is_array(q_arr, unit='s'))
        self.assertFalse(is_array(q_mtx, unit='mm'))
    def test_to_array(self):
        from immlib import (to_array, quant, is_quant, units)
        from numpy import (array, linspace, dot)
        from scipy.sparse import (csr_matrix, issparse)
        import torch, numpy as np, pint
        # We'll use a few objects throughout our tests, which we setup now.
        arr = linspace(0, 1, 25)
        mtx = dot(linspace(0, 1, 10)[:,None], linspace(0, 2, 10)[None,:])
        sp_mtx = csr_matrix(([1.0, 0.5, 0.5, 0.2, 0.1],
                             ([0, 0, 4, 5, 9], [4, 9, 4, 1, 8])),
                            shape=(10, 10), dtype=float)
        q_arr = quant(arr, 'mm')
        q_mtx = quant(arr, 'seconds')
        q_sp_mtx = quant(sp_mtx, 'kg')
        # For an object that is already a numpy array, any call that doesn't
        # request a copy and that doesn't change its parameters will return the
        # identical object.
        self.assertIs(arr, to_array(arr))
        self.assertIs(arr, to_array(arr, sparse=False, frozen=False))
        self.assertIs(arr, to_array(arr, quant=False))
        # If we change the parameters of the returned array, we will get
        # different (but typically equal) objects back.
        self.assertIsNot(arr, to_array(arr, frozen=True))
        self.assertTrue(np.array_equal(arr, to_array(arr, frozen=True)))
        # We can also request that a copy be made like with np.array.
        self.assertIsNot(arr, to_array(arr, copy=True))
        self.assertTrue(np.array_equal(arr, to_array(arr, copy=True)))
        # The sparse flag can be used to convert to/from a sparse array.
        self.assertIsInstance(to_array(sp_mtx, sparse=False), np.ndarray)
        self.assertTrue(np.array_equal(to_array(sp_mtx, sparse=False),
                                       sp_mtx.todense()))
        self.assertTrue(issparse(to_array(mtx, sparse=True)))
        self.assertTrue(np.array_equal(to_array(mtx, sparse=True).todense(),
                                       mtx))
        # The frozen flag ensures that the return value does or does not have
        # the writeable flag set.
        self.assertFalse(to_array(mtx, frozen=True).flags['WRITEABLE'])
        self.assertTrue(np.array_equal(to_array(mtx, frozen=True), mtx))
        self.assertIsNot(to_array(mtx, frozen=True), mtx)
        # The quant argument can be used to enforce the return of quantities or
        # non-quantities.
        self.assertIsInstance(to_array(arr, quant=True), units.Quantity)
        # The unit parameter can be used to specify what unit to use.
        self.assertTrue(np.array_equal(q_arr.m,
                                       to_array(arr, quant=True, unit='mm').m))
        # If no unit is provided, then dimensionless units are assumed (1
        # dimensionless is equivalent to 1 count, 1 turn, and a few others).
        self.assertEqual(to_array(arr, quant=True).u, units.dimensionless)
        # We can also use unit to extract a specific unit from a quantity.
        self.assertEqual(1000, to_array(1 * units.meter, unit='mm').m)
        # However, a non-quantity is always assumed to already have the units
        # requested, so converting it to a particular unit (but not converting
        # it to a quantity) results in the same object.
        self.assertIs(to_array(arr, quant=False, unit='mm'), arr)
        # If we simply request an array with a unit, without specifying that it
        # not be a quantity, we get a quantity back.
        self.assertIsInstance(to_array(arr, unit='mm'), pint.Quantity)
        # An error is raised if you try to request no units for a quantity.
        with self.assertRaises(ValueError):
            to_array(arr, quant=True, unit=None)

    # PyTorch Utilities ########################################################
    def test_is_torchdtype(self):
        from immlib.util import is_torchdtype
        import torch, numpy as np
        # is_torchdtype returns true for torch's dtypes and its dtypes alone.
        self.assertTrue(is_torchdtype(torch.int))
        self.assertTrue(is_torchdtype(torch.float))
        self.assertTrue(is_torchdtype(torch.bool))
        self.assertFalse(is_torchdtype('int'))
        self.assertFalse(is_torchdtype(float))
        self.assertFalse(is_torchdtype(np.bool_))
    def test_like_torchdtype(self):
        from immlib.util import like_torchdtype
        import torch, numpy as np
        # Anything that can be converted into a torch dtype object is considered
        # to be like a torch dtype.
        self.assertTrue(like_torchdtype('int'))
        self.assertTrue(like_torchdtype(float))
        self.assertTrue(like_torchdtype(np.bool_))
        self.assertFalse(like_torchdtype('abc'))
        self.assertFalse(like_torchdtype(10))
        self.assertFalse(like_torchdtype(...))
        # Note that None can be converted to a torch dtype (float64).
        self.assertTrue(like_torchdtype(None))
        # torch dtypes themselves are like torch dtypes.
        self.assertTrue(like_torchdtype(torch.float))
    def test_to_torchdtype(self):
        from immlib.util import to_torchdtype
        import torch, numpy as np
        # Converting a numpy dtype into a dtype results in the identical dtype.
        dt = torch.int
        self.assertIs(dt, to_torchdtype(dt))
        # Numpy dtypes can be converted into a torch dtype.
        self.assertEqual(torch.float64, to_torchdtype(np.dtype('float64')))
        self.assertEqual(torch.int32, to_torchdtype(np.int32))
    def test_is_tensor(self):
        from immlib import (is_tensor, quant)
        from scipy.sparse import csr_matrix
        import torch, numpy as np
        # By default, is_tensor() returns True for PyTorch tensors and
        # quantities whose magnitudes are PyTorch tensors.
        arr = torch.linspace(0, 1, 25)
        mtx = torch.mm(torch.linspace(0, 1, 10)[:,None],
                       torch.linspace(1, 2, 10)[None,:])
        sp_mtx = torch.sparse_coo_tensor(torch.tensor([[0, 0, 4, 5, 9],
                                                       [4, 9, 4, 1, 8]]),
                                         torch.tensor([1, 0.5, 0.5, 0.2, 0.1]),
                                         (10, 10),
                                         dtype=float)
        q_arr = quant(arr, 'mm')
        q_mtx = quant(arr, 'seconds')
        q_sp_mtx = quant(sp_mtx, 'kg')
        self.assertTrue(is_tensor(arr))
        self.assertTrue(is_tensor(mtx))
        self.assertTrue(is_tensor(sp_mtx))
        self.assertTrue(is_tensor(q_arr))
        self.assertTrue(is_tensor(q_mtx))
        self.assertTrue(is_tensor(q_sp_mtx))
        # Things like lists, numbers, and numpy arrays are not tensors.
        self.assertFalse(is_tensor('abc'))
        self.assertFalse(is_tensor(10))
        self.assertFalse(is_tensor([12.0, 0.5, 3.2]))
        self.assertFalse(is_tensor(np.array([1.0, 2.0, 3.0])))
        self.assertFalse(is_tensor(quant(np.array([1.0, 2.0, 3.0]), 'mm')))
        # We can use the dtype argument to restrict what we consider an array by
        # its dtype. The dtype of the is_array argument must be a sub-dtype of
        # the dtype parameter.
        self.assertTrue(is_tensor(arr, dtype=arr.dtype))
        self.assertFalse(is_tensor(arr, dtype=torch.int))
        # If a tuple is passed for the dtype, the dtype must match one of the
        # tuple's elements.
        self.assertTrue(is_tensor(mtx, dtype=(mtx.dtype,)))
        self.assertTrue(is_tensor(mtx, dtype=(mtx.dtype, torch.int)))
        self.assertFalse(is_tensor(mtx, dtype=(torch.int, torch.bool)))
        self.assertFalse(is_tensor(torch.tensor([], dtype=torch.int32),
                                   dtype=torch.int64))
        # torch dtypes can be interpreted into PyTorch dtypes.
        self.assertTrue(is_tensor(mtx, dtype=mtx.numpy().dtype))
        # We can use the ndim argument to restrict the number of dimensions that
        # an array can have in order to be considered a matching tensor.
        # Typically, this is just the number of dimensions.
        self.assertTrue(is_tensor(arr, ndim=1))
        self.assertTrue(is_tensor(mtx, ndim=2))
        self.assertFalse(is_tensor(arr, ndim=2))
        self.assertFalse(is_tensor(mtx, ndim=1))
        # Alternately, a tuple may be given, in which case any of the dimension
        # counts in the tuple are accepted.
        self.assertTrue(is_tensor(mtx, ndim=(1,2)))
        self.assertTrue(is_tensor(arr, ndim=(1,2)))
        self.assertFalse(is_tensor(mtx, ndim=(1,3)))
        self.assertFalse(is_tensor(arr, ndim=(0,2)))
        # Scalar tensors have 0 dimensions.
        self.assertTrue(is_tensor(torch.tensor(0), ndim=0))
        # The shape option is a more specific version of the ndim parameter. It
        # lets you specify what kind of shape is required of the tensor. The
        # most straightforward usage is to require a specific shape.
        self.assertTrue(is_tensor(arr, shape=(25,)))
        self.assertTrue(is_tensor(mtx, shape=(10,10)))
        self.assertFalse(is_tensor(arr, shape=(25,25)))
        self.assertFalse(is_tensor(mtx, shape=(10,)))
        # A -1 value that appears in the shape option represents any size along
        # that dimension (a wildcard). Any number of -1s can be included.
        self.assertTrue(is_tensor(arr, shape=(-1,)))
        self.assertTrue(is_tensor(mtx, shape=(-1,10)))
        self.assertTrue(is_tensor(mtx, shape=(10,-1)))
        self.assertTrue(is_tensor(mtx, shape=(-1,-1)))
        self.assertFalse(is_tensor(mtx, shape=(1,-1)))
        # No more than 1 ellipsis may be included in the shape to indicate that
        # any number of dimensions, with any sizes, can appear in place of the
        # ellipsis.
        self.assertTrue(is_tensor(arr, shape=(...,25)))
        self.assertTrue(is_tensor(arr, shape=(25,...)))
        self.assertFalse(is_tensor(arr, shape=(25,...,25)))
        self.assertTrue(is_tensor(mtx, shape=(...,10)))
        self.assertTrue(is_tensor(mtx, shape=(10,...)))
        self.assertTrue(is_tensor(mtx, shape=(10,...,10)))
        self.assertTrue(is_tensor(mtx, shape=(10,10,...)))
        self.assertTrue(is_tensor(mtx, shape=(...,10,10)))
        self.assertFalse(is_tensor(mtx, shape=(10,...,10,10)))
        self.assertFalse(is_tensor(mtx, shape=(10,10,...,10)))
        self.assertTrue(is_tensor(torch.zeros((1,2,3,4,5)), shape=(1,...,4,5)))
        # The numel option allows one to specify the number of elements that an
        # object must have. This does not care about dimensionality.
        self.assertTrue(is_tensor(arr, numel=25))
        self.assertFalse(is_tensor(arr, numel=26))
        self.assertTrue(is_tensor(torch.tensor(0), numel=1))
        self.assertTrue(is_tensor(torch.tensor([0]), numel=1))
        self.assertTrue(is_tensor(torch.tensor([[0]]), numel=1))
        # The sparse option can test whether an object is a sparse tensor or
        # not. By default sparse is None, meaning that it doesn't matter whether
        # an object is sparse, but sometimes you want to check for strict
        # sparsity requirements.
        self.assertTrue(is_tensor(arr, sparse=False))
        self.assertTrue(is_tensor(mtx, sparse=False))
        self.assertFalse(is_tensor(sp_mtx, sparse=False))
        self.assertFalse(is_tensor(arr, sparse=True))
        self.assertFalse(is_tensor(mtx, sparse=True))
        self.assertTrue(is_tensor(sp_mtx, sparse=True))
        # The quant option can be used to control whether the object must or
        # must not be a quantity.
        self.assertTrue(is_tensor(arr, quant=False))
        self.assertTrue(is_tensor(mtx, quant=False))
        self.assertFalse(is_tensor(arr, quant=True))
        self.assertFalse(is_tensor(mtx, quant=True))
        self.assertTrue(is_tensor(q_arr, quant=True))
        self.assertTrue(is_tensor(q_mtx, quant=True))
        self.assertFalse(is_tensor(q_arr, quant=False))
        self.assertFalse(is_tensor(q_mtx, quant=False))
        # The units option can be used to require that either an object have
        # no units (or is not a quantity) or that it have specific units.
        self.assertTrue(is_tensor(arr, unit=None))
        self.assertTrue(is_tensor(mtx, unit=None))
        self.assertFalse(is_tensor(arr, unit='mm'))
        self.assertFalse(is_tensor(mtx, unit='s'))
        self.assertFalse(is_tensor(q_arr, unit=None))
        self.assertFalse(is_tensor(q_mtx, unit=None))
        self.assertTrue(is_tensor(q_arr, unit='mm'))
        self.assertTrue(is_tensor(q_mtx, unit='s'))
        self.assertFalse(is_tensor(q_arr, unit='s'))
        self.assertFalse(is_tensor(q_mtx, unit='mm'))
    def test_to_tensor(self):
        from immlib import (to_tensor, quant, is_quant, units)
        import torch, numpy as np
        import pint
        # We'll use a few objects throughout our tests, which we setup now.
        arr = torch.linspace(0, 1, 25)
        mtx = torch.mm(torch.linspace(0, 1, 10)[:,None],
                       torch.linspace(0, 2, 10)[None,:])
        sp_mtx = torch.sparse_coo_tensor(torch.tensor([[0, 0, 4, 5, 9],
                                                       [4, 9, 4, 1, 8]]),
                                         torch.tensor([1, 0.5, 0.5, 0.2, 0.1]),
                                         (10,10),
                                         dtype=float)
        q_arr = quant(arr, 'mm')
        q_mtx = quant(arr, 'seconds')
        q_sp_mtx = quant(sp_mtx, 'kg')
        # For an object that is already a numpy array, any call that doesn't
        # request a copy and that doesn't change its parameters will return the
        # identical object.
        self.assertIs(arr, to_tensor(arr))
        self.assertIs(arr, to_tensor(arr, sparse=False))
        self.assertIs(arr, to_tensor(arr, quant=False))
        # If we change the parameters of the returned array, we will get
        # different (but typically equal) objects back.
        self.assertTrue(torch.equal(arr, to_tensor(arr, requires_grad=True)))
        # We can also request that a copy be made like with np.array.
        self.assertIsNot(arr, to_tensor(arr, copy=True))
        self.assertTrue(torch.equal(arr, to_tensor(arr, copy=True)))
        # The sparse flag can be used to convert to/from a sparse array.
        self.assertIsInstance(to_tensor(sp_mtx, sparse=False), torch.Tensor)
        self.assertTrue(torch.equal(to_tensor(sp_mtx, sparse=False),
                                    sp_mtx.to_dense()))
        self.assertTrue(to_tensor(mtx, sparse=True).is_sparse)
        self.assertTrue(torch.equal(to_tensor(mtx, sparse=True).to_dense(),
                                    mtx))
        # The quant argument can be used to enforce the return of quantities or
        # non-quantities.
        self.assertIsInstance(to_tensor(arr, quant=True), units.Quantity)
        # The unit parameter can be used to specify what unit to use.
        self.assertTrue(torch.equal(q_arr.m,
                                    to_tensor(arr, quant=True, unit='mm').m))
        # If no unit is provided, then dimensionless units are assumed (1
        # dimensionless is equivalent to 1 count, 1 turn, and a few others).
        self.assertEqual(to_tensor(arr, quant=True).u, units.dimensionless)
        # We can also use unit to extract a specific unit from a quantity.
        self.assertEqual(1000, to_tensor(1 * units.meter, unit='mm').m)
        # However, a non-quantity is always assumed to already have the units
        # requested, so converting it to a particular unit (but not converting
        # it to a quantity) results in the same object.
        self.assertIs(to_tensor(arr, quant=False, unit='mm'), arr)
        # If we simply request an array with a unit, without specifying that it
        # not be a quantity, we get a quantity back.
        self.assertIsInstance(to_tensor(arr, unit='mm'), pint.Quantity)
        # An error is raised if you try to request no units for a quantity.
        with self.assertRaises(ValueError):
            to_tensor(arr, quant=True, unit=None)

    # PyTorch and Numpy Helper Functions #######################################
    def test_is_numeric(self):
        from immlib import is_numeric
        import torch, numpy as np
        from scipy.sparse import csr_matrix
        # The is_numeric function is just a wrapper around is_array and
        # is_tensor that calls one or the other depending on whether the object
        # requested is a tensor or not. I.e., it passes all arguments through
        # and merely switches on the type.
        sp_a = csr_matrix(([0.5, 1.0], ([0,1], [3,2])), shape=(5,5))
        sp_t = torch.sparse_coo_tensor(torch.tensor([[0,1],[3,2]]),
                                       torch.tensor([0.5, 1]),
                                       (5,5))
        a = sp_a.todense()
        t = sp_t.to_dense()
        self.assertTrue(is_numeric(a))
        self.assertTrue(is_numeric(t))
        self.assertTrue(is_numeric(sp_a))
        self.assertTrue(is_numeric(sp_t))
        self.assertFalse(is_numeric('abc'))
        self.assertFalse(is_numeric([1,2,3]))
    def test_to_numeric(self):
        from immlib import to_numeric
        import torch, numpy as np
        from scipy.sparse import csr_matrix
        # The is_numeric function is just a wrapper around to_array and
        # to_tensor that calls one or the other depending on whether the object
        # requested is a tensor or not. I.e., it passes all arguments through
        # and merely switches on the type.
        sp_a = csr_matrix(([0.5, 1.0], ([0,1], [3,2])), shape=(5,5))
        sp_t = torch.sparse_coo_tensor(torch.tensor([[0,1],[3,2]]),
                                       torch.tensor([0.5, 1]),
                                       (5,5))
        a = np.array(sp_a.todense())
        t = sp_t.to_dense()
        self.assertIs(a, to_numeric(a))
        self.assertIs(t, to_numeric(t))
        self.assertIs(sp_a, to_numeric(sp_a))
        self.assertIs(sp_t, to_numeric(sp_t))
        self.assertIsInstance(to_numeric([1,2,3]), np.ndarray)
    def test_is_sparse(self):
        from immlib import is_sparse
        import torch, numpy as np
        from scipy.sparse import csr_matrix
        # is_sparse returns True for any sparse array and False for anything
        # other than a sparse array.
        sp_a = csr_matrix(([0.5, 1.0], ([0,1], [3,2])), shape=(5,5))
        sp_t = torch.sparse_coo_tensor(torch.tensor([[0,1],[3,2]]),
                                       torch.tensor([0.5, 1]),
                                       (5,5))
        self.assertTrue(is_sparse(sp_a))
        self.assertTrue(is_sparse(sp_t))
        self.assertFalse(is_sparse(sp_a.todense()))
        self.assertFalse(is_sparse(sp_t.to_dense()))
    def test_to_sparse(self):
        from immlib import to_sparse
        import torch, numpy as np
        from scipy.sparse import issparse
        # to_sparse supports the arguments of to_array and to_tensor (because it
        # simply calls through to these functions), but it always returns a
        # sparse object.
        m = np.array([[1.0, 0, 0, 0], [0, 0, 0, 0],
                      [0, 1.0, 0, 0], [0, 0, 0, 1.0]])
        t = torch.tensor(m)
        self.assertTrue(issparse(to_sparse(m)))
        self.assertTrue(to_sparse(t).is_sparse)
    def test_is_dense(self):
        from immlib import is_dense
        import torch, numpy as np
        from scipy.sparse import csr_array
        # is_dense returns True for any dense array and False for anything
        # other than a dense array.
        sp_a = csr_array(([0.5, 1.0], ([0,1], [3,2])), shape=(5,5))
        sp_t = torch.sparse_coo_tensor(
            torch.tensor([[0,1],[3,2]]),
            torch.tensor([0.5, 1]),
            (5,5))
        self.assertTrue(is_dense(sp_a.todense()))
        x = sp_t.to_dense()
        q = is_dense(sp_t.to_dense())
        self.assertTrue(q)
        self.assertFalse(is_dense(sp_a))
        self.assertFalse(is_dense(sp_t))
    def test_to_dense(self):
        from immlib import to_dense
        import torch, numpy as np
        from scipy.sparse import (issparse, csr_matrix)
        # to_dense supports the arguments of to_array and to_tensor (because it
        # simply calls through to these functions), but it always returns a
        # dense object.
        sp_a = csr_matrix(([0.5, 1.0], ([0,1], [3,2])), shape=(5,5))
        sp_t = torch.sparse_coo_tensor(torch.tensor([[0,1],[3,2]]),
                                       torch.tensor([0.5, 1]),
                                       (5,5))
        self.assertFalse(issparse(to_dense(sp_a)))
        self.assertFalse(to_dense(sp_t).is_sparse)
    def test_like_number(self):
        from immlib import like_number
        import torch, numpy as np
        # like_number returns True if the argument is a scalar number or if it
        # is convertible into a scalar number by the to_scalar function. Such
        # values include numbers and any numeric numpy array or PyTorch tensor
        # that has exactly one value.
        self.assertTrue(like_number(10))
        self.assertTrue(like_number(10.0))
        self.assertTrue(like_number(10.0 + 20.5j))
        self.assertTrue(like_number(True))
        self.assertTrue(like_number(np.array(10)))
        self.assertTrue(like_number(torch.tensor(10)))
        self.assertTrue(like_number([10]))
        self.assertTrue(like_number([[10]]))
        self.assertTrue(like_number([[[10]]]))
        self.assertTrue(like_number(np.array([10])))
        self.assertTrue(like_number(np.array([[10]])))
        self.assertTrue(like_number(np.array([[[10]]])))
        self.assertTrue(like_number(torch.tensor([10])))
        self.assertTrue(like_number(torch.tensor([[10]])))
        self.assertTrue(like_number(torch.tensor([[[10]]])))
        self.assertFalse(like_number('10'))
        self.assertFalse(like_number({'a':10}))
        self.assertFalse(like_number([1,2,3]))
    def test_is_number(self):
        from immlib import is_number
        import torch, numpy as np
        # is_number returns True if the argument is a scalar number; otherwise
        # it returns false.
        self.assertTrue(is_number(10))
        self.assertTrue(is_number(10.0))
        self.assertTrue(is_number(10.0 + 20.5j))
        self.assertTrue(is_number(True))
        self.assertTrue(is_number(np.array(10)))
        self.assertTrue(is_number(torch.tensor(10)))
        self.assertFalse(is_number([10]))
        self.assertFalse(is_number([[10]]))
        self.assertFalse(is_number([[[10]]]))
        self.assertFalse(is_number(np.array([10])))
        self.assertFalse(is_number(np.array([[10]])))
        self.assertFalse(is_number(np.array([[[10]]])))
        self.assertFalse(is_number(torch.tensor([10])))
        self.assertFalse(is_number(torch.tensor([[10]])))
        self.assertFalse(is_number(torch.tensor([[[10]]])))
        self.assertFalse(is_number('10'))
        self.assertFalse(is_number({'a':10}))
        self.assertFalse(is_number([1,2,3]))
    def test_to_number(self):
        from immlib import to_number
        import torch, numpy as np
        # to_number returns a scalar version of the given argument assuming that
        # the argument is like a scalar (see like_number).
        self.assertEqual(to_number(10), 10)
        self.assertEqual(to_number(10.0), 10.0)
        self.assertEqual(to_number(10.0 + 20.5j), 10.0 + 20.5j)
        self.assertEqual(to_number(True), True)
        self.assertEqual(to_number(np.array(10)), 10)
        self.assertEqual(to_number(torch.tensor(10)), 10)
        self.assertEqual(to_number([10]), 10)
        self.assertEqual(to_number([[10]]), 10)
        self.assertEqual(to_number([[[10]]]), 10)
        self.assertEqual(to_number(np.array([10])), 10)
        self.assertEqual(to_number(np.array([[10]])), 10)
        self.assertEqual(to_number(np.array([[[10]]])), 10)
        self.assertEqual(to_number(torch.tensor([10])), 10)
        self.assertEqual(to_number(torch.tensor([[10]])), 10)
        self.assertEqual(to_number(torch.tensor([[[10]]])), 10)
        with self.assertRaises(TypeError): to_number('10')
        with self.assertRaises(TypeError): to_number({'a':10})
        with self.assertRaises(TypeError): to_number([1,2,3])

    def test_is_numberdata(self):
        from immlib import is_numberdata
        import torch, numpy as np
        # is_numberdata returns True for numbers and False for non-numbers.
        self.assertTrue(is_numberdata(0))
        self.assertTrue(is_numberdata(5))
        self.assertTrue(is_numberdata(10.0))
        self.assertTrue(is_numberdata(-2.0 + 9.0j))
        self.assertTrue(is_numberdata(True))
        self.assertFalse(is_numberdata('abc'))
        self.assertFalse(is_numberdata('10'))
        self.assertFalse(is_numberdata(None))
        # Arrays and tensors are checked for their dtype to be such that their
        # elements are numbers.
        self.assertTrue(is_numberdata(np.array(5)))
        self.assertTrue(is_numberdata(np.array(10.0 + 2.0j)))
        self.assertTrue(is_numberdata(torch.tensor(5)))
        self.assertTrue(is_numberdata(torch.tensor(10.0 + 2.0j)))
        self.assertTrue(is_numberdata(torch.tensor([1,2,3])))
        self.assertTrue(is_numberdata(np.array([[-12.0]])))
        self.assertFalse(is_numberdata(np.array(['abc'])))
    def test_is_booldata(self):
        from immlib import is_booldata
        import torch, numpy as np
        # is_booldata returns True for booleans and False for non-booleans.
        self.assertFalse(is_booldata(0))
        self.assertFalse(is_booldata(5))
        self.assertFalse(is_booldata(10.0))
        self.assertFalse(is_booldata(-2.0 + 9.0j))
        self.assertTrue(is_booldata(True))
        self.assertFalse(is_booldata('abc'))
        self.assertFalse(is_booldata('10'))
        self.assertFalse(is_booldata(None))
        # Arrays and tensors are checked for their dtype to be such that their
        # elements are numbers.
        self.assertTrue(is_booldata(np.array(True)))
        self.assertFalse(is_booldata(np.array(10.0 + 2.0j)))
        self.assertFalse(is_booldata(torch.tensor(5)))
        self.assertFalse(is_booldata(torch.tensor(10.0 + 2.0j)))
        self.assertTrue(is_booldata(torch.tensor([True,False,False])))
        self.assertFalse(is_booldata(np.array([[12.0]])))
    def test_is_intdata(self):
        from immlib import is_intdata
        import torch, numpy as np
        # is_intdata returns True for integers and False for non-integers.
        self.assertTrue(is_intdata(0))
        self.assertTrue(is_intdata(5))
        self.assertFalse(is_intdata(10.0))
        self.assertFalse(is_intdata(-2.0 + 9.0j))
        self.assertTrue(is_intdata(True))
        self.assertFalse(is_intdata('abc'))
        self.assertFalse(is_intdata('10'))
        self.assertFalse(is_intdata(None))
        # Arrays and tensors are checked for their dtype to be such that their
        # elements are numbers.
        self.assertTrue(is_intdata(np.array(5)))
        self.assertFalse(is_intdata(np.array(10.0 + 2.0j)))
        self.assertTrue(is_intdata(torch.tensor(5)))
        self.assertFalse(is_intdata(torch.tensor(10.0 + 2.0j)))
        self.assertTrue(is_intdata(torch.tensor([1,2,3])))
        self.assertFalse(is_intdata(np.array([[12.0]])))
    def test_is_realdata(self):
        from immlib import is_realdata
        import torch, numpy as np
        # is_real returns True for reals and False for non-reals.
        self.assertTrue(is_realdata(0))
        self.assertTrue(is_realdata(5))
        self.assertTrue(is_realdata(10.0))
        self.assertTrue(is_realdata(True))
        self.assertFalse(is_realdata(-2.0 + 9.0j))
        self.assertFalse(is_realdata('abc'))
        self.assertFalse(is_realdata('10'))
        self.assertFalse(is_realdata(None))
        # Arrays and tensors are checked for their dtype to be such that their
        # elements are numbers.
        self.assertTrue(is_realdata(np.array(5)))
        self.assertFalse(is_realdata(np.array(10.0 + 2.0j)))
        self.assertTrue(is_realdata(torch.tensor(5)))
        self.assertFalse(is_realdata(torch.tensor(10.0 + 2.0j)))
        self.assertTrue(is_realdata(torch.tensor([1,2,3])))
        self.assertTrue(is_realdata(np.array([[12.0]])))
    def test_is_complexdata(self):
        from immlib import is_complexdata
        import torch, numpy as np
        # is_complexdata returns True for complexs and False for non-complexs.
        self.assertTrue(is_complexdata(0))
        self.assertTrue(is_complexdata(5))
        self.assertTrue(is_complexdata(10.0))
        self.assertTrue(is_complexdata(True))
        self.assertTrue(is_complexdata(-2.0 + 9.0j))
        self.assertFalse(is_complexdata('abc'))
        self.assertFalse(is_complexdata('10'))
        self.assertFalse(is_complexdata(None))
        # Arrays and tensors are checked for their dtype to be such that their
        # elements are numbers.
        self.assertTrue(is_complexdata(np.array(5)))
        self.assertTrue(is_complexdata(np.array(10.0 + 2.0j)))
        self.assertTrue(is_complexdata(torch.tensor(5)))
        self.assertTrue(is_complexdata(torch.tensor(10.0 + 2.0j)))
        self.assertTrue(is_complexdata(torch.tensor([1,2,3])))
        self.assertTrue(is_complexdata(np.array([[12.0]])))
