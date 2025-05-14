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
        self.assertTrue(is_number(np.array(5)))
        self.assertTrue(is_number(np.array(10.0 + 2.0j)))
        self.assertTrue(is_number(torch.tensor(5)))
        self.assertTrue(is_number(torch.tensor(10.0 + 2.0j)))
        # Arrays and tensors are *not* checked for their dtype to be such--that
        # is instead performed by is_numberdata.
        self.assertFalse(is_number(torch.tensor([1,2,3])))
        self.assertFalse(is_number(np.array([[-12.0]])))
        # Specific dtypes can also be tested for.
        self.assertTrue(is_number(0, dtype=int))
        self.assertTrue(is_number(5, dtype=int))
        self.assertTrue(is_number(10.0, dtype=float))
        self.assertTrue(is_number(-2.0 + 9.0j, dtype=complex))
        self.assertTrue(is_number(True, dtype=bool))
        self.assertFalse(is_number(0, dtype=bool))
        self.assertTrue(is_number(5, dtype=float))
        self.assertTrue(is_number(10.0, dtype=complex))
        self.assertFalse(is_number(-2.0 + 9.0j, dtype=int))
        self.assertTrue(is_number(True, dtype=float))
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
        # Make sure it throws errors when appropriate:
        with self.assertRaises(ValueError):
            is_number(0, dtype=str)
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
    def test_sparray_utils(self):
        import scipy.sparse as sps, numpy as np, torch, pint
        from immlib.util import (
            sparse_find, sparse_data, sparse_indices, sparse_layout,
            sparse_haslayout, sparse_tolayout, quant)
        from immlib import (units, quant)
        sparr = sps.csr_array(
            ([1.0, 0.5, 0.5, 0.2, 0.1],
             ([0, 0, 4, 5, 9], [4, 9, 4, 1, 8])),
            shape=(10, 10),
            dtype=float)
        sptns = torch.sparse_coo_tensor(
            torch.tensor([[0, 0, 4, 5, 9], [4, 9, 4, 1, 8]]),
            torch.tensor([1.0, 0.5, 0.5, 0.2, 0.1]),
            size=(10, 10),
            dtype=float)
        sptns = sptns.coalesce()
        # We can get a sparse layout from names or objects.
        for k in ('coo', 'csr', 'csc', 'bsr', 'bsc', 'dok', 'lil', 'dia'):
            sl = sparse_layout(k)
            self.assertEqual(k, sl.name)
            self.assertIs(sl, sparse_layout(sl))
        # We can also look up things by numpy array or torch tensor type.
        self.assertEqual('csr', sparse_layout(sparr).name)
        self.assertEqual('coo', sparse_layout(sptns).name)
        # sparse_layout works with quantities.
        self.assertEqual('csr', sparse_layout(quant(sparr, units.mm)).name)
        self.assertEqual('coo', sparse_layout(quant(sptns, units.mm)).name)
        # Unrecognized objects and names produce None:
        self.assertIs(None, sparse_layout('???'))
        self.assertIs(None, sparse_layout(object()))
        # We can convert between layouts:
        cooarr = sparse_tolayout(sparr, 'coo')
        self.assertEqual(cooarr.format, 'coo')
        self.assertTrue(np.array_equal(sparr.todense(), cooarr.todense()))
        csrtns = sparse_tolayout(sptns, 'csr')
        self.assertEqual(csrtns.layout, torch.sparse_csr)
        self.assertTrue(torch.equal(sptns.to_dense(), csrtns.to_dense()))
        q_sparr = quant(sparr, units.mm)
        self.assertIsInstance(q_sparr, pint.Quantity)
        q_cooarr = sparse_tolayout(q_sparr, 'coo')
        self.assertIsInstance(q_cooarr, pint.Quantity)
        self.assertTrue(np.array_equal(sparr.todense(), q_cooarr.m.todense()))
        f_sparr = sparr.copy()
        f_sparr.data.setflags(write=False)
        f_cooarr = sparse_tolayout(f_sparr, 'coo')
        self.assertTrue(np.array_equal(sparr.todense(), f_cooarr.todense()))
        self.assertFalse(f_cooarr.data.flags['WRITEABLE'])
        with self.assertRaises(TypeError):
            sparse_tolayout(object(), 'coo')
        # Check if a sparse object has a particular layout:
        self.assertTrue(sparse_haslayout(sparr, 'csr'))
        self.assertTrue(sparse_haslayout(cooarr, 'coo'))
        self.assertFalse(sparse_haslayout(sparr, 'coo'))
        self.assertFalse(sparse_haslayout(cooarr, 'csr'))
        self.assertTrue(sparse_haslayout(csrtns, 'csr'))
        self.assertTrue(sparse_haslayout(sptns, 'coo'))
        self.assertFalse(sparse_haslayout(csrtns, 'coo'))
        self.assertFalse(sparse_haslayout(sptns, 'csr'))
        self.assertTrue(isinstance(q_sparr, pint.Quantity))
        self.assertTrue(sps.issparse(q_sparr.m))
        self.assertTrue(sparse_haslayout(q_sparr, 'csr'))
        self.assertTrue(sparse_haslayout(q_cooarr, 'coo'))
        self.assertFalse(sparse_haslayout(object(), 'csr'))
        with self.assertRaises(ValueError):
            sparse_haslayout(sptns, '???')
        with self.assertRaises(ValueError):
            sparse_haslayout(sptns, object())
        # We can extract the various bits of data also:
        self.assertTrue(
            all(map(np.array_equal, sparse_find(sparr), sps.find(sparr))))
        self.assertTrue(
            np.array_equal(sparse_data(sparr), sparr.data))
        ii = np.stack(sparse_find(sparr)[:-1])
        self.assertTrue(
            np.array_equal(sparse_indices(sparr), ii))
        sfnd = sparse_find(sptns)
        ii = sptns.indices()
        vv = sptns.values()
        tfnd = tuple(ii) + (vv,)
        self.assertTrue(all(map(torch.equal, sfnd, tfnd)))
        self.assertTrue(
            torch.equal(vv, sparse_data(sptns)))
        self.assertTrue(
            torch.equal(ii, sparse_indices(sptns)))
        # These also work for quantities:
        q_sparr = quant(sparr, units.mm)
        self.assertIsInstance(q_sparr, pint.Quantity)
        self.assertTrue(
            all(np.array_equal(u.m if isinstance(u, pint.Quantity) else u, v) 
                for (u,v) in zip(sparse_find(q_sparr), sps.find(sparr))))
        q_spdat = sparse_data(q_sparr)
        self.assertIsInstance(q_spdat, pint.Quantity)
        self.assertTrue(
            np.array_equal(q_spdat.m, q_sparr.data))
        ii = np.stack(sparse_find(sparr)[:-1])
        self.assertTrue(
            np.array_equal(sparse_indices(q_sparr), ii))
        # They throw reasonable errors too:
        with self.assertRaises(TypeError):
            sparse_find('')
        with self.assertRaises(TypeError):
            sparse_indices('')
        with self.assertRaises(TypeError):
            sparse_data('')
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
        self.assertTrue(is_array(arr, shape=25))
        self.assertTrue(is_array(mtx, shape=(10,10)))
        self.assertFalse(is_array(arr, shape=(25,25)))
        self.assertFalse(is_array(mtx, shape=(10,)))
        self.assertTrue(is_array(np.array(100), shape=()))
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
        self.assertTrue(is_array(arr, numel=(25,26))) # Is numel 25 or 26?
        self.assertFalse(is_array(arr, numel=26))
        self.assertFalse(is_array(arr, numel=(24,26)))
        self.assertTrue(is_array(np.array(0), numel=1))
        self.assertTrue(is_array(np.array([0]), numel=1))
        self.assertTrue(is_array(np.array([[0]]), numel=1))
        # The frozen option can be used to test whether an array is frozen
        # or not. This is judged by the array's 'WRITEABLE' flag.
        self.assertFalse(is_array(arr, frozen=True))
        self.assertTrue(is_array(arr, frozen=False))
        self.assertFalse(is_array(mtx, frozen=True))
        self.assertTrue(is_array(mtx, frozen=False))
        with self.assertRaises(ValueError):
            is_array(mtx, frozen='fail')
        farr = arr.copy()
        farr.setflags(write=False)
        self.assertFalse(is_array(farr, frozen=False))
        self.assertTrue(is_array(farr, frozen=True))
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
        with self.assertRaises(ValueError):
            is_array(sp_mtx, sparse='???')
        with self.assertRaises(ValueError):
            is_array(sp_mtx, sparse=object())
        # Sparse and frozen can be tested together.
        self.assertTrue(is_array(arr, sparse=False, frozen=True))
        self.assertFalse(is_array(arr, sparse=False, frozen=False))
        self.assertFalse(is_array(arr, sparse=True, frozen=False))
        self.assertFalse(is_array(arr, sparse=True, frozen=True))
        self.assertFalse(is_array(sp_mtx, sparse=True, frozen=True))
        self.assertTrue(is_array(sp_mtx, sparse=True, frozen=False))
        self.assertFalse(is_array(sp_mtx, sparse=False, frozen=True))
        self.assertFalse(is_array(sp_mtx, sparse=False, frozen=False))
        sp_mtx.data.setflags(write=False)
        self.assertTrue(is_array(sp_mtx, sparse=True, frozen=True))
        self.assertFalse(is_array(sp_mtx, sparse=True, frozen=False))
        self.assertFalse(is_array(sp_mtx, sparse=False, frozen=True))
        self.assertFalse(is_array(sp_mtx, sparse=False, frozen=False))
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
        # We can also specify the units registry (Ellipsis means immlib.units).
        self.assertFalse(is_array(q_arr, unit='s', ureg=Ellipsis))
    def test_to_array(self):
        from immlib import (to_array, quant, is_quant, units, frozenarray)
        from numpy import (array, linspace, dot)
        from scipy.sparse import (csr_matrix, issparse)
        import torch, numpy as np, pint
        # We'll use a few objects throughout our tests, which we setup now.
        arr = linspace(0, 1, 25)
        tns = torch.linspace(0, 1, 25)
        mtx = dot(linspace(0, 1, 10)[:,None], linspace(0, 2, 10)[None,:])
        sp_mtx = csr_matrix(
            ([1.0, 0.5, 0.5, 0.2, 0.1],
             ([0, 0, 4, 5, 9], [4, 9, 4, 1, 8])),
            shape=(10, 10),
            dtype=float)
        sp_tns = torch.sparse_coo_tensor(
            torch.tensor([[0, 0, 4, 5, 9], [4, 9, 4, 1, 8]]),
            torch.tensor([1.0, 0.5, 0.5, 0.2, 0.1]),
            size=(10, 10),
            dtype=float)
        f_arr = frozenarray(arr)
        f_sp_mtx = frozenarray(sp_mtx)
        q_arr = quant(arr, 'mm')
        q_mtx = quant(arr, 'seconds')
        q_sp_mtx = quant(sp_mtx, 'kg')
        # For an object that is already a numpy array, any call that doesn't
        # request a copy and that doesn't change its parameters will return the
        # identical object.
        self.assertIs(arr, to_array(arr))
        self.assertIs(arr, to_array(arr, sparse=False, frozen=False))
        self.assertIs(arr, to_array(arr, quant=False))
        self.assertIs(f_arr, to_array(f_arr))
        self.assertIs(f_arr, to_array(f_arr, sparse=False, frozen=True))
        self.assertIs(f_arr, to_array(f_arr, quant=False))
        # to_array can be used to convert from tensors into arrays; the detach
        # parameter lets us automatically detach tensors from the gradient
        # system (this is the default) or raise an error if that would be
        # required (detach=False).
        self.assertIsInstance(to_array(tns), np.ndarray)
        gradtns = tns.clone().requires_grad_(True)
        self.assertIsInstance(to_array(gradtns), np.ndarray)
        self.assertTrue(np.isclose(to_array(tns), arr).all())
        self.assertTrue(np.isclose(to_array(gradtns), arr).all())
        with self.assertRaises(ValueError):
            to_array(gradtns, detach=False)
        # Sparse arrays/tensors should also convert fine.
        dn_tns = sp_tns.to_dense()
        x = to_array(sp_tns)
        self.assertTrue(issparse(x))
        self.assertEqual(x.format, 'coo')
        x = to_array(dn_tns, sparse='lil')
        self.assertTrue(issparse(x))
        self.assertEqual(x.format, 'lil')
        x = to_array(dn_tns, sparse=torch.sparse_csr)
        self.assertTrue(issparse(x))
        self.assertEqual(x.format, 'csr')
        x = to_array(dn_tns, sparse=False)
        self.assertFalse(issparse(x))
        self.assertTrue(
            np.array_equal(dn_tns.numpy(), x))
        x = to_array(sp_tns, sparse=False)
        self.assertTrue(
            np.array_equal(dn_tns.numpy(), x))
        with self.assertRaises(ValueError):
            to_array(dn_tns, sparse=object())
        with self.assertRaises(ValueError):
            to_array(dn_tns, sparse='???')
        x = to_array(sp_mtx, copy=False, dtype=complex)
        self.assertTrue(np.issubdtype(x.dtype, complex))
        self.assertTrue(
            np.all(np.isclose(x.todense().real, sp_mtx.todense())))
        self.assertTrue(
            np.all(np.abs(x.todense().imag) < 1e-9))
        sp_tns = sp_tns.coalesce()
        x = to_array(sp_tns, copy=False)
        self.assertTrue(
            np.shares_memory(x.data, sp_tns.values().detach().numpy()))
        x = to_array(sp_tns, copy=True)
        self.assertFalse(
            np.shares_memory(x.data, sp_tns.values().detach().numpy()))
        # If we change the parameters of the returned array, we will get
        # different (but typically equal) objects back.
        self.assertIsNot(arr, to_array(arr, frozen=True))
        self.assertIsNot(f_arr, to_array(f_arr, frozen=False))
        self.assertTrue(np.array_equal(arr, to_array(arr, frozen=True)))
        self.assertTrue(np.array_equal(f_arr, to_array(f_arr, frozen=False)))
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
        fsp_mtx = to_array(sp_mtx, frozen=True)
        self.assertTrue(np.array_equal(fsp_mtx.todense(), sp_mtx.todense()))
        self.assertFalse(fsp_mtx.data.flags['WRITEABLE'])
        tfsp_mtx = to_array(fsp_mtx, frozen=False)
        self.assertTrue(np.array_equal(tfsp_mtx.todense(), sp_mtx.todense()))
        self.assertTrue(tfsp_mtx.data.flags['WRITEABLE'])
        self.assertFalse(fsp_mtx.data.flags['WRITEABLE'])
        with self.assertRaises(ValueError):
            to_array(sp_mtx, frozen=object())
        # The quant argument can be used to enforce the return of quantities or
        # non-quantities, but you can't force a quantity without a unit:
        with self.assertRaises(ValueError):
            arr = to_array(arr, quant=True)
        # The unit parameter can be used to specify what unit to use.
        self.assertTrue(
            np.array_equal(q_arr.m, to_array(arr, quant=True, unit='mm').m))
        self.assertTrue(
            np.all(
                np.isclose(
                    to_array(q_arr, quant=True, unit='m').m,
                    to_array(arr, quant=True, unit='mm').m_as('m'))))
        self.assertTrue(
            np.all(
                np.isclose(
                    to_array(q_arr, quant=True, unit='m').m,
                    to_array(q_arr, quant=True, unit='mm').m_as('m'))))
        self.assertTrue(
            np.all(
                np.isclose(
                    to_array(arr, quant=False, unit='mm'),
                    to_array(q_arr, quant=False, unit='m')*1000)))
        # We can also use quant=False and a unit to extract the array in a
        # with a certain unit (like the mag function).
        e_arr = to_array(q_arr, quant=False, unit=...)
        self.assertIsInstance(e_arr, np.ndarray)
        self.assertTrue(np.all(np.isclose(e_arr, arr)))
        e_arr = to_array(q_arr, quant=False, unit='m')
        self.assertIsInstance(e_arr, np.ndarray)
        self.assertTrue(np.all(np.isclose(e_arr, arr/1000)))
        self.assertTrue(
            np.array_equal(q_arr.m, to_array(arr, quant=True, unit='mm').m))
        with self.assertRaises(ValueError):
            to_array(arr, quant=True, unit=Ellipsis)
        with self.assertRaises(ValueError):
            to_array(arr, quant=True, unit=None)
        with self.assertRaises(ValueError):
            to_array(q_arr, quant=True, unit=None)
        with self.assertRaises(ValueError):
            to_array(arr, quant=object())
        # We can also specify the units registry (Ellipsis means immlib.units).
        self.assertTrue(
            np.all(
                np.isclose(
                    to_array(q_arr, unit='m', ureg=Ellipsis).m,
                    q_arr.m / 1000.0)))
        # We can also use unit to extract a specific unit from a quantity.
        self.assertEqual(1000, to_array(quant(1, units.meter), unit='mm').m)
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
        from scipy.sparse import csr_matrix, csr_array
        import torch, numpy as np
        # By default, is_tensor() returns True for PyTorch tensors and
        # quantities whose magnitudes are PyTorch tensors.
        arr = torch.linspace(0, 1, 25)
        mtx = torch.mm(torch.linspace(0, 1, 10)[:,None],
                       torch.linspace(1, 2, 10)[None,:])
        sp_mtx = torch.sparse_coo_tensor(
            torch.tensor([[0, 0, 4, 5, 9],
                          [4, 9, 4, 1, 8]]),
            torch.tensor([1, 0.5, 0.5, 0.2, 0.1]),
            size=(10, 10),
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
        # You can also require a kind of sparse matrix.
        self.assertTrue(is_tensor(sp_mtx, sparse='coo'))
        self.assertFalse(is_tensor(sp_mtx, sparse='csc'))
        with self.assertRaises(ValueError):
            is_tensor(sp_mtx, sparse='???')
        with self.assertRaises(ValueError):
            is_tensor(sp_mtx, sparse=object())
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
        # We can also specify the units registry (Ellipsis means immlib.units).
        self.assertFalse(is_tensor(q_arr, unit='s', ureg=Ellipsis))
        # We can also test on torch data like device and requires_grad:
        self.assertTrue(is_tensor(arr, device='cpu'))
        self.assertFalse(is_tensor(arr, device='cuda'))
        self.assertTrue(is_tensor(arr, requires_grad=False))
        self.assertFalse(is_tensor(arr, requires_grad=True))
        gradtns = arr.clone().requires_grad_(True)
        self.assertFalse(is_tensor(gradtns, requires_grad=False))
        self.assertTrue(is_tensor(gradtns, requires_grad=True))
    def test_to_tensor(self):
        from immlib import (to_tensor, quant, is_quant, units)
        from immlib.util._numeric import torch__is_sparse
        from numpy import (linspace, dot)
        from scipy.sparse import (csr_array, issparse)
        import torch, numpy as np, pint
        # We'll use a few objects throughout our tests, which we setup now.
        arr = linspace(0, 1, 25)
        tns = torch.linspace(0, 1, 25)
        sp_arr = csr_array(
            ([1.0, 0.5, 0.5, 0.2, 0.1],
             ([0, 0, 4, 5, 9], [4, 9, 4, 1, 8])),
            shape=(10, 10),
            dtype=float)
        sp_tns = torch.sparse_coo_tensor(
            torch.tensor([[0, 0, 4, 5, 9], [4, 9, 4, 1, 8]]),
            torch.tensor([1.0, 0.5, 0.5, 0.2, 0.1]),
            size=(10, 10),
            dtype=float)
        q_arr = quant(arr, 'mm')
        q_tns = quant(tns, 'mm')
        q_sp_tns = quant(sp_tns, 'kg')
        q_sp_arr = quant(sp_arr, 'lb')
        # For an object that is already a numpy tensor, any call that doesn't
        # request a copy and that doesn't change its parameters will return the
        # identical object.
        self.assertIs(tns, to_tensor(tns))
        self.assertIs(tns, to_tensor(tns, quant=False))
        # to_tensor can be used to convert from arrayss into tensors
        self.assertIsInstance(to_tensor(arr), torch.Tensor)
        # Sparse tensors/tensors should also convert fine.
        dn_tns = sp_tns.to_dense()
        dn_arr = sp_arr.todense()
        x = to_tensor(sp_arr)
        self.assertTrue(torch__is_sparse(x))
        self.assertEqual(x.layout, torch.sparse_csr)
        x = to_tensor(dn_tns, sparse='coo')
        self.assertTrue(torch__is_sparse(x))
        self.assertEqual(x.layout, torch.sparse_coo)
        self.assertIsInstance(x, torch.Tensor)
        x = to_tensor(dn_tns, sparse=torch.sparse_csr)
        self.assertTrue(torch__is_sparse(x))
        self.assertEqual(x.layout, torch.sparse_csr)
        self.assertIsInstance(x, torch.Tensor)
        x = to_tensor(dn_tns, sparse=False)
        self.assertFalse(torch__is_sparse(x))
        self.assertTrue(torch.equal(dn_tns, x))
        self.assertIsInstance(x, torch.Tensor)
        x = to_tensor(sp_tns, sparse=False)
        self.assertTrue(torch.equal(dn_tns, x))
        self.assertIsInstance(x, torch.Tensor)
        x = to_tensor(sp_arr, sparse=False)
        self.assertTrue(torch.all(torch.isclose(dn_tns, x)))
        self.assertIsInstance(x, torch.Tensor)
        with self.assertRaises(ValueError):
            to_tensor(dn_tns, sparse=object())
        with self.assertRaises(ValueError):
            to_tensor(dn_tns, sparse='???')
        with self.assertRaises(ValueError):
            x = to_tensor(sp_tns, copy=False, dtype=complex)
        x = to_tensor(sp_tns, copy=None, dtype=complex)
        self.assertEqual(x.dtype, torch.complex128)
        self.assertTrue(
            torch.all(torch.isclose(x.to_dense().real, sp_tns.to_dense())))
        self.assertTrue(
            torch.all(torch.abs(x.to_dense().imag) < 1e-9))
        # We can also request that a copy be made like with np.array.
        self.assertIsNot(arr, to_tensor(arr, copy=True).numpy())
        self.assertTrue(torch.equal(tns, to_tensor(tns, copy=True)))
        self.assertTrue(
            np.shares_memory(arr.data, to_tensor(arr, copy=False).numpy().data))
        # The sparse flag can be used to convert to/from a sparse tensor.
        self.assertIsInstance(to_tensor(sp_tns, sparse=False), torch.Tensor)
        self.assertEqual(to_tensor(sp_tns, sparse=False).layout, torch.strided)
        self.assertTrue(
            torch.equal(to_tensor(sp_tns, sparse=False), sp_tns.to_dense()))
        self.assertTrue(torch__is_sparse(to_tensor(tns, sparse=True)))
        self.assertTrue(
            torch.equal(to_tensor(tns, sparse=True).to_dense(), tns))
        # The quant argument can be used to enforce the return of quantities or
        # non-quantities, but you can't force a quantity without a unit:
        with self.assertRaises(ValueError):
            arr = to_tensor(arr, quant=True, unit=None)
        # The unit parameter can be used to specify what unit to use.
        self.assertTrue(
            torch.equal(q_tns.m, to_tensor(tns, quant=True, unit='mm').m))
        self.assertTrue(
            torch.all(
                torch.isclose(
                    to_tensor(q_arr, quant=True, unit='m').m,
                    to_tensor(arr, quant=True, unit='mm').m_as('m'))))
        self.assertTrue(
            torch.all(
                torch.isclose(
                    to_tensor(q_arr, quant=True, unit='m').m,
                    to_tensor(q_arr, quant=True, unit='mm').m_as('m'))))
        self.assertTrue(
            torch.all(
                torch.isclose(
                    to_tensor(arr, quant=False, unit='mm'),
                    to_tensor(q_arr, quant=False, unit='m')*1000)))
        # We can also use quant=False and a unit to extract the tensor with a
        # certain unit (like the mag function).
        e_tns = to_tensor(q_tns, quant=False, unit=...)
        self.assertIsInstance(e_tns, torch.Tensor)
        self.assertTrue(torch.all(torch.isclose(e_tns, tns)))
        e_tns = to_tensor(q_tns, quant=False, unit='m')
        self.assertIsInstance(e_tns, torch.Tensor)
        self.assertTrue(torch.all(torch.isclose(e_tns, tns/1000)))
        self.assertTrue(
            torch.equal(q_tns.m, to_tensor(tns, quant=True, unit='mm').m))
        with self.assertRaises(ValueError):
            to_tensor(tns, quant=True, unit=Ellipsis)
        with self.assertRaises(ValueError):
            to_tensor(tns, quant=True, unit=None)
        with self.assertRaises(ValueError):
            to_tensor(q_tns, quant=True, unit=None)
        with self.assertRaises(ValueError):
            to_tensor(tns, quant=object())
        # We can also specify the units registry (Ellipsis means immlib.units).
        self.assertTrue(
            torch.all(
                torch.isclose(
                    to_tensor(q_tns, unit='m', ureg=Ellipsis).m,
                    q_tns.m / 1000.0)))
        # We can also use unit to extract a specific unit from a quantity.
        self.assertEqual(1000, to_tensor(quant(1, units.meter), unit='mm').m)
        # However, a non-quantity is always assumed to already have the units
        # requested, so converting it to a particular unit (but not converting
        # it to a quantity) results in the same object.
        self.assertIs(to_tensor(tns, quant=False, unit='mm'), tns)
        # If we simply request an tensor with a unit, without specifying that it
        # not be a quantity, we get a quantity back.
        self.assertIsInstance(to_tensor(tns, unit='mm'), pint.Quantity)
        # An error is raised if you try to request no units for a quantity.
        with self.assertRaises(ValueError):
            to_tensor(tns, quant=True, unit=None)
        # If we change the parameters of the returned array, we will get
        # different (but typically equal) objects back.
        self.assertTrue(torch.equal(tns, to_tensor(tns, requires_grad=True)))

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
        # ragged arrays are not like numbers:
        self.assertFalse(like_number([[1,2,3],[2,3]]))
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

    # The numapi Decorator #####################################################
    def test_numapi(self):
        from immlib.util import numapi
        import numpy as np, torch
        # Basic test:
        @numapi
        def l2_distance(pt1, pt2):
            "Calculates the L2 distance between two points."
            pass
        @l2_distance.array
        def _(pt1, pt2):
            return np.sqrt(np.sum((pt1 - pt2)**2, axis=0))
        @l2_distance.tensor
        def _(pt1, pt2):
            return torch.sqrt(torch.sum((pt1 - pt2)**2, axis=0))
        tns = l2_distance(torch.tensor([0,0]), [0,1])
        self.assertIsInstance(tns, torch.Tensor)
        self.assertEqual(tns, 1.0)
        tns = l2_distance([0,0], torch.tensor([0,1]))
        self.assertIsInstance(tns, torch.Tensor)
        self.assertEqual(tns, 1.0)
        arr = l2_distance([0,0], [0,1])
        self.assertIsInstance(arr, float)
        self.assertEqual(arr, 1.0)

    # The tensor_args, array_args, and numeric_args decorators ################
    def test_tensor_args(self):
        from immlib.util import tensor_args
        import numpy as np, torch
        # Without any arguments, it should just auto-tensorify the args.
        @tensor_args
        def test1(a, b, c='test'):
            return (torch.is_tensor(a), torch.is_tensor(b), torch.is_tensor(c))
        # By default it shouldn't convert things like strings or dicts into
        # tensors.
        (a, b, c) = test1(10, {'a':12, 'b':13})
        self.assertTrue(a)
        self.assertFalse(b)
        self.assertFalse(c)
        # But lists, numbers, and compatible arrays should get converted.
        (a, b, c) = test1(5.5, [10.1, 12.7], c=np.linspace(0,1,5))
        self.assertTrue(a)
        self.assertTrue(b)
        self.assertTrue(c)
        # Tensors should be passed through.
        (a, b, c) = test1(5.5, [10.1, 12.7], c=torch.linspace(0,1,5))
        self.assertTrue(a)
        self.assertTrue(b)
        self.assertTrue(c)
        # With the keep_arrays argument set to True, non-tensor arguments get
        # converted back to arrays if all the arguments were non-tensors.
        @tensor_args(keep_arrays=True)
        def test2(a, b, c='test'):
            return (torch.sqrt(a**2 + b**2), c)
        (x, y) = test2(10.0, [11.1, 12.2])
        self.assertFalse(torch.is_tensor(x))
        self.assertFalse(torch.is_tensor(y))
        # If there were any tensors, the results should remain as tensors.
        (x, y) = test2(torch.tensor(10.0), [11.1, 12.2])
        self.assertTrue(torch.is_tensor(x))
        self.assertFalse(torch.is_tensor(y))  # Still a string 'test' here.
        # With named arguments in the tensor_args arguments, only those args
        # are converted or considered.
        @tensor_args('a', keep_arrays=True)
        def test3(a, b, c='test'):
            return (torch.sqrt(a**2 + b**2), c)
        with self.assertRaises(TypeError):
            (x, y) = test3(torch.tensor(10.0), [11.1, 12.2])
        with self.assertRaises(TypeError):
            (x, y) = test3(10.0, [11.1, 12.2])
        (x, y) = test3(10.0, torch.tensor([11.1, 12.2]))
        # Because it skips parameter b, it doesn't consider this example to be
        # a case where the tensors should be maintained (keep_arrays indicates
        # that if any of the converted parameters were tensors it should not
        # convert results into arrays, but in this case parameter b isn't one
        # of the named parameters, so all that the conversion algorithm sees is
        # that parameter a isn't a tensor).
        self.assertFalse(torch.is_tensor(x))
        self.assertTrue(isinstance(y, str))
        (x, y) = test3(torch.tensor(10.0), torch.tensor([11.1, 12.2]))
        self.assertTrue(torch.is_tensor(x))
        self.assertTrue(isinstance(y, str))
    def test_array_args(self):
        from immlib.util import array_args
        import numpy as np, torch
        # Without any arguments, it should just auto-array all the args.
        @array_args
        def test1(a, b, c='test'):
            return (type(a), type(b), type(c))
        # Even strings and dictionaries get converted into arrays.
        (a, b, c) = test1(10, {'a':12, 'b':13})
        self.assertIs(a, np.ndarray)
        self.assertIs(a, np.ndarray)
        self.assertIs(a, np.ndarray)
        # Even tensors should be converted down when possible.
        (a, b, c) = test1(5.5, [10.1, 12.7], c=torch.linspace(0,1,5))
        self.assertIs(a, np.ndarray)
        self.assertIs(b, np.ndarray)
        self.assertIs(c, np.ndarray)
        # Tensors should be passed through.
        (a, b, c) = test1(5.5, [10.1, 12.7], c=torch.linspace(0,1,5))
        self.assertTrue(a)
        self.assertTrue(b)
        self.assertTrue(c)
        # With named arguments in the tensor_args arguments, only those args
        # are converted or considered.
        @array_args('a')
        def test3(a, b, c='test'):
            if not isinstance(a, np.ndarray):
                raise TypeError()
            if torch.is_tensor(b):
                raise TypeError()
            return (np.sqrt(a**2 + b**2), c)
        with self.assertRaises(TypeError):
            (x, y) = test3(10.0, torch.tensor([11.1, 12.2]))
        with self.assertRaises(TypeError):
            # Fails because you can't run [11.1, 12.2]**2.
            (x, y) = test3(10.0, [11.1, 12.2])
        (x, y) = test3(torch.tensor(10.0), np.array([11.1, 12.2]))
        self.assertFalse(torch.is_tensor(x))
        self.assertTrue(isinstance(y, str))
    def test_numeric_args(self):
        from immlib.util import numeric_args
        import numpy as np, torch
        # Without any arguments, it should just auto-array all the args.
        @numeric_args
        def test1(a, b, c='test'):
            return (type(a), type(b), type(c))
        # Even strings and dictionaries get converted into arrays.
        (a, b, c) = test1(10, {'a':12, 'b':13})
        self.assertIs(a, np.ndarray)
        self.assertIs(a, np.ndarray)
        self.assertIs(a, np.ndarray)
        # If there are tensors, then all args should be converted into tensors.
        (a, b, c) = test1(5.5, [10.1, 12.7], c=torch.linspace(0,1,5))
        self.assertIs(a, torch.Tensor)
        self.assertIs(b, torch.Tensor)
        self.assertIs(c, torch.Tensor)
        # With named arguments in the tensor_args arguments, only those args
        # are converted or considered.
        @numeric_args('a')
        def test3(a, b, c='test'):
            return (np.sqrt(a**2 + b**2), c)
        (x, y) = test3(10.0, torch.tensor([11.1, 12.2]))
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, str)
        with self.assertRaises(TypeError):
            # Fails because you can't run [11.1, 12.2]**2.
            (x, y) = test3(10.0, [11.1, 12.2])
        (x, y) = test3(torch.tensor(10.0), np.array([11.1, 12.2]))
        self.assertTrue(torch.is_tensor(x))
        self.assertTrue(isinstance(y, str))
