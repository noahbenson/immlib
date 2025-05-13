# -*- coding: utf-8 -*-
################################################################################
# immlib/test/workflow/test_core.py
#
# Tests of the core workflow module in immlib: i.e., tests for the code in the
# immlib.workflow._core module.


# Dependencies #################################################################

from unittest import TestCase

class TestWorkflowCore(TestCase):
    """Tests the immlib.workflow._core module."""
    def test_calc(self):
        from immlib.workflow import (calc, is_calc, is_calcfn)
        from pcollections import (ldict, pdict)
        # The calc decorator creates calculation objects.
        @calc
        def result(input_1, input_2=None):
            """Calculation for a result from input_1 and input_2.

            Returns a single value, `'result'`, which is a list whose length is
            `input_1` and whose elements are all `input_2`.
            
            Inputs
            ------
            input_1 : int
                The number of elements to include in the result.
            input_2 : object
                The object to put in the list.

            Outputs
            -------
            result : list
                A list of `input_1` occurrences of `input_2`.
            """
            return ([input_2] * input_1,)
        self.assertTrue(is_calcfn(result))
        self.assertTrue(is_calc(result.calc))
        # Calculation objects have a number of members that keep track of the
        # meta-data of the calculation.
        # First is the name of the calculation--this is the name of the
        # function.
        c = result.calc
        self.assertEqual(c.name, 'immlib.test.workflow.test_core.result')
        # The inputs of the calculation are a set of the inputs of the function.
        self.assertEqual(c.inputs, set(['input_1', 'input_2']))
        # The default values of the inputs are stored in the defaults member.
        self.assertEqual(c.defaults, {'input_2': None})
        # The outputs are a tuple of the output names. For a calc without
        # explicitly listed outputs has only one output, its name.
        self.assertEqual(c.outputs, ('result',))
        # The input documentation is stored in the input_docs member.
        self.assertIn('input_1', c.input_docs)
        self.assertIn('input_2', c.input_docs)
        self.assertEqual(len(c.input_docs), 2)
        self.assertIn('The number of elements to include in the result.',
                      c.input_docs['input_1'])
        self.assertIn('The object to put in the list.',
                      c.input_docs['input_2'])
        # The output documentation is stored in the output_docs member.
        self.assertIn('result', c.output_docs)
        self.assertIn('A list of `input_1` occurrences of `input_2`.',
                      c.output_docs['result'])
        self.assertEqual(len(c.output_docs), 1)
        # The calculation can be called using its normal signature.
        self.assertEqual(c(1), {'result': [None]})
        self.assertEqual(c(2, 0), {'result': [0, 0]})
        # The call method is basically an alias for the __call__ method.
        self.assertEqual(c.call(1), {'result': [None]})
        self.assertEqual(c.call(2, 0), {'result': [0, 0]})
        # The call can also be forced to be either eager or lazy--when lazy,
        # the return value is a lazy dict, and the calc isn't actually run until
        # the values are requested; when eager, the call is run right away, and
        # the return value is an frozendict instead of a lazydict.
        self.assertIsInstance(c.eager_call(1), pdict)
        self.assertEqual(c.eager_call(1), {'result': [None]})
        self.assertEqual(c.eager_call(2, 0), {'result': [0, 0]})
        self.assertIsInstance(c.lazy_call(1), ldict)
        self.assertEqual(c.lazy_call(1), {'result': [None]})
        self.assertEqual(c.lazy_call(2, 0), {'result': [0, 0]})
        # It can also be called using the mapcall method.
        m1 = dict(input_1=1)
        m2 = dict(input_1=2, input_2=0)
        self.assertEqual(c.mapcall(m1), {'result': [None]})
        self.assertEqual(c.mapcall(m2), {'result': [0,0]})
        # These can also be lazy or eager.
        self.assertIsInstance(c.eager_mapcall(m1), pdict)
        self.assertEqual(c.eager_mapcall(m1), {'result': [None]})
        self.assertEqual(c.eager_mapcall(m2), {'result': [0, 0]})
        self.assertIsInstance(c.lazy_mapcall(m1), ldict)
        self.assertEqual(c.lazy_mapcall(m1), {'result': [None]})
        self.assertEqual(c.lazy_mapcall(m2), {'result': [0, 0]})
        # Calculations can have multiple outputs as well as multiple inputs.
        @calc('out1', 'out2', 'out3')
        def sample_calc(in1, in2, in3):
            return (in1 + 1, in2 + 2, in3 + 3)
        res = sample_calc.calc(1, 2, 3)
        self.assertIsInstance(res, ldict)
        self.assertEqual(len(res), 3)
        self.assertEqual(res['out1'], 2)
        self.assertEqual(res['out2'], 4)
        self.assertEqual(res['out3'], 6)
        # New calcs can be made that change the names of the calculation
        # variables (the inputs and outputs) using the tr (translate) method.
        sample_tr = sample_calc.calc.rename_keys(out1='x', out2='y', in3='z')
        self.assertEqual(sample_tr.outputs, ('x', 'y', 'out3'))
        self.assertEqual(len(sample_tr.inputs), 3)
        self.assertIn('in1', sample_tr.inputs)
        self.assertIn('in2', sample_tr.inputs)
        self.assertIn('z', sample_tr.inputs)
        res = sample_tr.mapcall({'in1':1, 'in2':2, 'z':3})
        self.assertIsInstance(res, ldict)
        self.assertEqual(len(res), 3)
        self.assertEqual(res['x'], 2)
        self.assertEqual(res['y'], 4)
        self.assertEqual(res['out3'], 6)
    def test_is_calc(self):
        from immlib.workflow import (calc, is_calcfn)
        @calc
        def result(input_1, input_2=None):
            return ([input_2] * input_1,)
        # is_calc(x) is just an alias for isinstance(x, calc).
        self.assertTrue(is_calcfn(result))
        self.assertFalse(is_calcfn(lambda x:x))
        self.assertEqual(result.calc(2,0)['result'], [0, 0])
    def test_plan(self):
        import numpy as np
        from immlib.workflow import (calc, plan, plandict)
        # Plans are just collections of calc objects, each of which gets built
        # into a directed acyclic graph of calculation dependencies.
        @calc('weights', lazy=False)
        def normal_pdf(x, mu=0, std=1):
            """Calculates the probability densities for a normal distribution.

            Inputs
            ------
            x : array-like
                The input values at which to calculate the normal PDF.
            mu : number, optional
                The mean of the normal distribution; the default is 0.
            std : number, optional
                The standard deviation of the distribution; the default is 1.

            Outputs
            -------
            weights : array-like
                The probability densities of the normal distribution at the
                given set of values in `x`.
            """
            w = np.exp(-0.5 * ((x - mu)/std)**2) / (np.sqrt(2*np.pi) * std)
            return (w,)
        @calc('mean')
        def weighted_mean(x, weights):
            """Calculates the weighted mean.

            Inputs
            ------
            x : array-like
                The values to be averaged.
            weights : array-like
                The weights of the values in `x`.

            Outputs
            -------
            mean : number
                The weighted mean of the inputs.
            """
            mean = np.sum(x * weights) / np.sum(weights)
            return (mean,)
        # Filter calculations can be used to update the input variables to a
        # plan--they are calc units that accept only 1 input and that return
        # same input.
        @calc('x')
        def filter_x(x):
            x = np.asarray(x)
            assert len(x.shape) == 1, "x must be a vector"
            assert np.issubdtype(x.dtype, np.number), "x must be numeric"
            return (x,)
        # The calculations are given names (keys) and put together in a plan.
        nwm = plan(
            weights_step=normal_pdf,
            mean_step=weighted_mean,
            filter_x=filter_x)
        # This creates a plan object, which stores these computations.
        self.assertIsInstance(nwm, plan)
        # The plan keeps track lots of meta-data, including an agglomeration of
        # the meta-data of its calculations.
        self.assertEqual(nwm.inputs, set(['x', 'mu', 'std']))
        self.assertEqual(nwm.outputs, set(['weights', 'mean']))
        self.assertEqual(nwm.defaults, {'mu': 0, 'std': 1})
        # We can provide a plan with its parameters in order to create a
        # plandict, which is a lazydict that agglomerates all of the input and
        # output values of all the calculations.
        pd = nwm(x=[-1.0, 1.0, 2.0, 8.5], mu=1.5)
        self.assertIsInstance(pd, plandict)
        self.assertEqual(len(pd), 5)
        # In this case, because we have a non-lazy calc (normal_pdf), all of
        # that calc's inputs are also automatically ready (this is not a
        # surprise--its other inputs are plain params so are not lazy objects).
        self.assertFalse(pd.is_lazy('mu'))
        self.assertFalse(pd.is_lazy('std'))
        self.assertTrue(pd.is_ready('mu'))
        self.assertTrue(pd.is_ready('std'))
        # The weights outputs should be ready because it was declared to be
        # non-lazy; the mean should remain lazy, though.
        self.assertTrue(pd.is_lazy('weights'))
        self.assertFalse(pd.is_ready('weights'))
        self.assertTrue(pd.is_lazy('mean'))
        self.assertFalse(pd.is_ready('mean'))
        # It will have converted the x value into an array.
        self.assertIsInstance(pd['x'], np.ndarray)
        self.assertTrue(np.array_equal(pd['x'], [-1, 1, 2, 8.5]))
        self.assertEqual(pd['mu'], 1.5)
        self.assertEqual(pd['std'], 1)
        self.assertAlmostEqual(pd['mean'], 1.4392777559)
        # We can update the plandict by making a new one.
        pd2 = plandict(pd, x=[0, 1, 2, 8.5])
        self.assertIsInstance(pd2['x'], np.ndarray)
        self.assertTrue(np.array_equal(pd2['x'], [0, 1, 2, 8.5]))
        self.assertEqual(pd2['mu'], 1.5)
        self.assertEqual(pd2['std'], 1)
        self.assertAlmostEqual(pd2['mean'], 1.266956394834)
        pd2 = plandict(pd, mu=2.5)
        self.assertIsInstance(pd2['x'], np.ndarray)
        self.assertTrue(np.array_equal(pd2['x'], [-1, 1, 2, 8.5]))
        self.assertEqual(pd2['mu'], 2.5)
        self.assertEqual(pd2['std'], 1)
        self.assertAlmostEqual(pd2['mean'], 1.726118628968)
        # We can also make a transient plandict...
        tpd = pd.transient()
        tpd['x'] = [0, 1, 2, 8.5]
        self.assertIsInstance(tpd['x'], np.ndarray)
        self.assertTrue(np.array_equal(tpd['x'], [0, 1, 2, 8.5]))
        self.assertEqual(tpd['mu'], 1.5)
        self.assertEqual(tpd['std'], 1)
        self.assertAlmostEqual(tpd['mean'], 1.266956394834)
        tpd = pd.transient()
        tpd['mu'] = 2.5
        self.assertIsInstance(tpd['x'], np.ndarray)
        self.assertTrue(np.array_equal(tpd['x'], [-1, 1, 2, 8.5]))
        self.assertEqual(tpd['mu'], 2.5)
        self.assertEqual(tpd['std'], 1)
        self.assertAlmostEqual(tpd['mean'], 1.726118628968)
        # Since we marked the filter as non-lazy, it should raise errors when
        # the plan is fulfilled.
        with self.assertRaises(RuntimeError): nwm(x=10)
        # We should also make sure the documentation is getting loaded
        # correctly.
        for k in ('x', 'mu', 'std'):
            self.assertIn(k, nwm.inputs)
            self.assertIn(k, nwm.input_docs)
        for k in ('mean', 'weights'):
            self.assertIn(k, nwm.output_docs)
    def test_multifilter(self):
        """Tests the ability of plans to contain multi-input filters."""
        import numpy as np
        from immlib.workflow import calc, plan
        @calc('a', 'b', 'c', lazy=False)
        def filter_bccoords(a=None, b=None, c=None):
            n_given = 3 - (int(a is None) + int(b is None) + int(c is None))
            if n_given < 2:
                raise ValueError("at least two of a, b, and c must be provided")
            elif n_given == 2:
                if a is None:
                    a = 1 - (b + c)
                elif b is None:
                    b = 1 - (a + c)
                elif c is None:
                    c = 1 - (a + b)
            return (a, b, c)
        @calc('a_coords', 'b_coords', 'c_coords', lazy=False)
        def filter_tricoords(a_coords, b_coords, c_coords):
            a_coords = np.array(a_coords)
            b_coords = np.array(b_coords)
            c_coords = np.array(c_coords)
            a_coords.flags.writeable = False
            b_coords.flags.writeable = False
            c_coords.flags.writeable = False
            return (a_coords, b_coords, c_coords)
        @calc('coords')
        def calc_coords(a_coords, b_coords, c_coords, a, b, c):
            return (a*a_coords + b*b_coords + c*c_coords,)
        p = plan(
            bcfilter=filter_bccoords,
            trifilter=filter_tricoords,
            coords=calc_coords)
        # The main thing is that this plan should not have any trouble filling
        # in the three values.
        tri = {'a_coords': (0,0), 'b_coords':(0,1), 'c_coords': (1,0)}
        u = p(a=0.25, b=0.25, **tri)
        self.assertEqual(u['c'], 0.5)
        u = p(a=0.25, c=0.25, **tri)
        self.assertEqual(u['b'], 0.5)
        u = p(c=0.25, b=0.25, **tri)
        self.assertEqual(u['a'], 0.5)
    def test_pathcache(self):
        """Tests that the pathcache argument works correctly."""
        # We make a temporary cache path directory for all of this:
        from tempfile import TemporaryDirectory
        from joblib import Memory
        from immlib import calc, plan
        from immlib.workflow import to_calc
        with TemporaryDirectory() as tmpdir:
            self.pc_runcount = 0
            @calc('outputval1', 'outputval2', pathcache=tmpdir)
            def test_cache1(inputval1, inputval2):
                self.pc_runcount = self.pc_runcount + 1
                return (inputval1 // inputval2, inputval1 % inputval2)
            d = test_cache1.calc(10, 3)
            self.assertEqual(d['outputval1'], 3)
            self.assertEqual(d['outputval2'], 1)
            self.assertEqual(self.pc_runcount, 1)
            d = test_cache1.calc(10, 3)
            self.assertEqual(d['outputval1'], 3)
            self.assertEqual(d['outputval2'], 1)
            self.assertEqual(self.pc_runcount, 1)
            d = test_cache1.calc(10, 5)
            self.assertEqual(d['outputval1'], 2)
            self.assertEqual(d['outputval2'], 0)
            self.assertEqual(self.pc_runcount, 2)
            d = test_cache1.calc(10, 5)
            self.assertEqual(d['outputval1'], 2)
            self.assertEqual(d['outputval2'], 0)
            self.assertEqual(self.pc_runcount, 2)
            # We can also use pathcache=True and pass the tmpdir as a cache_path
            # parameter.
            self.pc_runcount = 0
            @calc('outputval1', 'outputval2', pathcache=True)
            def test_cache2(inputval1, inputval2):
                self.pc_runcount = self.pc_runcount + 1
                return (inputval1 // inputval2, inputval1 % inputval2)
            # The pathcache gets noted
            c = to_calc(test_cache2)
            self.assertTrue(c.pathcache)
            # Now make a plan.
            p = plan(test=test_cache2)
            # No cache_path, no caching.
            d = p(inputval1=10, inputval2=3)
            self.assertEqual(d['cache_path'], None)
            self.assertEqual(d['outputval1'], 3)
            self.assertEqual(d['outputval2'], 1)
            self.assertEqual(self.pc_runcount, 1)
            d = p(inputval1=10, inputval2=3)
            self.assertEqual(d['cache_path'], None)
            self.assertEqual(d['outputval1'], 3)
            self.assertEqual(d['outputval2'], 1)
            self.assertEqual(self.pc_runcount, 2)
            # With a cache_path, it gets cached.
            d = p(inputval1=10, inputval2=3, cache_path=tmpdir)
            self.assertEqual(d['cache_path'], tmpdir)
            self.assertEqual(d['outputval1'], 3)
            self.assertEqual(d['outputval2'], 1)
            self.assertEqual(self.pc_runcount, 3)
            d = p(inputval1=10, inputval2=3, cache_path=tmpdir)
            self.assertEqual(d['cache_path'], tmpdir)
            self.assertEqual(d['outputval1'], 3)
            self.assertEqual(d['outputval2'], 1)
            self.assertEqual(self.pc_runcount, 3)
            # We can also test the version of this where we include cache_path
            # as an input parameter.
            self.pc_runcount = 0
            @calc('outputval1', 'outputval2', 'out_cpath', pathcache=True)
            def test_cache3(inputval1, inputval2, cache_path=None):
                self.pc_runcount = self.pc_runcount + 1
                return (inputval1 // inputval2, inputval1 % inputval2,
                        cache_path)
            p = plan(test=test_cache3)
            # No cache_path, no caching.
            d = p(inputval1=10, inputval2=3)
            self.assertEqual(d['outputval1'], 3)
            self.assertEqual(d['outputval2'], 1)
            self.assertEqual(d['out_cpath'], None)
            self.assertEqual(self.pc_runcount, 1)
            d = p(inputval1=10, inputval2=3)
            self.assertEqual(d['outputval1'], 3)
            self.assertEqual(d['outputval2'], 1)
            self.assertEqual(d['out_cpath'], None)
            self.assertEqual(self.pc_runcount, 2)
            # With a cache_path, it gets cached.
            d = p(inputval1=10, inputval2=3, cache_path=tmpdir)
            self.assertEqual(d['outputval1'], 3)
            self.assertEqual(d['outputval2'], 1)
            self.assertEqual(d['out_cpath'], tmpdir)
            self.assertEqual(self.pc_runcount, 3)
            d = p(inputval1=10, inputval2=3, cache_path=tmpdir)
            self.assertEqual(d['outputval1'], 3)
            self.assertEqual(d['outputval2'], 1)
            self.assertEqual(d['out_cpath'], tmpdir)
            self.assertEqual(self.pc_runcount, 3)
    def test_decstack(self):
        "Tests the ability to stack calc decorations with other decorators."
        from immlib.workflow import calc, plan
        from immlib.util import tensor_args
        import numpy as np, torch
        # Create a calculation that computes a normalized vector `u` and a
        # length `xlen` given an unnormalized vector `x`.
        @tensor_args(keep_arrays=True)
        @calc('u', 'xlen')
        def normalize_vector(x):
            xlen = torch.sqrt(torch.sum(x**2))
            u = x / xlen
            return (u, xlen)
        # Create another calculation that finds the signed distance between a
        # point `y` and the vector `x`, as well as the point of intersection.
        @calc('distance', 'intersection')
        @tensor_args(keep_arrays=True)
        def point_vec_intersection(u, y):
            d = torch.dot(u, y)
            return (d, u*d)
        p = plan(step1=normalize_vector, step2=point_vec_intersection)
        pd = p(x=[0.0, 1.0], y=[1.0, 1.0])
        self.assertEqual(pd['distance'], 1.0)
        self.assertIsInstance(pd['u'], np.ndarray)
    def test_tplandict(self):
        "Tests the tplandict type."
        pass
