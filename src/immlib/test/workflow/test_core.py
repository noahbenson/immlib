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
        from immlib.workflow import calc, plan, plandict, is_tplandict
        @calc('y')
        def add_one(x):
            return x + 1
        @calc('z')
        def double_y(y):
            return 2 * y
        pd = plan(step1=add_one, step2=double_y)(x=1)
        td = pd.transient()
        self.assertTrue(is_tplandict(td))
        self.assertEqual(td['z'], 4)
        # Setting an input updates the downstream values.
        td['x'] = 2
        self.assertEqual(td['y'], 3)
        self.assertEqual(td['z'], 6)
        self.assertEqual(td.get('z'), 6)
        # Only inputs may be set.
        with self.assertRaises(ValueError):
            td['y'] = 10
        # Items can't be removed.
        with self.assertRaises(TypeError):
            del td['x']
        for method in ('clear', 'popitem'):
            with self.assertRaises(TypeError):
                getattr(td, method)()
        with self.assertRaises(TypeError):
            td.pop('x')
        # Converting back gives a plandict with the updated values.
        pd2 = td.persistent()
        self.assertIsInstance(pd2, plandict)
        self.assertEqual(dict(pd2), {'x': 2, 'y': 3, 'z': 6})
        # The original plandict is unchanged.
        self.assertEqual(dict(pd), {'x': 1, 'y': 2, 'z': 4})
    def test_plandict_no_delete(self):
        "Tests that items can't be removed from a plandict."
        from immlib.workflow import calc, plan
        @calc('y')
        def add_one(x):
            return x + 1
        pd = plan(step1=add_one)(x=1)
        with self.assertRaises(TypeError):
            pd.delete('x')
        with self.assertRaises(TypeError):
            pd.pop('y')
        with self.assertRaises(TypeError):
            pd.popitem()
        with self.assertRaises(TypeError):
            pd.clear()
        self.assertEqual(dict(pd), {'x': 1, 'y': 2})
    def test_plan_error(self):
        "Tests that failed plan calculations raise PlanError."
        from immlib.workflow import calc, plan, PlanError
        from pcollections import lazy, LazyError
        @calc('y')
        def checked_add_one(x):
            if x < 0:
                raise ValueError("x must be non-negative")
            return x + 1
        @calc('z')
        def double_y(y):
            return 2 * y
        @calc('w', lazy=False)
        def required_y(y):
            return y
        p = plan(step1=checked_add_one, step2=double_y)
        pd = p(x=-1)
        # The error names the requested key and the failing calc, and its
        # cause is the original exception.
        for _ in range(2):
            with self.assertRaises(PlanError) as cm:
                pd['z']
            err = cm.exception
            self.assertIsInstance(err, LazyError)
            self.assertIsInstance(err.__cause__, ValueError)
            self.assertIsInstance(err.root_cause, ValueError)
            self.assertTrue(err.__suppress_context__)
            self.assertEqual(err.key, 'z')
            self.assertIs(err.calc, checked_add_one.calc)
            msg = str(err)
            self.assertIn("'z'", msg)
            self.assertIn(checked_add_one.calc.name, msg)
            self.assertIn("test_core.py", msg)
            self.assertIn("x must be non-negative", msg)
        # get, items, and dict() raise the same way.
        with self.assertRaises(PlanError):
            pd.get('y')
        with self.assertRaises(PlanError):
            dict(pd)
        with self.assertRaises(PlanError):
            list(pd.values())
        # Values that don't depend on the failure are still available.
        self.assertEqual(pd['x'], -1)
        self.assertEqual(pd.get('missing', 10), 10)
        # A failing lazy input is reported as an input failure.
        pd = p(x=lazy(lambda: 1/0))
        with self.assertRaises(PlanError) as cm:
            pd['z']
        self.assertIsNone(cm.exception.calc)
        self.assertIn("input 'x'", str(cm.exception))
        self.assertIsInstance(cm.exception.__cause__, ZeroDivisionError)
        with self.assertRaises(PlanError) as cm:
            pd['x']
        self.assertIsInstance(cm.exception.__cause__, ZeroDivisionError)
        # Required (non-lazy) calculations fail at construction.
        p = plan(step1=checked_add_one, step2=required_y)
        with self.assertRaises(PlanError) as cm:
            p(x=-1)
        self.assertIsNone(cm.exception.key)
        self.assertIs(cm.exception.calc, checked_add_one.calc)
        pd = p(x=1)
        with self.assertRaises(PlanError):
            pd.set('x', -1)
        td = pd.transient()
        with self.assertRaises(PlanError):
            td['x'] = -1
        # tplandict reads raise PlanError too.
        td = plan(step1=checked_add_one, step2=double_y)(x=1).transient()
        td['x'] = -2
        with self.assertRaises(PlanError) as cm:
            td['z']
        self.assertIsInstance(cm.exception.__cause__, ValueError)
    def test_plan_error_filtered_plan(self):
        "Tests PlanError for plans whose calcs are renamed internally."
        from immlib.workflow import calc, plan, PlanError
        @calc('data', lazy=False)
        def filter_data(data):
            return (list(data),)
        @calc('first')
        def first_over_second(data):
            if data[1] == 0:
                raise ZeroDivisionError("second element is zero")
            return data[0] / data[1]
        pd = plan(filt=filter_data, ratio=first_over_second)(data=[1, 0])
        with self.assertRaises(PlanError) as cm:
            pd['first']
        # The error reports the calc the user created, not the plan's
        # internal copy of it.
        self.assertIs(cm.exception.calc, first_over_second.calc)
        self.assertNotIn('.rename', str(cm.exception))
        self.assertIn(first_over_second.calc.name, str(cm.exception))
    def test_calc_with_caches(self):
        "Tests calc.with_lrucache and calc.with_pathcache."
        import tempfile
        from immlib.workflow import calc
        runs = []
        @calc('y')
        def add_one(x):
            runs.append(x)
            return x + 1
        c = add_one.calc
        c2 = c.with_lrucache(10)
        self.assertIsInstance(c2, calc)
        self.assertIsNotNone(c2.lrucache)
        self.assertEqual(c2.eager_call(x=1)['y'], 2)
        self.assertEqual(c2.eager_call(x=1)['y'], 2)
        self.assertEqual(runs, [1])
        self.assertIs(c.with_pathcache(None), c)
        with tempfile.TemporaryDirectory() as tmpdir:
            c3 = c.with_pathcache(tmpdir)
            self.assertIsInstance(c3, calc)
            self.assertIsNotNone(c3.pathcache)
            self.assertEqual(c3.eager_call(x=5)['y'], 6)
            self.assertEqual(c3.eager_call(x=5)['y'], 6)
            self.assertEqual(runs, [1, 5])
        # The original calc is unchanged.
        self.assertIsNone(c.lrucache)
        self.assertIsNone(c.pathcache)
    def test_pickle_plandict(self):
        "Tests pickling calcs, plans, plandicts, and tplandicts."
        import pickle, tempfile, os
        from pcollections import lazy
        from immlib import save, load
        from immlib.workflow import (calc, plan, plandict, is_tplandict,
                                     save_ready, PlanError)
        # Calcs and plans are pickled by reference to their functions.
        c = pickle.loads(pickle.dumps(_pickle_add_one.calc))
        self.assertIs(c, _pickle_add_one.calc)
        p = pickle.loads(pickle.dumps(_pickle_plan))
        self.assertIsInstance(p, plan)
        self.assertEqual(dict(p), dict(_pickle_plan))
        # Calc copies and locally defined calcs can't be pickled.
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(_pickle_add_one.calc.rename_keys(x='u'))
        @calc('v')
        def local_calc(u):
            return u
        with self.assertRaises((pickle.PicklingError, AttributeError)):
            pickle.dumps(local_calc.calc)
        # By default, only the plan and inputs are saved, so values are
        # recomputed after unpickling.
        _PICKLE_RUNS.clear()
        pd = _pickle_plan(x=1)
        self.assertEqual(pd['z'], 4.0)
        self.assertEqual(_PICKLE_RUNS, ['check_x', 'add_one', 'double'])
        data = pickle.dumps(pd)
        _PICKLE_RUNS.clear()
        pd2 = pickle.loads(data)
        self.assertIsInstance(pd2, plandict)
        self.assertEqual(pd2.inputs, pd.inputs)
        self.assertFalse(pd2.is_ready('z'))
        self.assertEqual(dict(pd2), dict(pd))
        self.assertEqual(_PICKLE_RUNS, ['check_x', 'add_one', 'double'])
        # With save_ready, computed values are restored, not recomputed.
        _PICKLE_RUNS.clear()
        pd = _pickle_plan(x=3)
        pd['y']
        _PICKLE_RUNS.clear()
        with save_ready():
            data = pickle.dumps(pd)
        pd2 = pickle.loads(data)
        self.assertEqual(_PICKLE_RUNS, [])
        self.assertTrue(pd2.is_ready('y'))
        self.assertEqual(pd2['y'], 4.0)
        self.assertEqual(_PICKLE_RUNS, [])
        self.assertEqual(pd2['z'], 8.0)
        self.assertEqual(_PICKLE_RUNS, ['double'])
        # Unpickled plandicts can be updated as usual.
        self.assertEqual(pd2.set('x', 0)['z'], 2.0)
        # Outside of save_ready, values are not saved.
        pd2 = pickle.loads(pickle.dumps(pd))
        self.assertFalse(pd2.is_ready('y'))
        # Failed calculations are not saved; they fail again when requested.
        pd = _pickle_plan(x=-1)
        with self.assertRaises(PlanError):
            pd['z']
        with save_ready():
            data = pickle.dumps(pd)
        pd2 = pickle.loads(data)
        with self.assertRaises(PlanError):
            pd2['z']
        # Lazy inputs are computed when pickled.
        pd = _pickle_plan(x=lazy(lambda: 2))
        pd2 = pickle.loads(pickle.dumps(pd))
        self.assertEqual(pd2['z'], 6.0)
        # tplandicts are pickled as tplandicts.
        td = _pickle_plan(x=1).transient()
        td['x'] = 10
        td2 = pickle.loads(pickle.dumps(td))
        self.assertTrue(is_tplandict(td2))
        self.assertEqual(dict(td2.persistent()),
                         {'x': 10.0, 'y': 11.0, 'z': 22.0})
        td2['x'] = 5
        self.assertEqual(td2['z'], 12.0)
        # immlib.save's pickle format supports save_ready.
        with tempfile.TemporaryDirectory() as tmpdir:
            flnm = os.path.join(tmpdir, 'pd.pkl')
            pd = _pickle_plan(x=1)
            pd['z']
            save(flnm, pd, 'pickle', save_ready=True)
            _PICKLE_RUNS.clear()
            pd2 = load(flnm)
            self.assertEqual(dict(pd2), dict(pd))
            self.assertEqual(_PICKLE_RUNS, [])
            save(flnm, pd, 'pickle')
            pd2 = load(flnm)
            self.assertEqual(dict(pd2), dict(pd))
            self.assertEqual(_PICKLE_RUNS, ['check_x', 'add_one', 'double'])
    def test_pickle_plandict_cache_path(self):
        "Tests that unpickled plandicts read from their cache path."
        import pickle, tempfile
        _PICKLE_RUNS.clear()
        with tempfile.TemporaryDirectory() as dir1, \
             tempfile.TemporaryDirectory() as dir2:
            pd = _pickle_cache_plan(x=2, cache_path=dir1)
            self.assertEqual(pd['y'], 4)
            self.assertEqual(_PICKLE_RUNS, ['cached_square'])
            pd2 = pickle.loads(pickle.dumps(pd))
            self.assertEqual(pd2.inputs['cache_path'], dir1)
            self.assertEqual(pd2['y'], 4)
            # The value was read from the cache, not recomputed.
            self.assertEqual(_PICKLE_RUNS, ['cached_square'])
            # A different cache path does not have the value.
            pd3 = pd2.set('cache_path', dir2)
            self.assertEqual(pd3['y'], 4)
            self.assertEqual(_PICKLE_RUNS, ['cached_square'] * 2)


# Module-level calcs and plans for the pickling tests (calcs are pickled by
# reference to their functions, so they can't be defined inside a test).
from immlib.workflow import calc as _calc, plan as _plan
_PICKLE_RUNS = []
@_calc('x', lazy=False)
def _pickle_check_x(x):
    _PICKLE_RUNS.append('check_x')
    return (float(x),)
@_calc('y')
def _pickle_add_one(x):
    _PICKLE_RUNS.append('add_one')
    if x < 0:
        raise ValueError("x must be non-negative")
    return x + 1
@_calc('z')
def _pickle_double(y):
    _PICKLE_RUNS.append('double')
    return 2 * y
_pickle_plan = _plan(
    check=_pickle_check_x, add=_pickle_add_one, double=_pickle_double)
@_calc('y', pathcache=True)
def _pickle_cached_square(x):
    _PICKLE_RUNS.append('cached_square')
    return x * x
_pickle_cache_plan = _plan(square=_pickle_cached_square)
