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
        from immlib.workflow import calc
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
        self.assertIsInstance(result, calc)
        # Calculation objects have a number of members that keep track of the
        # meta-data of the calculation.
        # Firs is the name of the calculation--this is the name of the function.
        self.assertEqual(result.name, 'immlib.test.workflow.test_core.result')
        # The inputs of the calculation are a set of the inputs of the function.
        self.assertEqual(result.inputs, set(['input_1', 'input_2']))
        # The default values of the inputs are stored in the defaults member.
        self.assertEqual(result.defaults, {'input_2': None})
        # The outputs are a tuple of the output names. For a calc without
        # explicitly listed outputs has only one output, its name.
        self.assertEqual(result.outputs, ('result',))
        # The input documentation is stored in the input_docs member.
        self.assertIn('input_1', result.input_docs)
        self.assertIn('input_2', result.input_docs)
        self.assertEqual(len(result.input_docs), 2)
        self.assertIn('The number of elements to include in the result.',
                      result.input_docs['input_1'])
        self.assertIn('The object to put in the list.',
                      result.input_docs['input_2'])
        # The output documentation is stored in the output_docs member.
        self.assertIn('result', result.output_docs)
        self.assertIn('A list of `input_1` occurrences of `input_2`.',
                      result.output_docs['result'])
        self.assertEqual(len(result.output_docs), 1)
        # The calculation can be called using its normal signature.
        self.assertEqual(result(1), {'result': [None]})
        self.assertEqual(result(2, 0), {'result': [0, 0]})
        # The call method is basically an alias for the __call__ method.
        self.assertEqual(result.call(1), {'result': [None]})
        self.assertEqual(result.call(2, 0), {'result': [0, 0]})
        # The call can also be forced to be either eager or lazy--when lazy,
        # the return value is a lazy dict, and the calc isn't actually run until
        # the values are requested; when eager, the call is run right away, and
        # the return value is an frozendict instead of a lazydict.
        self.assertIsInstance(result.eager_call(1), pdict)
        self.assertEqual(result.eager_call(1), {'result': [None]})
        self.assertEqual(result.eager_call(2, 0), {'result': [0, 0]})
        self.assertIsInstance(result.lazy_call(1), ldict)
        self.assertEqual(result.lazy_call(1), {'result': [None]})
        self.assertEqual(result.lazy_call(2, 0), {'result': [0, 0]})
        # It can also be called using the mapcall method.
        m1 = dict(input_1=1)
        m2 = dict(input_1=2, input_2=0)
        self.assertEqual(result.mapcall(m1), {'result': [None]})
        self.assertEqual(result.mapcall(m2), {'result': [0,0]})
        # These can also be lazy or eager.
        self.assertIsInstance(result.eager_mapcall(m1), pdict)
        self.assertEqual(result.eager_mapcall(m1), {'result': [None]})
        self.assertEqual(result.eager_mapcall(m2), {'result': [0, 0]})
        self.assertIsInstance(result.lazy_mapcall(m1), ldict)
        self.assertEqual(result.lazy_mapcall(m1), {'result': [None]})
        self.assertEqual(result.lazy_mapcall(m2), {'result': [0, 0]})
        # Calculations can have multiple outputs as well as multiple inputs.
        @calc('out1', 'out2', 'out3')
        def sample_calc(in1, in2, in3):
            return (in1 + 1, in2 + 2, in3 + 3)
        res = sample_calc(1, 2, 3)
        self.assertIsInstance(res, ldict)
        self.assertEqual(len(res), 3)
        self.assertEqual(res['out1'], 2)
        self.assertEqual(res['out2'], 4)
        self.assertEqual(res['out3'], 6)
        # New calcs can be made that change the names of the calculation
        # variables (the inputs and outputs) using the tr (translate) method.
        sample_tr = sample_calc.tr(out1='x', out2='y', in3='z')
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
        from immlib.workflow import (calc, is_calc)
        @calc
        def result(input_1, input_2=None):
            return ([input_2] * input_1,)
        # is_calc(x) is just an alias for isinstance(x, calc).
        self.assertTrue(is_calc(result))
        self.assertFalse(is_calc(lambda x:x))
        self.assertEqual(result(2,0)['result'], [0, 0])
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
            return np.exp(-0.5 * ((x - mu)/std)**2) / (np.sqrt(2*np.pi) * std)
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
            return np.sum(x * weights) / np.sum(weights)
        # Filter calculations can be used to update the input variables to a
        # plan--they are calc units that accept only 1 input and that return
        # same input.
        @calc('x')
        def filter_x(x):
            x = np.asarray(x)
            assert len(x.shape) == 1, "x must be a vector"
            assert np.issubdtype(x.dtype, np.number), "x must be numeric"
            return x
        # The calculations are given names (keys) and put together in a plan.
        nwm = plan(weights_step=normal_pdf,
                   mean_step=weighted_mean,
                   # filters must be named filter_<filtered-param>
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
        # There is a filter on x, which dictates that x be a lazy value
        # but immediately cached (ready).
        self.assertTrue(pd.is_lazy('x'))
        self.assertTrue(pd.is_ready('x'))
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
        self.assertTrue(pd.is_ready('weights'))
        self.assertTrue(pd.is_lazy('mean'))
        self.assertFalse(pd.is_ready('mean'))
        # It will have converted the x value into an array.
        self.assertIsInstance(pd['x'], np.ndarray)
        self.assertTrue(np.array_equal(pd['x'], [-1, 1, 2, 8.5]))
        self.assertEqual(pd['mu'], 1.5)
        self.assertEqual(pd['std'], 1)
        self.assertAlmostEqual(pd['mean'], 1.4392777559)
        # Since we marked the filter as non-lazy, it should raise errors when
        # the plan is fulfilled.
        with self.assertRaises(RuntimeError): nwm(x=10)
