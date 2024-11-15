# -*- coding: utf-8 -*-
################################################################################
# immlib/test/workflow/test_plantype.py
#
# Tests of the plantype system in immlib: i.e., tests for the code in the
# immlib.workflow._plantype module.


# Dependencies #################################################################

from unittest import TestCase
import numpy as np

from ...workflow import (planobject, calc)

# The plantype type can be used as a metaclass for a class in order to
# make that class into a workflow/plantype class. Alternately, we can
# just inherit from planobject.
class TriangleData(planobject):
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
    @calc('a', lazy=False)
    def filter_a(a):
        a = np.array(a)
        assert a.shape == (2,)
        return a
    @calc('b', lazy=False)
    def filter_b(b):
        b = np.array(b)
        assert b.shape == (2,)
        return b
    @calc('c', lazy=False)
    def filter_c(c):
        c = np.array(c)
        assert c.shape == (2,)
        return c
    @calc('base', 'height')
    def calc_triangle_base(a, b, c):
        '''calc_triangle_base computes the base (x-width) of the
        triangle a-b-c.

        Inputs
        ------
        a : list-like
            The (x,y) coordinate of point a in triangle a-b-c.
        b : list-like
            The (x,y) coordinate of point b in triangle a-b-c.
        c : list-like
            The (x,y) coordinate of point c in triangle a-b-c.

        Outputs
        -------
        base : number
            The base, or width, of the triangle a-b-c.
        height : number
            The height of the triangle a-b-c.
        '''
        print('Calculating base...')
        xs = [a[0], b[0], c[0]]
        xmin = min(xs)
        xmax = max(xs)
        print('Calculating height...')
        ys = [a[1], b[1], c[1]]
        ymin = min(ys)
        ymax = max(ys)
        return (xmax - xmin, ymax - ymin)
    @calc('area')
    def calc_triangle_area(base, height):
        '''calc_triangle_are computes the area of a triangle with a
        given base and height.

        Inputs
        ------
        base : number
            The base of the triangle.
        height : number
            The height of the triangle.

        Outputs
        -------
        area : number
            The area of the triangle with the given base and height.
        '''
        print('Calculating area...')
        return {'area': base * height * 0.5}


class TestWorkflowPlanType(TestCase):
    """Tests the immlib.workflow._plantype module."""
    def test_plantype(self):
        import numpy as np
        from immlib.workflow import (planobject, plantype, calc)
        import sys, io
        # We can instantiate the plantype as normal:
        tri = TriangleData((0,0), (1,0), (0,1))
        # We've setup the class to print some messages the first time values get
        # calculated, so we capture them here.
        sys.stdout = io.StringIO()
        self.assertEqual(tri.base, 1)
        self.assertEqual(sys.stdout.getvalue(),
                         'Calculating base...\nCalculating height...\n')
        self.assertEqual(tri.height, 1)
        self.assertEqual(sys.stdout.getvalue(),
                         'Calculating base...\nCalculating height...\n')
        sys.stdout = io.StringIO()
        self.assertEqual(tri.area, 0.5)
        self.assertEqual(sys.stdout.getvalue(), 'Calculating area...\n')
        sys.stdout = sys.__stdout__
        # Objects of type planobject are immutable:
        with self.assertRaises(TypeError):
            tri.a = (2,2)
        with self.assertRaises(TypeError):
            del tri.a
        # Test some basic attribute stuff...
        with self.assertRaises(AttributeError):
            tri.notanattribute
        # The dir function should return normal object items as well as the
        # plan's values:
        d = dir(tri)
        for k in ('a', 'b', 'c', 'base', 'height', 'area', '__class__',
                  '__plandict__', '__new__'):
            self.assertIn(k, d)
        # Trying to initialize an object after it has been initialized should
        # raise an error.
        with self.assertRaises(RuntimeError):
            tri.__init__((0,0), (1,0), (0,1))
        # Init methods can be called in parent classes without causing problems.
        class ChildTri(TriangleData):
            def __init__(self, a, b, c):
                super().__init__(a, b, c)
                a = np.asarray(a)
                self.a = (0,0)
                self.b = b - a
                self.c = c - a
        tri = ChildTri((0,1), (1,0), (0,2))
        self.assertTrue(np.array_equal(tri.a, (0,0)))
        self.assertTrue(np.array_equal(tri.b, (1,-1)))
        self.assertTrue(np.array_equal(tri.c, (0,1)))
        # If we try to make a planobject that sets a non-input in its __init__
        # method, it should raise an error.
        class BadTriData(TriangleData):
            def __init__(self, a, b, c):
                super().__init__(a, b, c)
                self.base = 10
        with self.assertRaises(ValueError):
            tri = BadTriData((0,0), (1,0), (0,1))
        # There is an automatically defined init function if we don't define it.
        class AutoInit(planobject):
            @calc('x')
            def filter_x(x):
                return (float(x),)
            @calc('y')
            def filter_y(y):
                return (float(y),)
            @calc('z')
            def filter_z(z):
                return (float(z),)
            @calc('f')
            def calc_outputs(x, y, z=0):
                result = x + y*z
                return (result,)
        ai = AutoInit(x=1, y=2, z='4')
        self.assertIsInstance(ai, AutoInit)
        self.assertEqual(ai.x, 1)
        self.assertEqual(ai.y, 2)
        self.assertEqual(ai.z, 4)
        self.assertEqual(ai.f, 9)
        self.assertIsInstance(ai.x, float)
        self.assertIsInstance(ai.y, float)
        self.assertIsInstance(ai.z, float)
        self.assertIsInstance(ai.f, float)
        # If we don't provide an object with all its inputs, there's an error.
        with self.assertRaises(ValueError):
            ai = AutoInit(x=1, z='4')
        # planobjects can be serialized and deserialized:
        import pickle
        tri = TriangleData((0,0), (1,0), (0,1))
        self.assertEqual(tri, pickle.loads(pickle.dumps(tri)))
