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
        # We pick AB as the base.
        base = b - a
        baselen = np.hypot(base[0], base[1])
        ubase = base / baselen
        print('Calculating height...')
        uortho = np.array([[0,-1],[1,0]]) @ ubase
        height = np.abs(np.dot(uortho, c - a))
        return (baselen, height)
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
    def __eq__(self, other):
        if not isinstance(other, TriangleData):
            return False
        return (
            np.array_equal(self.a, other.a) and
            np.array_equal(self.b, other.b) and
            np.array_equal(self.c, other.c))
    def __hash__(self):
        return super().__hash__()

class TestWorkflowPlanType(TestCase):
    """Tests the immlib.workflow._plantype module."""
    def test_plantype(self):
        import numpy as np
        from immlib.workflow import (
            planobject, plantype, calc, is_plantype, is_planobject)
        import sys, io, pickle
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
                  '_plandict_', '__new__'):
            self.assertIn(k, d)
        # Trying to initialize an object after it has been initialized should
        # raise an error.
        with self.assertRaises(RuntimeError):
            tri.__init__((0,0), (1,0), (0,1))
        # Init methods can be called in parent classes without problems.
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
        # There is an automatically defined init function.
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
        tri = TriangleData((0,0), (1,0), (0,1))
        self.assertEqual(tri, pickle.loads(pickle.dumps(tri)))
        # planobjects are by default only equal on the basis of their type and
        # their inputs.
        tri1 = TriangleData((0,0), (1,0), (0,1))
        tri2 = TriangleData((0.0,0.0), (1.0,0.0), (0.0,1.0))
        tri3 = TriangleData((0.1,0.0), (1.0,0.0), (0.0,1.0))
        tri4 = ChildTri((0,0), (1,0), (0,1))
        self.assertEqual(tri1, tri2)
        self.assertNotEqual(tri2, tri3)
        self.assertNotEqual(tri1, tri4)
        self.assertEqual(hash(tri1), hash(tri2))
        # A planobject can be made transient and a transient one can be made
        # persistent again.
        ttri1 = tri1.transient()
        self.assertEqual(tri1, ttri1)
        self.assertTrue(tri1.is_persistent())
        self.assertFalse(ttri1.is_persistent())
        self.assertTrue(np.array_equal(ttri1.a, (0, 0)))
        self.assertEqual(ttri1.area, 0.5)
        ttri1.a = (0.2, 0.2)
        self.assertTrue(np.array_equal(ttri1.a, (0.2, 0.2)))
        self.assertAlmostEqual(ttri1.area, 0.3)
        tri5 = ttri1.persistent()
        self.assertEqual(ttri1, tri5)
        self.assertTrue(tri5.is_persistent())
        self.assertTrue(np.array_equal(tri5.a, (0.2, 0.2)))
        self.assertAlmostEqual(tri5.area, 0.3)
        # At this point, we can return the normal standard output (we should be
        # done printing things).
        sys.stdout = sys.__stdout__
        # We can turn planobjects into strings:
        class SimpleObj(planobject):
            @calc('x')
            def filter_x(x):
                return (int(x),)
            @calc('z')
            def calc_z(x, y):
                return ((x*y),)
        obj = SimpleObj(x=10.0, y=2.0)
        self.assertEqual(str(obj), 'SimpleObj(x=<lazy>, y=2.0; z=<lazy>)')
        self.assertTrue(
            repr(obj).startswith(
                f'{__name__}.SimpleObj(x=10, y=2.0; z=lazy(<'))
        self.assertTrue(
            repr(obj).endswith(
                '>: waiting))'))
        self.assertIsInstance(obj.x, int)
        self.assertEqual(str(obj), 'SimpleObj(x=10, y=2.0; z=<lazy>)')
        # If the class has no outputs, there is no semicolon.
        class TrivialObj(planobject):
            @calc('x')
            def filter_x(x):
                return (int(x),)
        trivobj = TrivialObj(x=10.0)
        self.assertEqual(str(trivobj), 'TrivialObj(x=<lazy>)')
        self.assertEqual(trivobj.x, 10)
        self.assertEqual(repr(trivobj), f'{__name__}.TrivialObj(x=10)')
        # There are tests for the objects and types also:
        self.assertTrue(is_planobject(obj))
        self.assertFalse(is_planobject(None))
        self.assertTrue(is_plantype(SimpleObj))
        self.assertFalse(is_plantype(type))
