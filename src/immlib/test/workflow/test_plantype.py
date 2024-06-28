# -*- coding: utf-8 -*-
################################################################################
# immlib/test/workflow/test_plantype.py
#
# Tests of the plantype system in immlib: i.e., tests for the code in the
# immlib.workflow._plantype module.


# Dependencies #################################################################

from unittest import TestCase

class TestWorkflowPlanType(TestCase):
    """Tests the immlib.workflow._plantype module."""
    def test_plantype(self):
        import numpy as np
        from immlib.workflow import (planobject, plantype, calc)
        import sys, io
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
        # We can instantiate such a type as normal:
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
