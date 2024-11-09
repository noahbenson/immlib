# -*- coding: utf-8 -*-
################################################################################
# immlib/test/types/test_core.py

"""Tests of the core types module in immlib: i.e., tests for the code in the
immlib.types._core module.
"""


# Dependencies #################################################################

from unittest import TestCase

import numpy as np


# Tests ########################################################################

class TestTypesCore(TestCase):
    """Tests the immlib.types._core module."""
    
    def test_metaobject(self):
        "Tests the MetaObject class."
        from ...types import MetaObject
        from pcollections import ldict
        # We can make objects with metadata (and the metadata get saved as
        # persistent dictionaries).
        o = MetaObject(metadata={'a': 1, 'b': 2})
        self.assertIsInstance(o.metadata, ldict)
        self.assertEqual(o.metadata, dict(a=1, b=2))
        # We can add and subtract from these metadata maps.
        oo = o.withmeta(c=3)
        self.assertEqual(oo.metadata, dict(a=1, b=2, c=3))
        ooo = oo.dropmeta('a')
        self.assertEqual(ooo.metadata, dict(b=2, c=3))
        # The original object must stay the same through all this.
        self.assertEqual(o.metadata, dict(a=1, b=2))
        self.assertEqual(oo.metadata, dict(a=1, b=2, c=3))
        # We can also set the metadata directly.
        oo = o.set_metadata(dict(e=5, f=6))
        self.assertEqual(oo.metadata, dict(e=5, f=6))
        self.assertIsInstance(oo.metadata, ldict)
        # Finally we ca clear the metadata.
        self.assertEqual(o.clear_metadata().metadata, dict())
    def test_immutable(self):
        "Test the Immutable type."
        from ...types import Immutable
        class immtest(Immutable):
            __slots__ = ('a', 'b')
            def __init__(self, a, b):
                self.a = a
                self.b = b
        immobj = immtest(1, 2)
        self.assertEqual(immobj.a, 1)
        self.assertEqual(immobj.b, 2)
        with self.assertRaises(TypeError):
            immobj.a = 10
        with self.assertRaises(TypeError):
            immobj.b = 20
        with self.assertRaises(TypeError):
            del immobj.b
    def test_array_index(self):
        "Tests the ArrayIndex type."
        from ...types import ArrayIndex
        # Test with basic integers.
        arr = np.reshape(np.arange(2*3*4), (2,3,4))
        ai = ArrayIndex(arr)
        for num in arr.flat:
            self.assertEqual(arr[ai.find(num)], num)
        # We can also find multiple elements at once:
        els = [2,5,9]
        self.assertTrue(np.array_equal(arr[ai.find(els)], els))
        # Default values can be provided when not found:
        self.assertEqual(ai.find(100, default=-1, ravel=True), -1)
        # Otherwise, if we look for something not there, we get a KeyError:
        with self.assertRaises(KeyError):
            ai.find(100)
        # The index freezes its array.
        self.assertFalse(ai.array.flags['WRITEABLE'])
        # But we can instruct the array to stay writeable if desired.
        ai = ArrayIndex(arr, freeze=False)
        self.assertTrue(ai.array.flags['WRITEABLE'])
        # Tests with strings instead of integers.
        lorem = '''Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed
            do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut
            enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi
            ut aliquip ex ea commodo consequat. Duis aute irure dolor in
            reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla
            pariatur. Excepteur sint occaecat cupidatat non proident, sunt in
            culpa qui officia deserunt mollit anim id est laborum.'''
        words = lorem.split()[:24]
        arr = np.reshape(words, (4,3,2))
        ai = ArrayIndex(arr)
        for word in words:
            self.assertEqual(arr[ai.find(word)], word)
        els = [words[k] for k in [2,5,9]]
        self.assertTrue(np.array_equal(arr[ai.find(els)], els))
        # ArrayIndex objects are immutable.
        with self.assertRaises(TypeError):
            del ai.flatdata
        with self.assertRaises(TypeError):
            del ai[0]
        with self.assertRaises(TypeError):
            ai.flatdata = ()
        with self.assertRaises(TypeError):
            ai[0] = 0
        # They raise some other errors too:
        with self.assertRaises(TypeError):
            ai.find(0, xyz=10)

