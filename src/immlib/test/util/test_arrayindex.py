# -*- coding: utf-8 -*-
################################################################################
# immlib/test/util/test_arrayindex.py

"""Tests of the ArrayIndex type, which lives in immlib.util.

`ArrayIndex` was moved here from the former `immlib.types` subpackage; these
tests moved with it.
"""


# Dependencies #################################################################

from unittest import TestCase

import numpy as np


# Tests ########################################################################

class TestArrayIndex(TestCase):
    """Tests immlib.util.ArrayIndex."""

    def test_array_index(self):
        "Tests the ArrayIndex type."
        from ...util import ArrayIndex
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

    def test_array_index_find_default(self):
        """Tests ArrayIndex.find's handling of missing identities.

        Without a default, a missing identity is a KeyError. With one, the
        default fills its place -- and for an unraveled result there is one
        default per *dimension* of the indexed array, not one per identity
        looked up, which is a different number except by coincidence.
        """
        from ...util import ArrayIndex
        a = np.array([[10, 20, 30], [40, 50, 60]])
        ai = ArrayIndex(a)
        # Present identities unravel into one coordinate array per
        # dimension.
        (rows, cols) = ai.find([20, 50])
        self.assertTrue(np.array_equal(rows, [0, 1]))
        self.assertTrue(np.array_equal(cols, [1, 1]))
        # A missing identity raises without a default, whatever the shape of
        # the query.
        with self.assertRaises(KeyError):
            ai.find(99)
        with self.assertRaises(KeyError):
            ai.find([20, 99])
        # With a default, every coordinate of a missing entry gets it. A
        # scalar query is the case that used to return uninitialized memory
        # in every coordinate after the first.
        idx = ai.find(99, default=-1)
        self.assertEqual(len(idx), a.ndim)
        self.assertTrue(all(int(c) == -1 for c in idx))
        (rows, cols) = ai.find([20, 99], default=-1)
        self.assertTrue(np.array_equal(rows, [0, -1]))
        self.assertTrue(np.array_equal(cols, [1, -1]))
        # More identities than dimensions works too.
        (rows, cols) = ai.find([20, 99, 50, 10], default=-1)
        self.assertTrue(np.array_equal(rows, [0, -1, 1, 0]))
        self.assertTrue(np.array_equal(cols, [1, -1, 1, 0]))
        # A tuple default gives a different value per dimension.
        (rows, cols) = ai.find([20, 99], default=(-1, -2))
        self.assertTrue(np.array_equal(rows, [0, -1]))
        self.assertTrue(np.array_equal(cols, [1, -2]))
        # A tuple of the wrong length is an error rather than a silently
        # half-filled result.
        with self.assertRaises(ValueError):
            ai.find([20, 99], default=(-1,))
        with self.assertRaises(ValueError):
            ai.find([20, 99], default=(-1, -2, -3))
        # With ravel, the result is a flat index and the default fills it
        # directly.
        self.assertTrue(
            np.array_equal(ai.find([20, 99], default=-1, ravel=True), [1, -1]))
        self.assertTrue(
            np.array_equal(ai.find([20, 50], ravel=True), [1, 4]))
        # The same holds at other ranks.
        b = ArrayIndex(np.arange(5) * 10)
        self.assertTrue(np.array_equal(b.find([10, 99], default=-1)[0], [1, -1]))
        self.assertEqual(len(b.find(99, default=-1)), 1)
        c = ArrayIndex(np.arange(24).reshape(2, 3, 4))
        idx = c.find(99, default=-1)
        self.assertEqual(len(idx), 3)
        self.assertTrue(all(int(x) == -1 for x in idx))
        idx = c.find([5, 99], default=(-1, -2, -3))
        self.assertTrue(np.array_equal(idx[0], [0, -1]))
        self.assertTrue(np.array_equal(idx[1], [1, -2]))
        self.assertTrue(np.array_equal(idx[2], [1, -3]))
        # An identity larger than everything in the index exercises the
        # out-of-range path of searchsorted.
        self.assertTrue(
            np.array_equal(ai.find([999999], default=-1, ravel=True), [-1]))
