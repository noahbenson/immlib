# -*- coding: utf-8 -*-
################################################################################
# immlib/test/util/test_core.py

"""Tests of the core utilities module in immlib: i.e., tests for the code in the
immlib.util._core module.
"""


# Dependencies #################################################################

from unittest import TestCase


# Tests ########################################################################

class TestUtilCore(TestCase):
    """Tests the immlib.util._core module."""

    # String Functions #########################################################
    def test_is_str(self):
        from immlib import is_str
        # is_str is just a wrapper for isinstance(obj, str).
        self.assertTrue(is_str('abc'))
        self.assertTrue(is_str(''))
        self.assertFalse(is_str(100))
        self.assertFalse(is_str(None))
    def test_strnorm(self):
        from immlib import strnorm
        # There are a lot of string encoding details that should probably be
        # tested carefully here, but for now, we're mostly concerned that the
        # most basic strings get normalized properly.
        self.assertEqual('abc', strnorm('abc'))
        self.assertEqual('abc', strnorm('aBc', case=True))
    def test_strcmp(self):
        from immlib import strcmp
        # strcmp is, at its simplest, just a string-comparison function.
        self.assertEqual(0,  strcmp('abc', 'abc'))
        self.assertEqual(-1, strcmp('abc', 'bca'))
        self.assertEqual(1,  strcmp('bca', 'abc'))
        # There are a few bells and whistles for strcmp, thought. First, the
        # case option lets you decide whether to ignore case (via strnorm).
        self.assertEqual(-1, strcmp('ABC', 'abc'))
        self.assertEqual(0,  strcmp('ABC', 'abc', case=False))
        # The strip option lets one ignore whitespace on either side of the
        # arguments.
        self.assertEqual(-1, strcmp(' abc', 'abc  '))
        self.assertEqual(0,  strcmp(' abc', 'abc  ', strip=True))
        # The split argument lets you split on whitespace then compare the
        # individual split parts (i.e., this option should make all strings that
        # are identical up to the amount of whitespace should be equal).
        self.assertEqual(1, strcmp('abc def ghi', ' abc  def ghi '))
        self.assertEqual(0, strcmp('abc def ghi', ' abc  def ghi ', split=True))
        # If one of the arguments isn't a string, strcmp returns None.
        self.assertIsNone(strcmp(None, 10))
    def test_streq(self):
        from immlib import streq
        # streq is just a string equality predicate function.
        self.assertTrue(streq('abc', 'abc'))
        self.assertFalse(streq('abc', 'def'))
        # The case option can tell it to ignore case.
        self.assertFalse(streq('ABC', 'abc'))
        self.assertTrue(streq('ABC', 'abc', case=False))
        # The strip option can be used to ignore trailing/leading whitespace.
        self.assertFalse(streq(' abc', 'abc  '))
        self.assertTrue(streq(' abc', 'abc  ', strip=True))
        # The split argument lets you split on whitespace then compare the
        # individual split parts (i.e., this option should make all strings that
        # are identical up to the amount of whitespace should be equal).
        self.assertFalse(streq('abc def ghi', ' abc  def ghi '))
        self.assertTrue(streq('abc def ghi', ' abc  def ghi ', split=True))
        # Nonstring arguments return None.
        self.assertIsNone(streq(None, 'abc'))
    def test_strends(self):
        from immlib import strends
        # strends is just a string equality predicate function.
        self.assertTrue(strends('abcdef', 'def'))
        self.assertFalse(strends('abcdef', 'bcd'))
        # The case option can tell it to ignore case.
        self.assertFalse(strends('ABCDEF', 'def'))
        self.assertTrue(strends('ABCDEF', 'def', case=False))
        # The strip option can be used to ignore trailing/leading whitespace.
        self.assertFalse(strends(' abcdef ', 'def  '))
        self.assertTrue(strends(' abcdef ', 'def  ', strip=True))
        # Nonstring arguments return None.
        self.assertIsNone(strends(None, 'abc'))
    def test_strstarts(self):
        from immlib import strstarts
        # strstarts is just a string equality predicate function.
        self.assertTrue(strstarts('abcdef', 'abc'))
        self.assertFalse(strstarts('abcdef', 'bcd'))
        # The case option can tell it to ignore case.
        self.assertFalse(strstarts('ABCDEF', 'abc'))
        self.assertTrue(strstarts('ABCDEF', 'abc', case=False))
        # The strip option can be used to ignore trailing/leading whitespace.
        self.assertFalse(strstarts(' abcdef ', '  abc'))
        self.assertTrue(strstarts(' abcdef ', '  abc', strip=True))
        # Nonstring arguments return None.
        self.assertIsNone(strstarts(None, 'abc'))
    def test_strissym(self):
        from immlib import strissym
        # strissym tests whether a string is both a string and a valid Python
        # symbol.
        self.assertTrue(strissym('abc'))
        self.assertTrue(strissym('def123'))
        self.assertTrue(strissym('_10xyz'))
        self.assertFalse(strissym('abc def'))
        self.assertFalse(strissym(' abcdef '))
        self.assertFalse(strissym('a-b'))
        self.assertFalse(strissym('10'))
        # Keywords are allowed.
        self.assertTrue(strissym('for'))
        self.assertTrue(strissym('and'))
        # Non-strings return Nonee.
        self.assertFalse(strissym(None))
        self.assertFalse(strissym(10))
    def test_striskey(self):
        from immlib import striskey
        # striskey tests whether a string is (1) a string, (2) a valid Python
        # symbol, and (3) an existing Python keyword.
        self.assertTrue(striskey('for'))
        self.assertTrue(striskey('and'))
        self.assertTrue(striskey('None'))
        self.assertFalse(striskey('abc'))
        self.assertFalse(striskey('def123'))
        self.assertFalse(striskey('_10xyz'))
        self.assertFalse(striskey('abc def'))
        self.assertFalse(striskey(' abcdef '))
        self.assertFalse(striskey('a-b'))
        self.assertFalse(striskey('10'))
        # Non-strings return None.
        self.assertIsNone(striskey(None))
        self.assertIsNone(striskey(10))
    def test_strisvar(self):
        from immlib import strisvar
        # strisvar tests whether a string is (1) a string, (2) a valid Python
        # symbol, and (3) an *not* existing Python keyword.
        self.assertFalse(strisvar('for'))
        self.assertFalse(strisvar('and'))
        self.assertFalse(strisvar('None'))
        self.assertTrue(strisvar('abc'))
        self.assertTrue(strisvar('def123'))
        self.assertTrue(strisvar('_10xyz'))
        self.assertFalse(strisvar('abc def'))
        self.assertFalse(strisvar(' abcdef '))
        self.assertFalse(strisvar('a-b'))
        self.assertFalse(strisvar('10'))
        # Non-strings return Nonee.
        self.assertIsNone(strisvar(None))
        self.assertIsNone(strisvar(10))

    # Other Utilities ##########################################################
    def test_hashsafe(self):
        from immlib import hashsafe
        # hashsafe returns hash(x) if x is hashable and None otherwise.
        self.assertIsNone(hashsafe({}))
        self.assertIsNone(hashsafe([1, 2, 3]))
        self.assertIsNone(hashsafe(set(['a', 'b'])))
        self.assertEqual(hash(10), hashsafe(10))
        self.assertEqual(hash('abc'), hashsafe('abc'))
        self.assertEqual(hash((1, 10, 100)), hashsafe((1, 10, 100)))
    def test_can_hash(self):
        from immlib import can_hash
        # can_hash(x) returns True if hash(x) will successfully return a hash
        # and returns False if such a call would raise an error.
        self.assertTrue(can_hash(10))
        self.assertTrue(can_hash('abc'))
        self.assertTrue(can_hash((1, 10, 100)))
        self.assertFalse(can_hash({}))
        self.assertFalse(can_hash([1, 2, 3]))
        self.assertFalse(can_hash(set(['a', 'b'])))
    def test_itersafe(self):
        from immlib import itersafe
        # itersafe returns iter(x) if x is iterable and None otherwise.
        self.assertIsNone(itersafe(10))
        self.assertIsNone(itersafe(lambda x:x))
        self.assertEqual(list(itersafe([1, 2, 3])), [1, 2, 3])
    def test_can_iter(self):
        from immlib import can_iter
        # can_iter(x) returns True if iter(x) will successfully return an
        # iterator and returns False if such a call would raise an error.
        self.assertTrue(can_iter('abc'))
        self.assertTrue(can_iter([]))
        self.assertTrue(can_iter((1, 10, 100)))
        self.assertFalse(can_iter(10))
        self.assertFalse(can_iter(lambda x:x))

    # Freeze/Thaw Utilities ####################################################
    def test_to_frozenarray(self):
        from immlib.util import to_frozenarray
        import numpy as np
        # frozenarray converts a read-write numpy array into a frozen one.
        x = np.linspace(0, 1, 25)
        y = to_frozenarray(x)
        self.assertTrue(np.array_equal(x, y))
        self.assertIsNot(x, y)
        self.assertTrue(x.flags['WRITEABLE'])
        self.assertFalse(y.flags['WRITEABLE'])
        # If a frozenarray of an already frozen array is requested, the array is
        # returned as-is.
        self.assertIs(y, to_frozenarray(y))
        # However, one can override this with the copy argument.
        self.assertIsNot(y, to_frozenarray(y, copy=True))
        # Typically a copy is made of the original array if it is not already
        # frozen, but one can request that it not be copied.
        z = to_frozenarray(x, copy=False)
        self.assertIs(z, x)
        self.assertFalse(x.flags['WRITEABLE'])
