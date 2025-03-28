# -*- coding: utf-8 -*-
################################################################################
# immlib/test/util/test_core.py

"""Tests of the core utilities module in immlib: i.e., tests for the code in the
immlib.util._core module.
"""


# Dependencies #################################################################

from unittest import TestCase

import numpy as np


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
        self.assertEqual(-1, strcmp('_abc', 'abc__', strip=True))
        self.assertEqual(0,  strcmp('_abc', 'abc__', strip='_'))
        # The split argument lets you split on whitespace then compare the
        # individual split parts (i.e., this option should make all strings that
        # are identical up to the amount of whitespace should be equal).
        self.assertEqual(1, strcmp('abc def ghi', ' abc  def ghi '))
        self.assertEqual(0, strcmp('abc def ghi', ' abc  def ghi ', split=True))
        self.assertEqual(0, strcmp('abc_def_ghi', 'abc_def_ghi', split='_'))
        self.assertEqual(-1, strcmp('abc def ghi', ' bbc def ghi ', split=True))
        self.assertEqual(1, strcmp('abc eef ghi', ' abc def ghi ', split=True))
        self.assertEqual(0, strcmp('a b c d', 'a b c d', split=True))
        self.assertEqual(1, strcmp('a b c d', 'a b c', split=True))
        self.assertEqual(-1, strcmp('a b c', 'a b c d', split=True))
        self.assertEqual(0, strcmp('abc def ghi', ' abc def ghi ', split=True))
        # In some cases, we can split and strip:
        self.assertEqual(
            strcmp('abc_def_ghi', '_abc_def_ghi_', split='_', strip='_'),
            0)
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

    # Freeze/Thaw Utilities ####################################################
    def test_frozenarray(self):
        "Tests the frozenarray() and freezearray() functions."
        from immlib.util import frozenarray, freezearray
        from immlib import quant
        import numpy as np
        # frozenarray converts a read-write numpy array into a frozen one.
        x = np.linspace(0, 1, 25)
        y = frozenarray(x)
        self.assertTrue(np.array_equal(x, y))
        self.assertIsNot(x, y)
        self.assertTrue(x.flags['WRITEABLE'])
        self.assertFalse(y.flags['WRITEABLE'])
        # If a frozenarray of an already frozen array is requested, the array is
        # returned as-is.
        self.assertIs(y, frozenarray(y))
        # However, one can override this with the copy argument.
        self.assertIsNot(y, frozenarray(y, copy=True))
        # Typically a copy is made of the original array if it is not already
        # frozen, but one can use freezearray to prevent copying.
        z = frozenarray(x)
        self.assertIsNot(z, x)
        self.assertTrue(x.flags['WRITEABLE'])
        freezearray(x)
        self.assertFalse(x.flags['WRITEABLE'])
        # frozenarray also works with sparse arrays.
        import scipy.sparse as sps
        x = sps.csr_array(
            ([1.0, 2.0, 3.5, 6.0], ([0,1,3,4], [3,4,2,1])),
            shape=(5,5))
        y = frozenarray(x)
        self.assertIsNot(x, y)
        self.assertIs(frozenarray(y), y)
        self.assertIsNot(y, frozenarray(y, copy=True))
        self.assertFalse(y.data.flags['WRITEABLE'])
        self.assertTrue(x.data.flags['WRITEABLE'])
        self.assertTrue(np.array_equal(x.data, y.data))
        # As does freeze:
        freezearray(x)
        self.assertFalse(x.data.flags['WRITEABLE'])
        # Both also work with quantities.
        x = quant(
            sps.csr_array(
                ([1.0, 2.0, 3.5, 6.0], ([0,1,3,4], [3,4,2,1])),
                shape=(5,5)),
            'mm')
        y = frozenarray(x)
        self.assertIsNot(x, y)
        self.assertIs(frozenarray(y), y)
        self.assertIsNot(y, frozenarray(y, copy=True))
        self.assertFalse(y.m.data.flags['WRITEABLE'])
        self.assertTrue(x.m.data.flags['WRITEABLE'])
        self.assertTrue(np.array_equal(x.m.data, y.m.data))
        freezearray(x)
        self.assertFalse(x.m.data.flags['WRITEABLE'])
        # We can also use frozenarray as a substitute for the array function:
        x = frozenarray([1,2,3,4])
        self.assertIsInstance(x, np.ndarray)
        self.assertFalse(x.flags['WRITEABLE'])
        self.assertTrue(np.array_equal(x, [1,2,3,4]))
        # freezearray fails when given objects not compatible with arrays:
        with self.assertRaises(TypeError):
            freezearray(Ellipsis)
    def test_to_pcoll(self):
        from immlib.util import to_pcoll
        from pcollections import pdict, plist, pset
        x = pdict(a=1, b=2)
        y = plist([1,3,4,5,6])
        z = pset(['a', 'b', 'c'])
        # to_pcoll leaves persistent collections untouched.
        self.assertIs(to_pcoll(x), x)
        self.assertIs(to_pcoll(y), y)
        self.assertIs(to_pcoll(z), z)
        # They convert equivalent non-persistent objects into persistent ones.
        self.assertIsInstance(to_pcoll(dict(x)), pdict)
        self.assertEqual(to_pcoll(dict(x)), x)
        self.assertIsInstance(to_pcoll(list(y)), plist)
        self.assertEqual(to_pcoll(list(y)), y)
        self.assertIsInstance(to_pcoll(set(z)), pset)
        self.assertEqual(to_pcoll(set(z)), z)
        # If the argument isn't a valid collection, it raises an error.
        with self.assertRaises(TypeError):
            to_pcoll(10)
    def test_to_tcoll(self):
        from immlib.util import to_tcoll
        from pcollections import pdict, plist, pset, tdict, tlist, tset
        x = pdict(a=1, b=2)
        y = plist([1,3,4,5,6])
        z = pset(['a', 'b', 'c'])
        # to_tcoll makes equal copies of the transient types.
        tx = to_tcoll(x)
        ty = to_tcoll(y)
        tz = to_tcoll(z)
        self.assertEqual(tx, x)
        self.assertEqual(ty, y)
        self.assertEqual(tz, z)
        self.assertIsInstance(tx, tdict)
        self.assertIsInstance(ty, tlist)
        self.assertIsInstance(tz, tset)
        # to_tcoll always makes a copy...
        self.assertIsNot(tx, to_tcoll(tx))
        # ...unless requested not to...
        self.assertIs(tx, to_tcoll(tx, copy=False))
        # They convert equivalent non-transient objects into transient ones.
        self.assertIsInstance(to_tcoll(dict(x)), tdict)
        self.assertEqual(to_tcoll(dict(x)), x)
        self.assertIsInstance(to_tcoll(list(y)), tlist)
        self.assertEqual(to_tcoll(list(y)), y)
        self.assertIsInstance(to_tcoll(set(z)), tset)
        self.assertEqual(to_tcoll(set(z)), z)
        # If the argument isn't a valid collection, it raises an error.
        with self.assertRaises(TypeError):
            to_tcoll(10)
    def test_to_mcoll(self):
        from immlib.util import to_mcoll
        from pcollections import pdict, plist, pset, tdict, tlist, tset
        x = pdict(a=1, b=2)
        y = plist([1,3,4,5,6])
        z = pset(['a', 'b', 'c'])
        # to_mcoll makes equal copies of the mutable types.
        mx = to_mcoll(x)
        my = to_mcoll(y)
        mz = to_mcoll(z)
        self.assertEqual(mx, x)
        self.assertEqual(my, y)
        self.assertEqual(mz, z)
        self.assertIsInstance(mx, dict)
        self.assertIsInstance(my, list)
        self.assertIsInstance(mz, set)
        # to_mcoll always makes a copy...
        self.assertIsNot(mx, to_mcoll(mx))
        # ...unless requested not to.
        self.assertIs(mx, to_mcoll(mx, copy=False))
        # They convert equivalent non-transient objects into transient ones.
        self.assertIsInstance(to_mcoll(dict(x)), dict)
        self.assertEqual(to_mcoll(dict(x)), x)
        self.assertIsInstance(to_mcoll(list(y)), list)
        self.assertEqual(to_mcoll(list(y)), y)
        self.assertIsInstance(to_mcoll(set(z)), set)
        self.assertEqual(to_mcoll(set(z)), z)
        # If the argument isn't a valid collection, it raises an error.
        with self.assertRaises(TypeError):
            to_mcoll(10)
        
    # Other Utilities ##########################################################
    def test_predicates(self):
        from immlib.util import (
            is_acallable, is_lambda, is_asized, is_acontainer,
            is_aiterable, is_aiterator, is_areversible, is_acoll,
            is_abytes, is_bytes, is_ahashable, is_tuple, is_frozenset,
            is_aseq, is_amseq, is_apseq,
            is_aset, is_amset, is_apset,
            is_amap, is_ammap, is_apmap,
            is_list, is_plist, is_tlist, is_llist,
            is_set, is_pset, is_tset,
            is_dict, is_odict, is_ddict, is_pdict, is_tdict, is_ldict,
            is_pcoll, is_tcoll, is_mcoll)
        from pcollections import (
            pset, tset, pdict, tdict, plist, tlist, ldict, llist)
        from collections import (OrderedDict, defaultdict)
        # For each of these we just do one True and one False example:
        self.assertTrue(is_acallable(lambda:True))
        self.assertFalse(is_acallable(10))
        self.assertTrue(is_lambda(lambda:0))
        self.assertFalse(is_lambda(10))
        self.assertTrue(is_asized([]))
        self.assertFalse(is_asized(0))
        self.assertTrue(is_acontainer([]))
        self.assertFalse(is_acontainer(None))
        self.assertTrue(is_aiterable([]))
        self.assertFalse(is_aiterable(None))
        self.assertTrue(is_aiterator(iter('abc')))
        self.assertFalse(is_aiterator('abc'))
        self.assertTrue(is_areversible([]))
        self.assertFalse(is_areversible(set([1,2,3])))
        self.assertTrue(is_acoll([]))
        self.assertFalse(is_acoll(10))
        self.assertTrue(is_abytes(b'abc'))
        self.assertFalse(is_abytes('abc'))
        self.assertTrue(is_bytes(b'abc'))
        self.assertFalse(is_bytes('abc'))
        self.assertTrue(is_ahashable('abc'))
        self.assertFalse(is_ahashable({}))
        self.assertTrue(is_tuple((1,2,3)))
        self.assertFalse(is_tuple([]))
        self.assertTrue(is_frozenset(frozenset((1,2,3))))
        self.assertFalse(is_frozenset(set()))
        self.assertTrue(is_aseq([1,2,3]))
        self.assertFalse(is_aseq({}))
        self.assertTrue(is_amseq([]))
        self.assertTrue(is_amseq(tlist()))
        self.assertFalse(is_amseq(plist()))
        self.assertTrue(is_apseq(plist()))
        self.assertFalse(is_apseq(tlist()))
        self.assertFalse(is_apseq([]))
        self.assertTrue(is_aset(set()))
        self.assertFalse(is_aset({}))
        self.assertTrue(is_amset(tset()))
        self.assertFalse(is_amset(pset()))
        self.assertTrue(is_apset(pset()))
        self.assertFalse(is_apset(set()))
        self.assertTrue(is_amap(dict()))
        self.assertFalse(is_amap([]))
        self.assertTrue(is_ammap(tdict()))
        self.assertFalse(is_ammap(pdict()))
        self.assertTrue(is_apmap(pdict()))
        self.assertFalse(is_apmap(dict()))
        self.assertTrue(is_list([]))
        self.assertFalse(is_list(llist()))
        self.assertTrue(is_plist(plist()))
        self.assertFalse(is_plist([]))
        self.assertTrue(is_tlist(tlist()))
        self.assertFalse(is_tlist([]))
        self.assertTrue(is_llist(llist()))
        self.assertFalse(is_llist(plist()))
        self.assertTrue(is_set(set()))
        self.assertFalse(is_set(pset()))
        self.assertTrue(is_pset(pset()))
        self.assertFalse(is_pset(tset()))
        self.assertTrue(is_tset(tset()))
        self.assertFalse(is_tset(set()))
        self.assertTrue(is_dict(dict()))
        self.assertFalse(is_dict(pdict()))
        self.assertTrue(is_odict(OrderedDict()))
        self.assertFalse(is_odict(dict()))
        self.assertTrue(is_ddict(defaultdict(lambda:[])))
        self.assertFalse(is_ddict(dict()))
        self.assertTrue(is_pdict(pdict()))
        self.assertFalse(is_pdict(dict()))
        self.assertTrue(is_tdict(tdict()))
        self.assertFalse(is_tdict(pdict()))
        self.assertTrue(is_ldict(ldict()))
        self.assertFalse(is_ldict(pdict()))
        self.assertTrue(is_pcoll(pdict()))
        self.assertFalse(is_pcoll([]))
        self.assertTrue(is_tcoll(tset()))
        self.assertFalse(is_tcoll(pset()))
        self.assertTrue(is_mcoll({}))
        self.assertFalse(is_mcoll(pdict()))
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
    def test_get(self):
        from immlib import get, nestget
        # get just extracts things from simple containers; nestget does so from
        # nested containers.
        x = {'a': [1, 2, 3], 'b': [{'x':0, 'y':1}, {'x':10, 'y':11}]}
        self.assertEqual(get(x, 'a'), [1,2,3])
        self.assertEqual(nestget(x, 'a'), [1,2,3])
        self.assertEqual(nestget(x, 'a', 1), 2)
        with self.assertRaises(KeyError):
            get(x, 'q')
        with self.assertRaises(KeyError):
            get(x['a'], 'q')
        with self.assertRaises(KeyError):
            get(x['a'], 10)
        with self.assertRaises(KeyError):
            nestget(x, 'a', 5)
        with self.assertRaises(TypeError):
            nestget(x, 'b', 0, 'x', 10)
        with self.assertRaises(KeyError):
            nestget(x, 'b', 'q')
        with self.assertRaises(TypeError):
            get(None, 'q')
        self.assertEqual(get(x, 'q', default=...), ...)
        self.assertEqual(nestget(x, 'b', 4, default=...), ...)
        # Other errors that can be caused:
        with self.assertRaises(TypeError):
            get(x, 'a', other=10)
        with self.assertRaises(TypeError):
            nestget(x, 'a', other=10)
    def test_maps(self):
        from immlib.util import (
            lazyvalmap, valmap, lazykeymap, keymap, lazyitemmap, itemmap,
            dictmap, pdictmap, ldictmap)
        from pcollections import pdict, ldict, lazy
        d = dict(a=1, b=2, c=3)
        make_ld = lambda:ldict(a=lazy(lambda:1), b=lazy(lambda:2), c=3)
        # lazyvalmap
        md = lazyvalmap(lambda x: x+1, d)
        self.assertIsInstance(md, ldict)
        self.assertTrue(md.is_lazy('a'))
        self.assertFalse(md.is_ready('a'))
        self.assertEqual(md['a'], 2)
        self.assertTrue(md.is_ready('a'))
        mmd = lazyvalmap(lambda x: x+1, md)
        self.assertIsInstance(mmd, ldict)
        self.assertTrue(mmd.is_lazy('a'))
        self.assertTrue(md.is_lazy('b'))
        self.assertFalse(mmd.is_ready('a'))
        self.assertFalse(md.is_ready('b'))
        self.assertEqual(mmd['a'], 3)
        self.assertEqual(md, dict(a=2, b=3, c=4))
        self.assertEqual(mmd, dict(a=3, b=4, c=5))
        # valmap
        ld = make_ld()
        md = valmap(lambda x: x+1, d)
        mld = valmap(lambda x: x+1, ld)
        mpd = valmap(lambda x: x+1, pdict(d))
        self.assertIs(type(md), dict)
        self.assertIs(type(mpd), pdict)
        self.assertIs(type(mld), ldict)
        self.assertTrue(mld.is_lazy('a'))
        self.assertEqual(mld['a'], 2)
        self.assertTrue(mld.is_ready('a'))
        self.assertEqual(md, dict(a=2, b=3, c=4))
        self.assertEqual(mpd, dict(a=2, b=3, c=4))
        self.assertEqual(mld, dict(a=2, b=3, c=4))
        # lazykeymap
        md = lazykeymap(lambda x:x, d)
        self.assertIsInstance(md, ldict)
        self.assertTrue(md.is_lazy('a'))
        self.assertFalse(md.is_ready('a'))
        self.assertEqual(md['a'], 'a')
        self.assertTrue(md.is_ready('a'))
        self.assertEqual(md, dict(a='a', b='b', c='c'))
        # keymap
        ld = make_ld()
        md = keymap(lambda x: x, d)
        mld = keymap(lambda x: x, ld)
        mpd = keymap(lambda x: x, pdict(d))
        self.assertIs(type(md), dict)
        self.assertIs(type(mpd), pdict)
        self.assertIs(type(mld), pdict)
        self.assertEqual(mld['a'], 'a')
        self.assertEqual(md, dict(a='a', b='b', c='c'))
        self.assertEqual(mpd, dict(a='a', b='b', c='c'))
        self.assertEqual(mld, dict(a='a', b='b', c='c'))
        # lazyitemmap
        md = lazyitemmap(lambda x,y: x*y, d)
        self.assertIsInstance(md, ldict)
        self.assertTrue(md.is_lazy('a'))
        self.assertFalse(md.is_ready('a'))
        self.assertEqual(md['a'], 'a')
        self.assertTrue(md.is_ready('a'))
        mmd = lazyitemmap(lambda x,y: x*y, ldict(d))
        self.assertIsInstance(mmd, ldict)
        self.assertTrue(mmd.is_lazy('a'))
        self.assertTrue(md.is_lazy('b'))
        self.assertFalse(mmd.is_ready('a'))
        self.assertFalse(md.is_ready('b'))
        self.assertEqual(mmd['a'], 'a')
        self.assertEqual(md, dict(a='a', b='bb', c='ccc'))
        self.assertEqual(mmd, dict(a='a', b='bb', c='ccc'))
        # itemmap
        ld = make_ld()
        md = itemmap(lambda x,y: x*y, d)
        mld = itemmap(lambda x,y: x*y, ld)
        mpd = itemmap(lambda x,y: x*y, pdict(d))
        self.assertIs(type(md), dict)
        self.assertIs(type(mpd), pdict)
        self.assertIs(type(mld), ldict)
        self.assertTrue(mld.is_lazy('a'))
        self.assertEqual(mld['a'], 'a')
        self.assertTrue(mld.is_ready('a'))
        self.assertEqual(md, dict(a='a', b='bb', c='ccc'))
        self.assertEqual(mpd, dict(a='a', b='bb', c='ccc'))
        self.assertEqual(mld, dict(a='a', b='bb', c='ccc'))
        # dictmap
        ks = list('abcdefghij')
        ref = dict(zip(ks, range(10)))
        d = dictmap(lambda k: ref[k] + 1, ks)
        pd = pdictmap(lambda k: ref[k] + 1, ks)
        ld = ldictmap(lambda k: ref[k] + 1, ks)
        self.assertIs(type(pd), pdict)
        self.assertIs(type(ld), ldict)
        self.assertEqual(valmap(lambda x:x+1, ref), d)
        self.assertEqual(valmap(lambda x:x+1, ref), pd)
        self.assertTrue(ld.is_lazy('a'))
        self.assertFalse(ld.is_ready('a'))
        self.assertEqual(ld['a'], 1)
        self.assertTrue(ld.is_ready('a'))
        self.assertEqual(valmap(lambda x:x+1, ref), ld)
    def test_merge(self):
        from immlib.util import merge, rmerge
        from pcollections import ldict, lazy
        d1 = dict(a=1, b=2, c=3)
        d2 = dict(b=3, c=4, d=5)
        d3 = ldict(c=lazy(lambda:5), d=6, e=lazy(lambda:7))
        d = merge(d1, d2, d3, a=0)
        self.assertIs(type(d), ldict)
        self.assertTrue(d.is_lazy('c'))
        self.assertTrue(d.is_lazy('e'))
        self.assertFalse(d.is_ready('c'))
        self.assertFalse(d.is_ready('e'))
        self.assertFalse(d.is_lazy('d'))
        self.assertEqual(d['c'], 5)
        self.assertTrue(d.is_ready('c'))
        self.assertFalse(d.is_ready('e'))
        self.assertEqual(d, dict(a=0, b=3, c=5, d=6, e=7))
        d3 = ldict(c=lazy(lambda:5), d=6, e=lazy(lambda:7))
        d = rmerge(d1, d2, d3, a=0)
        self.assertIs(type(d), ldict)
        self.assertFalse(d.is_lazy('c'))
        self.assertTrue(d.is_lazy('e'))
        self.assertFalse(d.is_ready('e'))
        self.assertEqual(d['c'], 3)
        self.assertEqual(d['e'], 7)
        self.assertTrue(d.is_ready('e'))
        self.assertEqual(d, dict(a=1, b=2, c=3, d=5, e=7))
    def test_assoc(self):
        from immlib.util import assoc, dissoc
        from pcollections import pdict
        with self.assertRaises(ValueError):
            assoc({}, 'a')
        with self.assertRaises(TypeError):
            assoc('abc', 'a', 1)
        with self.assertRaises(TypeError):
            dissoc('abc', 'a')
        d = dict(a=1, b=2, c=3)
        self.assertEqual(assoc(d, d=4), dict(d, d=4))
        self.assertEqual(assoc(d, 'd', 4), dict(d, d=4))
        self.assertEqual(assoc(d, 'd', 4, e=5), dict(d, d=4, e=5))
        self.assertIsNot(assoc(d, 'd', 4), d)
        self.assertEqual(dissoc(d, 'b'), dict(a=1, c=3))
        self.assertEqual(dissoc(d, 'b', 'c'), dict(a=1))
        self.assertIsNot(dissoc(d, 'b'), d)
        d = pdict(a=1, b=2, c=3)
        self.assertEqual(assoc(d, d=4), dict(d, d=4))
        self.assertEqual(assoc(d, 'd', 4), dict(d, d=4))
        self.assertEqual(assoc(d, 'd', 4, e=5), dict(d, d=4, e=5))
        self.assertIsNot(assoc(d, 'd', 4), d)
        self.assertEqual(dissoc(d, 'b'), dict(a=1, c=3))
        self.assertEqual(dissoc(d, 'b', 'c'), dict(a=1))
        self.assertIsNot(dissoc(d, 'b'), d)
    def test_lambdadict(self):
        from immlib.util import lambdadict
        from pcollections import ldict
        d = lambdadict(a=1, b=2, c=lambda a,b: a+b)
        self.assertIsInstance(d, ldict)
        self.assertTrue(d.is_lazy('c'))
        self.assertFalse(d.is_ready('c'))
        self.assertEqual(d['c'], 3)
        self.assertTrue(d.is_ready('c'))
        self.assertEqual(d, dict(a=1, b=2, c=3))
    def test_args(self):
        from immlib.util import args, argfilter
        # First test the args type:
        (a, kw) = args(1, 2, 3, a=1, b=2)
        self.assertEqual(a, (1,2,3))
        self.assertEqual(kw, dict(a=1, b=2))
        aa = args(1, 2, 3, a=1, b=2)
        self.assertIs(type(aa), args)
        self.assertEqual(aa, ((1,2,3), {'a':1,'b':2}))
        bb = aa.copy(args=(2,3,4))
        self.assertEqual(bb, ((2,3,4), {'a':1,'b':2}))
        cc = aa.copy(kwargs={'c':3})
        self.assertEqual(cc, ((1,2,3), {'c':3}))
        # Next test the argfilter:
        @argfilter
        def fix_angle(angle, *, unit):
            angle = np.asarray(angle)
            if unit == 'degrees':
                angle = np.pi / 180 * angle
            elif unit != 'radians':
                raise ValueError(f'unrecognized unit: {unit}')
            return (angle,)
        @fix_angle
        def cos_halfangle(angle, unit='radians'):
            return np.cos(angle / 2)
        self.assertEqual(cos_halfangle(0), 1.0)
        self.assertEqual(cos_halfangle(0, unit='degrees'), 1.0)
        self.assertTrue(np.abs(cos_halfangle(np.pi)) < 1e-9)
        self.assertTrue(np.abs(cos_halfangle(180, unit='degrees')) < 1e-9)
    def test_unitregistry(self):
        from immlib.util import unitregistry
        from immlib import unit, units, mag, quant, default_ureg
        with default_ureg(units):
            q = quant(10.5, 'mm')
            p = quant(55, 's')
            self.assertIs(units, unitregistry(units))
            self.assertIs(units, unitregistry(...))
            self.assertIs(units, unitregistry(q))
            self.assertIs(units, unitregistry(p))
            self.assertIs(units, unitregistry(q.u))
            self.assertIs(units, unitregistry(p.u))
            with self.assertRaises(TypeError):
                unitregistry(units, None, 1)
            with self.assertRaises(TypeError):
                unitregistry()
            with self.assertRaises(TypeError):
                unitregistry(units, None, 1)
            

