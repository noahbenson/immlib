# -*- coding: utf-8 -*-
################################################################################
# immlib/test/types/test_core.py

"""Tests of the core types module in immlib: i.e., tests for the code in the
immlib.types._core module.
"""


# Dependencies #################################################################

from unittest import TestCase


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

