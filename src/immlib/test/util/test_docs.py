# -*- coding: utf-8 -*-
################################################################################
# immlib/test/util/test_docs.py
#
# Tests that immlib's own documentation is valid: every public object's
# docstring must parse as NumPy-style documentation and must agree with the
# object's signature.


# Dependencies #################################################################

import inspect
from unittest import TestCase

from docshare import (docparse, docwrap, DocShareError)


def public_objects():
    """Yields ``(name, object)`` for every public function and class in the
    ``immlib`` and ``immlib.math`` namespaces that has a docstring."""
    import immlib
    import immlib.math
    seen = set()
    for mod in (immlib, immlib.math):
        modname = mod.__name__
        for name in getattr(mod, '__all__', ()) or dir(mod):
            if name.startswith('_'):
                continue
            obj = getattr(mod, name, None)
            if not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            if inspect.getdoc(obj):
                yield (f'{modname}.{name}', obj)

class TestDocs(TestCase):
    """Tests the validity of immlib's own documentation."""
    def test_docstrings_parse(self):
        "Every public docstring parses as NumPy-style documentation."
        count = 0
        for (name, obj) in public_objects():
            with self.subTest(name=name):
                try:
                    doc = docparse(obj, format='numpy')
                except DocShareError as e:
                    self.fail(f"{name}: {type(e).__name__}: {e}")
                self.assertTrue(doc.summary, f"{name} has no summary line")
                count += 1
        # Guard against the collection silently finding nothing.
        self.assertGreater(count, 100)
    def test_docstrings_match_signatures(self):
        "Every public docstring documents only parameters that exist."
        for (name, obj) in public_objects():
            if not inspect.isfunction(obj):
                continue
            with self.subTest(name=name):
                try:
                    docwrap(format='numpy')(obj)
                except DocShareError as e:
                    self.fail(f"{name}: {type(e).__name__}: {e}")
    def test_inherited_documentation(self):
        "Documentation inherited from other functions is included."
        from immlib import (streq, strcmp, is_quant, is_unit, to_dense,
                            to_numeric)
        # streq inherits strcmp's parameters.
        for param in ('case', 'unicode', 'strip', 'split'):
            self.assertIn(f'{param} :', streq.__doc__)
        self.assertIn('Parameters', streq.__doc__)
        # The inherited Parameters section precedes the Returns section.
        self.assertLess(streq.__doc__.index('Parameters'),
                        streq.__doc__.index('Returns'))
        # is_quant inherits is_unit's Raises section.
        self.assertIn('TypeError', is_quant.__doc__)
        self.assertIn('pint.UnitRegistry', is_quant.__doc__)
        # to_dense inherits to_numeric's parameters.
        for param in ('dtype', 'quant', 'ureg', 'unit'):
            self.assertIn(f'{param} :', to_dense.__doc__)
        # Nothing is left interpolated.
        for obj in (streq, strcmp, is_quant, is_unit, to_dense, to_numeric):
            self.assertNotIn('%(', obj.__doc__)
