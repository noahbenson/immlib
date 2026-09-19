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

from immlib.workflow import CALC_DOC_SECTIONS


def immlib_modules():
    """Yields every module of the ``immlib`` package except the tests."""
    import importlib
    import pkgutil
    import immlib
    yield immlib
    for m in pkgutil.walk_packages(immlib.__path__, 'immlib.'):
        if '.test' in m.name:
            continue
        try:
            yield importlib.import_module(m.name)
        except ImportError:      # pragma: no cover - an optional dependency
            continue

def public_objects():
    """Yields ``(name, object)`` for every public function, class, method and
    property of the ``immlib`` package that has a docstring.

    Every module is walked, not only the top-level ``immlib`` and
    ``immlib.math`` namespaces, and the members of a public class are walked
    along with the class: a docstring that no one imports is still
    documentation, and the two defects this test first caught were both in
    places a namespace-only walk does not reach.
    """
    seen = set()
    def members(obj, qualname):
        for (name, value) in vars(obj).items():
            if name.startswith('_'):
                continue
            if isinstance(value, property):
                value = value.fget
            elif isinstance(value, (staticmethod, classmethod)):
                value = value.__func__
            if not (inspect.isfunction(value) or inspect.isclass(value)):
                continue
            if getattr(value, '__module__', '') and \
                    not value.__module__.startswith('immlib'):
                continue
            if id(value) in seen:
                continue
            seen.add(id(value))
            # A docstring of the object's own, not one inherited from a base
            # class: immlib is answerable for the documentation it writes.
            if value.__doc__ and value.__doc__.strip():
                yield (f'{qualname}.{name}', value)
            if inspect.isclass(value):
                yield from members(value, f'{qualname}.{name}')
    for mod in immlib_modules():
        yield from members(mod, mod.__name__)

class TestDocs(TestCase):
    """Tests the validity of immlib's own documentation."""
    def test_docstrings_parse(self):
        "Every public docstring parses as NumPy-style documentation."
        count = 0
        for (name, obj) in public_objects():
            with self.subTest(name=name):
                try:
                    doc = docparse(obj, format='numpy',
                                   custom=CALC_DOC_SECTIONS)
                except DocShareError as e:
                    self.fail(f"{name}: {type(e).__name__}: {e}")
                self.assertTrue(doc.summary, f"{name} has no summary line")
                count += 1
        # Guard against the collection silently finding nothing.
        self.assertGreater(count, 200)
    def test_docstrings_match_signatures(self):
        """Every public docstring documents only parameters that exist.

        A parameter that a function takes from ``**kwargs`` is documented
        deliberately although the signature does not name it, so for a
        function that has a ``**kwargs`` the documented extras are declared
        with ``extraparam`` rather than reported. A function without one has
        nowhere for an extra name to come from, so it is reported.
        """
        for (name, obj) in public_objects():
            if not inspect.isfunction(obj):
                continue
            with self.subTest(name=name):
                try:
                    extra = self._kwargs_params(obj)
                    docwrap(format='numpy', custom=CALC_DOC_SECTIONS,
                            extraparam=extra)(obj)
                except DocShareError as e:
                    self.fail(f"{name}: {type(e).__name__}: {e}")
    @staticmethod
    def _kwargs_params(obj):
        """Returns the parameters `obj` documents that its signature does not
        name, when it has a ``**kwargs`` for them to come from."""
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):     # pragma: no cover
            return ()
        params = sig.parameters
        if not any(p.kind is p.VAR_KEYWORD for p in params.values()):
            return ()
        doc = docparse(obj, format='numpy', custom=CALC_DOC_SECTIONS)
        section = doc.section('parameters')
        if section is None:
            return ()
        return tuple(nm for item in section.items for nm in item.names
                     if nm not in params)
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

    def test_math_documents_its_arguments(self):
        """Every public immlib.math function documents every argument it
        takes and the value it returns.

        The descriptions are inherited from the prototypes in
        immlib.math._core (see its "Shared documentation" section), so a
        function that takes a familiar argument needs no new text; this
        test is what keeps a new function from being added without any.
        """
        import immlib.math as im
        for name in im.__all__:
            obj = getattr(im, name)
            if not inspect.isfunction(obj):
                continue
            with self.subTest(name=f'immlib.math.{name}'):
                doc = docparse(obj, format='numpy')
                sections = {s.name.lower() for s in doc.sections}
                params = inspect.signature(obj).parameters
                named = [p for p in params if p != 'kwargs']
                if named:
                    self.assertIn(
                        'parameters', sections,
                        f"immlib.math.{name} documents no parameters")
                    documented = {nm for item in doc.section('parameters').items
                                  for nm in item.names}
                    for p in named:
                        self.assertIn(
                            p, documented,
                            f"immlib.math.{name} does not document '{p}'")
                self.assertIn('returns', sections,
                              f"immlib.math.{name} documents no return value")

    def test_quantity_documents_its_members(self):
        """Quantity's own methods and properties are documented here rather
        than left to inherit Pint's, which describe Pint's behavior and not
        immlib's."""
        import immlib
        cls = immlib.Quantity
        for name in ('units', 'u', 'dimensionless', 'dimensionality',
                     'backend', 'check', 'to', 'ito', 'm_as', 'compare',
                     'sum', 'mean', 'reshape', 'astype'):
            with self.subTest(name=f'Quantity.{name}'):
                member = cls.__dict__.get(name)
                self.assertIsNotNone(
                    member, f"Quantity has no member '{name}'")
                fn = member.fget if isinstance(member, property) else member
                self.assertTrue(
                    fn.__doc__ and fn.__doc__.strip(),
                    f"Quantity.{name} has no docstring of its own")

    def test_sections_are_in_numpy_order(self):
        """Parameters precedes Returns in every public docstring.

        docshare appends an inherited section after the sections an object
        documents for itself, so a function that writes its own Returns and
        inherits its Parameters gets them in the wrong order unless it
        declares an empty Parameters header for them to fill. Sphinx and
        numpydoc both expect the standard order, so it is checked here.
        """
        order = ('Parameters', 'Returns', 'Raises')
        for (name, obj) in public_objects():
            doc = inspect.getdoc(obj)
            positions = [(doc.find(f'{s}\n{"-" * len(s)}'), s) for s in order]
            positions = [(i, s) for (i, s) in positions if i >= 0]
            with self.subTest(name=name):
                self.assertEqual(
                    positions, sorted(positions),
                    f"{name}'s sections are out of order:"
                    f" {[s for (_, s) in positions]}")
