# -*- coding: utf-8 -*-
################################################################################
# immlib/test/util/test_pint.py
#
# Checks the parts of Pint's implementation that immlib relies on.
#
# immlib.Quantity subclasses pint.Quantity and does much of its work by
# overriding Pint's own private methods and reading its private attributes:
# `_add_sub` / `_iadd_sub` / `_mul_div` / `_imul_div` (the choke points that
# Pint's +, -, *, and / funnel through, which is how immlib gives its
# `units=None` quantities their own semantics), `_magnitude`, `_units`,
# `_REGISTRY`, `_dimensionality`, `_convert_magnitude`, `_is_multiplicative`,
# the `UnitsContainer` helper, and Pint's NumPy dispatch hooks. None of that is
# part of Pint's public API, so a new Pint release is free to change or remove
# it. These tests are the tripwire: they fail when Pint changes a piece immlib
# depends on, so the incompatibility is found here--with a message naming the
# broken assumption--rather than as a silently wrong result at run time. They
# are also what justifies the Pint upper bound declared in pyproject.toml.


# Dependencies #################################################################

from unittest import TestCase

import numpy as np
import pint

from immlib import Quantity, UnitRegistry, quant


class TestPintContract(TestCase):
    """Tests the Pint internals that immlib depends on."""

    def test_supported_pint_version(self):
        """The installed Pint is within the range immlib declares support for.

        Bumping this test is the deliberate act of claiming a new Pint works;
        the rest of this file is what justifies the claim.
        """
        parts = pint.__version__.split('.')
        (major, minor) = (int(parts[0]), int(parts[1]))
        self.assertEqual(
            (major, minor), (0, 24),
            f"Pint {pint.__version__} is outside immlib's declared support"
            f" range (>= 0.24, < 0.25); verify the internals this file checks"
            f" still hold, then update pyproject.toml and this test")

    def test_dispatch_methods_exist(self):
        """The private dispatch methods immlib overrides exist on
        pint.Quantity."""
        for name in ('_add_sub', '_iadd_sub', '_mul_div', '_imul_div'):
            self.assertTrue(
                hasattr(pint.Quantity, name),
                f"pint.Quantity has no {name}; immlib's override of it would"
                f" silently stop being called")

    def test_immlib_overrides_the_dispatch_methods(self):
        """immlib.Quantity overrides the private dispatch methods it relies on
        (rather than inheriting Pint's)."""
        for name in ('_add_sub', '_iadd_sub', '_mul_div', '_imul_div'):
            self.assertIsNot(
                getattr(Quantity, name), getattr(pint.Quantity, name),
                f"immlib.Quantity does not override {name}")

    def test_add_routes_through_add_sub(self):
        """``+`` and ``-`` are dispatched through ``_add_sub``.

        This is the property that lets immlib give its ``units=None``
        quantities their own arithmetic; a Pint that stopped calling
        ``_add_sub`` would break it.
        """
        calls = []
        class Probe(pint.Quantity):
            def _add_sub(self, *args, **kwargs):
                calls.append(1)
                return super()._add_sub(*args, **kwargs)
        _ = Probe(3.0, 'm') + Probe(4.0, 'm')
        self.assertEqual(len(calls), 1)

    def test_mul_routes_through_mul_div(self):
        """``*`` and ``/`` are dispatched through ``_mul_div``."""
        calls = []
        class Probe(pint.Quantity):
            def _mul_div(self, *args, **kwargs):
                calls.append(1)
                return super()._mul_div(*args, **kwargs)
        _ = Probe(3.0, 'm') * Probe(4.0, 'm')
        self.assertEqual(len(calls), 1)

    def test_private_attributes_exist(self):
        """The private attributes immlib reads off any pint.Quantity exist."""
        q = pint.Quantity(1.0, 'm')
        for name in ('_magnitude', '_units', '_REGISTRY', '_dimensionality'):
            self.assertTrue(hasattr(q, name),
                            f"pint.Quantity has no {name}")
        self.assertTrue(
            hasattr(q, 'UnitsContainer'),
            "pint.Quantity has no UnitsContainer helper")

    def test_conversion_helpers_exist(self):
        """The conversion helpers immlib calls exist on pint.Quantity."""
        q = pint.Quantity(1.0, 'm')
        for name in ('_convert_magnitude', '_is_multiplicative'):
            self.assertTrue(hasattr(q, name), f"pint.Quantity has no {name}")

    def test_numpy_dispatch_hooks_exist(self):
        """Pint implements NumPy dispatch, which immlib extends for its
        ``units=None`` quantities and falls back to otherwise."""
        for name in ('__array_ufunc__', '__array_function__'):
            self.assertTrue(
                hasattr(pint.Quantity, name),
                f"pint.Quantity has no {name}; immlib's override has nothing"
                f" to fall back to")

    def test_dimensionality_error_exists(self):
        """immlib raises ``pint.DimensionalityError`` for unit mismatches."""
        self.assertTrue(hasattr(pint, 'DimensionalityError'))
        self.assertTrue(issubclass(pint.DimensionalityError, Exception))

    def test_registry_produces_immlib_quantities(self):
        """An immlib.UnitRegistry produces immlib.Quantity objects, and a
        plain pint.UnitRegistry does not."""
        self.assertTrue(issubclass(UnitRegistry.Quantity, Quantity))
        self.assertIsInstance(UnitRegistry()('1 m'), Quantity)
        self.assertNotIsInstance(pint.UnitRegistry()('1 m'), Quantity)

    def test_units_none_is_immlib_specific(self):
        """The ``units=None`` convention is immlib's: a plain pint registry
        cannot represent it, which is why immlib.Quantity must override Pint's
        arithmetic and conversion methods at all."""
        q = quant(np.arange(3.0), None, ureg=UnitRegistry())
        self.assertIsNone(q.units)
        with self.assertRaises(ValueError):
            quant(np.arange(3.0), None, ureg=pint.UnitRegistry())
