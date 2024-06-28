# -*- coding: utf-8 -*-
################################################################################
# immlib/test/util/test_quantity.py
#
# Tests of the quantity module in immlib: i.e., tests for the code in the
# immlib.util._quantity module.


# Dependencies #################################################################

from unittest import TestCase

class TestUtilQuantity(TestCase):
    """Tests the immlib.util._quantity module."""
    # Pint Utilities ###########################################################
    def test_is_ureg(self):
        from immlib import (units, is_ureg)
        from pint import UnitRegistry
        # immlib.units is a registry.
        self.assertTrue(is_ureg(units))
        # So is any new UnitsRegistry we create.
        self.assertTrue(is_ureg(UnitRegistry()))
        # Other objects are not.
        self.assertFalse(is_ureg(None))
    def test_is_unit(self):
        from immlib import (units, is_unit)
        from pint import UnitRegistry
        # We will use an alternate unit registry in some tests.
        alt_units = UnitRegistry()
        # Units from any unit registry are allowed by default.
        self.assertTrue(is_unit(units.mm))
        self.assertTrue(is_unit(units.gram))
        self.assertTrue(is_unit(alt_units.mm))
        self.assertTrue(is_unit(alt_units.gram))
        # Things that aren't units are never units.
        self.assertFalse(is_unit('mm'))
        self.assertFalse(is_unit(10.0))
        self.assertFalse(is_unit(None))
        self.assertFalse(is_unit(10.0 * units.mm))
        # If the ureg parameter is ..., then only units from the immlib units
        # registry are allowed.
        self.assertTrue(is_unit(units.mm, ureg=...))
        self.assertFalse(is_unit(alt_units.mm, ureg=...))
        # Alternately, the ureg parameter may be a specific unit registry.
        self.assertFalse(is_unit(units.mm, ureg=alt_units))
        self.assertTrue(is_unit(alt_units.mm, ureg=alt_units))
    def test_is_quant(self):
        from immlib import (units, is_quant)
        from pint import UnitRegistry
        # We will use an alternate unit registry in some tests.
        alt_units = UnitRegistry()
        # By default, it does not matter what registry a quantity comes from;
        # it is considered a quantity.
        q = 10.0 * units.mm
        alt_q = 10.0 * alt_units.mm
        self.assertTrue(is_quant(q))
        self.assertTrue(is_quant(alt_q))
        # Other objects aren't quantities.
        self.assertFalse(is_quant(10.0))
        self.assertFalse(is_quant(units.mm))
        self.assertFalse(is_quant(None))
        # We can require that a quantity meet a certain kind of unit type.
        self.assertTrue(is_quant(q, unit=units.mm))
        self.assertTrue(is_quant(q, unit='inches'))
        self.assertFalse(is_quant(q, unit=units.grams))
        self.assertFalse(is_quant(q, unit='seconds'))
        # The ureg parameter changes whether any unit registry is allowed (the
        # default, or ureg=None), only immlib.units is allowed (ureg=Ellipsis),
        # or a specific unit registry is allowed.
        self.assertTrue(is_quant(q, ureg=...))
        self.assertFalse(is_quant(alt_q, ureg=...))
        self.assertFalse(is_quant(q, ureg=alt_units))
        self.assertTrue(is_quant(alt_q, ureg=alt_units))
    def test_default_ureg(self):
        from immlib import default_ureg
        from pint import UnitRegistry
        # You can set the immlib default units registry (immlib.units) temporarily
        # in an execution context using the default_ureg function:
        ureg = UnitRegistry()
        with default_ureg(ureg):
            from immlib import units
            self.assertIs(units, ureg)
        # This only affects the code inside the with-block.
        from immlib import units
        self.assertIsNot(units, ureg)
    def test_like_unit(self):
        from immlib import (like_unit, units)
        # like_unit returns True when its argument is like a unit. This can be,
        # for one, objects that already are units.
        self.assertTrue(like_unit(units.mm))
        self.assertTrue(like_unit(units.count))
        # Otherwise, only strings may be unit-like.
        self.assertFalse(like_unit(None))
        self.assertFalse(like_unit(10))
        self.assertFalse(like_unit([]))
        # Strings that name units are unit-like.
        self.assertTrue(like_unit('mm'))
        self.assertTrue(like_unit('count'))
    def test_unit(self):
        from immlib import (unit, units)
        from immlib.util import unitregistry
        from pint.errors import UndefinedUnitError
        from pint import UnitRegistry
        # unit converts its argument into a unit. Units themselves are returned
        # as-is.
        u = units.mm
        self.assertIs(u, unit(u))
        u = units.count
        self.assertIs(u, unit(u))
        # Strings that name units can be converted into units.
        self.assertEqual(units.mm, unit('mm'))
        self.assertEqual(units.count, unit('count'))
        # If the argument isn't a valid unit, then an error is raised.
        with self.assertRaises(ValueError): unit(None)
        with self.assertRaises(ValueError): unit(10)
        with self.assertRaises(ValueError): unit([])
        with self.assertRaises(UndefinedUnitError): unit('fdjsklfajdk')
        # If Ellipsis is passed for the ureg argument of unit, it converts the
        # unit from another unit registry into the immlib.units registry.
        ureg = UnitRegistry()
        self.assertIsNot(ureg.mm, unit(ureg.mm, ...))
        self.assertIs(unitregistry(unit(ureg.count, ...)), units)
        # Similarly we can convert back to the other ureg.
        self.assertIsNot(units.mm, unit(units.mm, ureg))
        self.assertIs(unitregistry(unit(units.count, ureg)), ureg)
    def test_alike_units(self):
        from immlib import (alike_units, units)
        # alike_units tells us if two units are of the same unit category, like
        # meters and feet both being lengths.
        self.assertTrue(alike_units(units.mm, units.feet))
        self.assertTrue(alike_units(units.seconds, units.days))
        self.assertTrue(alike_units(units.rad, units.degree))
        self.assertFalse(alike_units(units.rad, units.feet))
        self.assertFalse(alike_units(units.days, units.mm))
        self.assertFalse(alike_units(units.mm, units.degree))
    def test_quant(self):
        from immlib import (default_ureg, units, quant)
        from pint import (UnitRegistry, Quantity)
        import torch, numpy as np
        # The quant function lets you create pint quantities; by default these
        # are registered in the immlib.units UnitRegistry.
        circ = np.linspace(0, 1, 25)
        self.assertIsInstance(quant(10, 'mm'), Quantity)
        self.assertIsInstance(quant([10, 30, 40], 'days'), Quantity)
        self.assertIsInstance(quant(circ, 'turns'), Quantity)
        self.assertEqual(quant(10, 'mm').m, 10)
        self.assertTrue(np.array_equal(quant(circ, 'turns').m, circ))
        # Iterables are upgraded to numpy arrays when applicable.
        self.assertIsInstance(quant([10, 30, 40], 'days').m, np.ndarray)
        self.assertTrue(np.array_equal(quant([10, 30, 40], 'days').m,
                                       [10, 30, 40]))
        # Units are registered in the immlib.units registry by default.
        self.assertEqual(quant(10, 'mm').u, units.mm)
        self.assertEqual(quant([10, 30, 40], 'days').u, units.days)
        self.assertEqual(quant(circ, 'turns').u, units.turns)
        # Tensors also work as quantities.
        t = torch.linspace(0,1,5)
        tq = quant(t, 'mm')
        self.assertIsInstance(tq.m, torch.Tensor)
        self.assertIs(tq.m, t)
        # Changing the default unit registry changes how these are registered.
        ureg = UnitRegistry()
        with default_ureg(ureg):
            q = quant(10, 'mm')
        self.assertIsInstance(q, ureg.Quantity)
        self.assertFalse(isinstance(q, units.Quantity))
        # This can also be done with the ureg option.
        q = quant(10, 'mm', ureg=ureg)
        self.assertIsInstance(q, ureg.Quantity)
        self.assertFalse(isinstance(q, units.Quantity))
    def test_mag(self):
        from immlib import (mag, units, quant)
        import numpy as np
        from pint.errors import DimensionalityError
        # The mag function extracts the magnitude from a quantity.
        m = np.linspace(0, 1, 5)
        q = quant(m, 'second')
        self.assertIs(m, mag(q))
        self.assertEqual(10, mag(10 * units.mm))
        # mag can extract in the quantity's native units if none are given, or
        # in another unit, if requested.
        self.assertTrue(np.array_equal(m * 1000, mag(q, 'ms')))
        # If the value passed to mag is not a quantity, it is returned as-is,
        # and is assumed to be in the correct unit.
        self.assertIs(m, mag(m))
        # The same is true, even if a unit is passed: non-quantities are always
        # assumed to be in the correct units.
        self.assertIs(m, mag(m, 'ms'))
        self.assertIs(m, mag(m, 'days'))
        self.assertIs(m, mag(m, 'feet'))
        # If unit=None, then the argument must not be a quantity.
        self.assertIs(m, mag(m, None))
        with self.assertRaises(ValueError): mag(q, None)
        # If the unit doesn't match, a DimensionalityError is raised.
        with self.assertRaises(DimensionalityError): mag(q, 'miles')
    def test_promote(self):
        from immlib import (promote, units, quant)
        import torch, numpy as np
        # promote converts all arguments into quantities.
        a = np.linspace(0, 1, 5)
        b = np.arange(10)
        (qa, qb) = promote(a, b)
        self.assertNotIsInstance(qa, units.Quantity)
        self.assertNotIsInstance(qb, units.Quantity)
        self.assertIs(qa, a)
        self.assertIs(qb, b)
        # If one of the arguments is a tensor, all results will be tensors.
        c = quant(torch.linspace(0, 1, 5), 'mm')
        (qa, qb, qc) = promote(a, b, c)
        self.assertNotIsInstance(qa, units.Quantity)
        self.assertNotIsInstance(qb, units.Quantity)
        self.assertIsInstance(qc, units.Quantity)
        self.assertIsInstance(qa, torch.Tensor)
        self.assertIsInstance(qb, torch.Tensor)
        self.assertIsInstance(qc.m, torch.Tensor)
        self.assertTrue(np.array_equal(qa.numpy(), a))
        self.assertTrue(np.array_equal(qb.numpy(), b))
        self.assertIs(qc, c)
