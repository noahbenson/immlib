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
        # Anything that isn't a unit, a quantity, or a unit-name string
        # (None, a plain number, etc.) is treated as dimensionless for this
        # comparison. This fallback goes through pint's own
        # is_compatible_with, which relies on pint's internal dimensionless
        # parsing continuing to work correctly (see
        # test_quantity_omitted_units).
        self.assertTrue(alike_units(None, 'dimensionless'))
        self.assertTrue(alike_units(None, None))
        self.assertTrue(alike_units(5, 'dimensionless'))
        self.assertFalse(alike_units(None, units.mm))
        self.assertFalse(alike_units(5, units.mm))
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
    # The Quantity Type ########################################################
    def test_quantity_class(self):
        from immlib import Quantity, quant, units
        import pint, numpy as np, torch
        # quant() produces instances of immlib.Quantity, which remains a
        # true pint.Quantity, and units.Quantity is patched to be (a
        # subclass of) immlib.Quantity.
        q = quant(10, 'mm')
        self.assertIsInstance(q, Quantity)
        self.assertIsInstance(q, pint.Quantity)
        self.assertTrue(issubclass(units.Quantity, Quantity))
        # The backend property reports numpy or torch depending on the
        # magnitude's type.
        self.assertIs(quant(np.array([1, 2, 3]), 'mm').backend, np)
        self.assertIs(quant(torch.tensor([1, 2, 3]), 'mm').backend, torch)
    def test_quantity_none_units(self):
        from immlib import Quantity, quant, mag, is_quant, units
        # quant(x) and quant(x, None) both produce a Quantity whose units
        # are None--immlib's representation of "no units"--rather than
        # raising or defaulting to dimensionless.
        q1 = quant(10)
        q2 = quant(10, None)
        self.assertIsInstance(q1, Quantity)
        self.assertIsNone(q1.units)
        self.assertIsNone(q1.u)
        self.assertIsNone(q2.units)
        self.assertEqual(q1.m, 10)
        # This is distinct from dimensionless.
        qd = quant(10, 'dimensionless')
        self.assertIsNotNone(qd.units)
        # is_quant(x, unit=None) matches a None-units quantity (and a
        # non-quantity), but not a quantity with real (even dimensionless)
        # units.
        self.assertTrue(is_quant(q1, unit=None))
        self.assertFalse(is_quant(10, unit=None))
        self.assertFalse(is_quant(qd, unit=None))
        # mag() with unit=None succeeds for a None-units quantity (it
        # returns the magnitude), but still raises for a quantity with real
        # units.
        self.assertEqual(mag(q1, None), 10)
        with self.assertRaises(ValueError):
            mag(qd, None)
        # quant(q, None) always succeeds, stripping whatever units q had,
        # without performing any unit conversion.
        stripped = quant(qd, None)
        self.assertIsNone(stripped.units)
        self.assertEqual(stripped.m, qd.m)
        # .to() attaches units to a None-units quantity without conversion,
        # and can likewise strip units from a real quantity back to None.
        attached = q1.to('mm')
        self.assertEqual(attached.u, units.mm)
        self.assertEqual(attached.m, 10)
        back = attached.to(None)
        self.assertIsNone(back.units)
        self.assertEqual(back.m, 10)
        # .m_as() raises for a None-units quantity when a real unit is
        # requested (there's nothing to convert from), but succeeds and
        # returns the bare magnitude when None is requested.
        with self.assertRaises(ValueError):
            q1.m_as('mm')
        self.assertEqual(q1.m_as(None), 10)
        # A None-units quantity is never dimensionless (dimensionless is a
        # real unit in pint's sense; None means "no units" at all), and its
        # dimensionality and check() are correspondingly restricted.
        self.assertFalse(q1.dimensionless)
        self.assertTrue(qd.dimensionless)
        with self.assertRaises(TypeError):
            q1.dimensionality
        self.assertFalse(q1.check('[length]'))
        self.assertFalse(q1.check(None))
        # The numeric dunders bypass the usual dimensionless-only gate for
        # a None-units quantity, since there are no units to strip.
        self.assertEqual(int(quant(10)), 10)
        self.assertEqual(float(quant(10.5)), 10.5)
        self.assertEqual(complex(quant(10)), complex(10))
        # .ito() is the in-place version of .to(): it mutates the quantity
        # itself rather than returning a new one.
        m = quant(10)
        ret = m.ito('mm')
        self.assertIsNone(ret)
        self.assertEqual(m.u, units.mm)
        self.assertEqual(m.m, 10)
        m.ito(None)
        self.assertIsNone(m.units)
        self.assertEqual(m.m, 10)
    def test_quantity_none_arithmetic(self):
        from immlib import quant
        import numpy as np
        # Arithmetic between two None-units quantities operates directly on
        # the magnitudes and produces another None-units quantity.
        a = quant(np.array([1.0, 2.0, 3.0]))
        b = quant(np.array([4.0, 5.0, 6.0]))
        c = a + b
        self.assertIsNone(c.units)
        self.assertTrue(np.array_equal(c.m, [5.0, 7.0, 9.0]))
        c2 = a * 2
        self.assertIsNone(c2.units)
        self.assertTrue(np.array_equal(c2.m, [2.0, 4.0, 6.0]))
        # Arithmetic between a None-units quantity and a real-unit quantity
        # defers entirely to the real-unit quantity's own rules: addition
        # requires compatible units (raises here, since a is unitless and d
        # is not), while multiplication does not.
        d = quant(np.array([1.0, 2.0, 3.0]), 'mm')
        with self.assertRaises(Exception):
            a + d
        e = a * d
        self.assertEqual(e.u, d.u)
        self.assertTrue(np.array_equal(e.m, a.m * d.m))
        # The in-place operators (+=, *=) mutate the quantity itself
        # (identity is preserved) rather than returning a new object.
        f = quant(np.array([1.0, 2.0, 3.0]))
        f_id = id(f)
        f += quant(np.array([1.0, 1.0, 1.0]))
        self.assertEqual(id(f), f_id)
        self.assertIsNone(f.units)
        self.assertTrue(np.array_equal(f.m, [2.0, 3.0, 4.0]))
        f *= 2
        self.assertEqual(id(f), f_id)
        self.assertIsNone(f.units)
        self.assertTrue(np.array_equal(f.m, [4.0, 6.0, 8.0]))
    def test_quantity_none_pow_and_comparisons(self):
        from immlib import quant
        import numpy as np, pint
        # ** on a None-units quantity operates directly on the magnitude,
        # just like +, -, *, and / (test_quantity_none_arithmetic above).
        a = quant(np.array([2.0, 3.0]))
        r = a ** 2
        self.assertIsNone(r.units)
        self.assertTrue(np.array_equal(r.m, [4.0, 9.0]))
        # 0 and 1 exponents are handled without hitting pint's own
        # unit-exponentiation machinery (which would otherwise crash trying
        # to raise a units=None container to a power).
        r0 = a ** 0
        self.assertIsNone(r0.units)
        self.assertTrue(np.array_equal(r0.m, [1.0, 1.0]))
        # In-place **= mutates the quantity itself.
        b = quant(np.array([2.0, 3.0]))
        b_id = id(b)
        b **= 2
        self.assertEqual(id(b), b_id)
        self.assertIsNone(b.units)
        self.assertTrue(np.array_equal(b.m, [4.0, 9.0]))
        # A None-units exponent on a real-unit base behaves like a plain
        # number exponent (rather than raising, which is what happens if
        # pint sees a non-dimensionless quantity as an exponent).
        base = quant(np.array([2.0, 3.0]), 'm')
        exponent = quant(2, None)
        r = base ** exponent
        self.assertEqual(str(r.units), 'meter ** 2')
        self.assertTrue(np.array_equal(r.m, [4.0, 9.0]))
        # Reflected power (number ** None-units-quantity) also works.
        self.assertEqual(3.0 ** quant(2.0, None), 9.0)
        # Comparisons between two None-units quantities compare magnitudes
        # directly.
        c = quant(5.0)
        d = quant(5.0)
        e = quant(6.0)
        self.assertTrue(c == d)
        self.assertFalse(c == e)
        self.assertTrue(c != e)
        self.assertTrue(c < e)
        self.assertTrue(c <= d)
        self.assertTrue(e > c)
        self.assertTrue(d >= c)
        # A None-units quantity compares unequal (rather than raising) to a
        # quantity with real units.
        real = quant(5.0, 'm')
        self.assertFalse(c == real)
        self.assertTrue(c != real)
        # Comparisons on array-backed None-units quantities return element-
        # wise bool arrays, exactly as pint's own comparisons do for
        # real-unit quantities.
        av = quant(np.array([1.0, 2.0, 3.0]))
        bv = quant(np.array([1.0, 5.0, 3.0]))
        req = av == bv
        self.assertIsInstance(req, np.ndarray)
        self.assertTrue(np.array_equal(req, [True, False, True]))
        # A zero-magnitude None-units quantity does not crash comparison
        # (pint's own __eq__ has a special zero/NaN case that calls
        # .dimensionality, which is undefined--and raises--for None units).
        z1 = quant(0.0)
        z2 = quant(0.0)
        self.assertTrue(z1 == z2)
        # Real-unit comparisons are unaffected by any of the above (they
        # never hit the None-aware code paths).
        m1 = quant(5.0, 'm')
        m2 = quant(3.0, 'm')
        self.assertTrue(m1 > m2)
        with self.assertRaises(pint.DimensionalityError):
            m1 < quant(5.0, 's')
    def test_quantity_str_and_format(self):
        from immlib import quant
        import numpy as np
        # str()/format() on a None-units quantity must not crash (pint's
        # own formatter assumes a real UnitsContainer and, prior to this
        # fix, raised AttributeError for units=None); it simply shows the
        # bare magnitude, with no unit suffix.
        q = quant(np.array([1.0, 2.0, 3.0]))
        self.assertEqual(str(q), str(q.m))
        self.assertEqual(f"{q}", str(q.m))
        self.assertEqual(format(q), str(q.m))
        # repr() already worked before this fix and is unaffected.
        self.assertIn('None', repr(q))
        # A real-unit quantity's str()/format()/repr() are unaffected.
        real = quant(5.0, 'mm')
        self.assertIn('millimeter', str(real))
        self.assertIn('mm', f"{real:~}")
    def test_quantity_omitted_units(self):
        from immlib import Quantity, UnitRegistry, quant
        ureg = UnitRegistry()
        # Omitting the units argument entirely defers to pint's own
        # default: real dimensionless units, not immlib's None-units
        # sentinel.
        omitted = ureg.Quantity(5)
        self.assertIsInstance(omitted, Quantity)
        self.assertTrue(omitted.dimensionless)
        self.assertIsNotNone(omitted.units)
        self.assertEqual(omitted.m, 5)
        # Pint's own internal parsing (used by, e.g., is_compatible_with)
        # relies on this same omitted-argument default, and must therefore
        # still produce real dimensionless quantities.
        parsed = ureg.parse_expression('5 dimensionless')
        self.assertTrue(parsed.dimensionless)
        self.assertIsNotNone(parsed.units)
        self.assertEqual(parsed.m, 5)
        empty = ureg.parse_expression('')
        self.assertTrue(empty.dimensionless)
        self.assertIsNotNone(empty.units)
        # By contrast, an explicit units=None--how immlib's own quant() and
        # Quantity() calls always construct a "no units" quantity--is never
        # confused with the omitted-argument case above.
        none_q = quant(5, None, ureg=ureg)
        self.assertIsNone(none_q.units)
        self.assertFalse(none_q.dimensionless)
    def test_quantity_unitregistry(self):
        from immlib import Quantity, UnitRegistry, quant
        import pint
        from pint import Quantity as PintQuantity
        # immlib.UnitRegistry is a pint.UnitRegistry subclass whose own
        # Quantity type is immlib.Quantity; a plain pint.UnitRegistry is
        # never touched by immlib at all, even when passed explicitly via
        # `ureg`.
        ureg = UnitRegistry()
        plain = pint.UnitRegistry()
        self.assertTrue(issubclass(ureg.Quantity, Quantity))
        self.assertFalse(issubclass(plain.Quantity, Quantity))
        # quant() on a raw (non-quantity) magnitude uses whatever registry
        # it's given (or immlib.units by default): an immlib.UnitRegistry
        # yields an immlib.Quantity...
        q = quant(10, 'mm', ureg=ureg)
        self.assertIsInstance(q, Quantity)
        # ...but a plain pint.UnitRegistry yields an ordinary pint.Quantity,
        # not promoted to immlib.Quantity.
        p = quant(10, 'mm', ureg=plain)
        self.assertIsInstance(p, PintQuantity)
        self.assertNotIsInstance(p, Quantity)
        self.assertIs(p._REGISTRY, plain)
        # A unit-less (unit=None) quantity can only belong to an
        # immlib.UnitRegistry; requesting one against a plain
        # pint.UnitRegistry raises.
        with self.assertRaises(ValueError):
            quant(10, unit=None, ureg=plain)
        none_q = quant(10, unit=None, ureg=ureg)
        self.assertIsNone(none_q.units)
        # quant(mag) on a magnitude that is already a pint.Quantity returns
        # it completely unchanged, in its own registry--even a plain one;
        # a plain pint.Quantity is never promoted to immlib.Quantity by
        # quant().
        already = plain.Quantity(5, 'meter')
        self.assertIs(quant(already), already)
        self.assertNotIsInstance(quant(already), Quantity)
        # An immlib.UnitRegistry's own ordinary (non-immlib-aware) use, such
        # as unit multiplication, also produces immlib.Quantity objects,
        # which nonetheless remain fully functional pint.Quantity objects.
        viameter = 5 * ureg.meter
        self.assertIsInstance(viameter, Quantity)
        self.assertIsInstance(viameter, PintQuantity)
        self.assertEqual(str(viameter.units), 'meter')
    def test_quantity_matmul(self):
        from immlib import quant
        import torch, numpy as np
        # A tensor-backed matmul must go through torch.matmul directly
        # rather than the inherited np.matmul(self, other), which would
        # otherwise force the tensor magnitude to be silently converted to
        # NumPy (or crash outright for a tensor that requires grad); see
        # _immlib_matmul.
        ta = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        tb = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        qa = quant(ta, 'mm')
        qb = quant(tb, 'seconds')
        r = qa @ qb
        self.assertTrue(torch.is_tensor(r.m))
        self.assertTrue(torch.equal(r.m, torch.matmul(ta, tb)))
        self.assertEqual(r.u, qa.u * qb.u)
        # A grad-tracking tensor must not be silently detached/converted
        # (which is exactly what routing through np.matmul would risk).
        tg = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        qg = quant(tg, 'mm')
        rg = qg @ qb
        self.assertTrue(torch.is_tensor(rg.m))
        self.assertTrue(rg.m.requires_grad)
        # If one side is array-backed and the other tensor-backed, the
        # array side is promoted to a tensor (the same array/tensor
        # promotion rule used elsewhere in immlib; see promote()), rather
        # than the tensor being downgraded to an array.
        # aa is a plain NumPy array, whose default dtype (float64) differs
        # from tb's (float32, since tb was created from Python float
        # literals): the promoted tensor must match tb's dtype, not just
        # its device, or torch.matmul itself raises a dtype-mismatch
        # RuntimeError.
        aa = np.array([[1.0, 2.0], [3.0, 4.0]])
        qaa = quant(aa, 'mm')
        r2 = qaa @ qb
        self.assertTrue(torch.is_tensor(r2.m))
        self.assertEqual(r2.m.dtype, tb.dtype)
        self.assertTrue(
            torch.equal(
                r2.m, torch.matmul(torch.as_tensor(aa, dtype=tb.dtype), tb)))
        self.assertEqual(r2.u, qaa.u * qb.u)
        # This also works in the reflected direction (a plain tensor times
        # an array-backed quantity, or an array-backed quantity times a
        # plain tensor), via __rmatmul__.
        r3 = tb @ qaa
        self.assertTrue(torch.is_tensor(r3.m))
        self.assertEqual(r3.u, qaa.u)
        r4 = qaa @ tb
        self.assertTrue(torch.is_tensor(r4.m))
        self.assertEqual(r4.u, qaa.u)
        # None-units quantities combine with real-units quantities the
        # same way they do for ordinary multiplication: the real side's
        # units propagate.
        qn = quant(ta)
        self.assertIsNone(qn.units)
        rn = qn @ qb
        self.assertTrue(torch.is_tensor(rn.m))
        self.assertEqual(rn.u, qb.u)
        # Two None-units tensor-backed quantities combine into another
        # None-units quantity.
        qn2 = quant(tb)
        rn2 = qn @ qn2
        self.assertIsNone(rn2.units)
        self.assertTrue(torch.equal(rn2.m, torch.matmul(ta, tb)))
        # When neither side is tensor-backed, matmul is untouched (still
        # goes through pint's own array-backed handling).
        ab = np.array([[5.0, 6.0], [7.0, 8.0]])
        qab = quant(ab, 'seconds')
        r5 = qaa @ qab
        self.assertIsInstance(r5.m, np.ndarray)
        self.assertTrue(np.array_equal(r5.m, np.matmul(aa, ab)))
        self.assertEqual(r5.u, qaa.u * qab.u)
    def test_quantity_torch_function(self):
        from immlib import quant, is_quant
        import torch, pint
        # __torch_function__ lets a handful of common torch.* functions be
        # called directly on a Quantity (in addition to the Python
        # operators, which already work without it); anything not
        # explicitly handled returns NotImplemented, and torch raises its
        # own error rather than immlib guessing at unit semantics.
        ta = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        tb = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        qa = quant(ta, 'mm')
        qb = quant(tb, 'mm')
        # Operator-mirroring functions behave exactly like the operator.
        self.assertTrue(torch.equal(torch.add(qa, qb).m, (qa + qb).m))
        self.assertEqual(torch.add(qa, qb).u, (qa + qb).u)
        self.assertTrue(torch.equal(torch.sub(qa, qb).m, (qa - qb).m))
        self.assertTrue(torch.equal(torch.mul(qa, 2).m, (qa * 2).m))
        # torch.matmul(q1, q2), called as a function rather than via @,
        # goes through the same tensor-safe path as q1 @ q2.
        self.assertTrue(torch.equal(torch.matmul(qa, qb).m, (qa @ qb).m))
        self.assertEqual(torch.matmul(qa, qb).u, (qa @ qb).u)
        # A keyword argument we don't support (e.g. alpha=) is refused
        # rather than silently ignored.
        with self.assertRaises(TypeError):
            torch.add(qa, qb, alpha=2)
        # A raw (non-Quantity) tensor combined with a Quantity, with the
        # tensor on the LEFT, must resolve via the Quantity's reflected
        # dunder (__radd__, __rmatmul__, etc.) rather than by evaluating
        # the operator expression directly--this is a real, previously
        # observed bug: PyTorch's own Tensor arithmetic dunders are
        # themselves implemented via the same overridable-function
        # mechanism this dispatch hooks into, so evaluating
        # `raw_tensor + quantity` as an expression re-enters
        # __torch_function__ with the same arguments and recurses forever
        # (RecursionError). quant(ta) below has no units (None), so the
        # combination with a raw tensor is well-defined and should not
        # raise, unlike combining a raw tensor with a real-unit quantity.
        qn = quant(ta)
        self.assertIsNone(qn.units)
        rr = torch.add(tb, qn)
        self.assertTrue(is_quant(rr))
        self.assertIsNone(rr.units)
        self.assertTrue(torch.equal(rr.m, tb + ta))
        rr2 = tb @ qn
        self.assertTrue(is_quant(rr2))
        self.assertIsNone(rr2.units)
        self.assertTrue(torch.equal(rr2.m, torch.matmul(tb, ta)))
        # Unit-preserving shape/reduction functions apply to the
        # magnitude and keep the original units.
        r = torch.sum(qa)
        self.assertTrue(is_quant(r))
        self.assertEqual(r.u, qa.u)
        self.assertEqual(r.m, torch.sum(ta))
        r = torch.reshape(qa, (4,))
        self.assertEqual(r.u, qa.u)
        self.assertTrue(torch.equal(r.m, torch.reshape(ta, (4,))))
        r = torch.transpose(qa, 0, 1)
        self.assertEqual(r.u, qa.u)
        self.assertTrue(torch.equal(r.m, torch.transpose(ta, 0, 1)))
        # torch.cat/torch.stack require compatible units across every
        # quantity in the sequence, and convert to the first quantity's
        # unit when they match but aren't identical.
        r = torch.cat([qa, qb])
        self.assertEqual(r.u, qa.u)
        self.assertTrue(torch.equal(r.m, torch.cat([ta, tb])))
        r = torch.stack([qa, qb])
        self.assertEqual(r.u, qa.u)
        self.assertTrue(torch.equal(r.m, torch.stack([ta, tb])))
        qcm = quant(tb, 'cm')
        r = torch.cat([qa, qcm])
        self.assertEqual(r.u, qa.u)
        # tb is in cm and qa/r are in mm, so the converted values are 10x.
        self.assertTrue(
            torch.allclose(r.m, torch.cat([ta, tb * 10.0])))
        qs = quant(tb, 'seconds')
        with self.assertRaises(pint.DimensionalityError):
            torch.cat([qa, qs])
        # A function outside the supported minimal subset is not handled;
        # torch raises its own error rather than immlib guessing.
        with self.assertRaises(TypeError):
            torch.linalg.det(qa)

    def test_quantity_numpy_dispatch(self):
        from immlib import quant, is_quant
        import immlib.math as im
        import numpy as np
        # __array_ufunc__/__array_function__ delegate a fixed set of
        # common NumPy functions to immlib.math (see _numpy_math_dispatch
        # in immlib.util._quantity), so that calling e.g. np.exp(q)
        # directly on a Quantity behaves the same as im.exp(q) -- in
        # particular, this is what makes a None-units quantity usable
        # with a plain `np.func(q)` call, which pint's own dispatch does
        # not handle correctly for units=None.
        an = quant(np.array([0.0, 1.0, 2.0]), None)
        r = np.exp(an)
        self.assertTrue(is_quant(r))
        self.assertIsNone(r.units)
        self.assertTrue(np.allclose(r.m, im.exp(an).m))

        # Real-units quantities: sqrt raises the magnitude's units to the
        # 1/2 power, exactly as im.sqrt does.
        asq = quant(np.array([4.0, 9.0, 16.0]), 'm**2')
        r = np.sqrt(asq)
        self.assertTrue(is_quant(r))
        self.assertEqual(str(r.units), 'meter')
        self.assertTrue(np.allclose(r.m, [2.0, 3.0, 4.0]))

        # Binary arithmetic ufuncs (np.add, ...) go through the same
        # unit-aligning path as the + operator/im.add--specifically, by
        # calling the Quantity operand's own dunder directly
        # (_NUMPY_OPERATOR_METHODS), not by re-evaluating an
        # immlib.math expression from inside __array_ufunc__ itself.
        a = quant(np.array([1.0, 2.0, 3.0]), 'm')
        b = quant(np.array([100.0, 200.0, 300.0]), 'cm')
        r = np.add(a, b)
        self.assertTrue(is_quant(r))
        self.assertEqual(str(r.units), 'meter')
        self.assertTrue(np.allclose(r.m, [2.0, 4.0, 6.0]))

        # np.absolute and its alias np.abs both resolve to the same
        # dunder call (_NUMPY_UNARY_OPERATOR_METHODS maps 'absolute' to
        # Quantity.__abs__).
        aneg = quant(np.array([-1.0, 2.0, -3.0]), 'm')
        self.assertTrue(np.allclose(np.abs(aneg).m, [1.0, 2.0, 3.0]))
        self.assertTrue(np.allclose(np.absolute(aneg).m, [1.0, 2.0, 3.0]))

        # Regression: an array-backed None-units quantity combined with
        # an array-backed real-units quantity via a binary arithmetic
        # ufunc/operator used to recurse infinitely (RecursionError).
        # The None-units side unwraps to a bare ndarray inside
        # _binop_none, and combining a bare ndarray with a Quantity that
        # implements __array_ufunc__ is itself resolved via NumPy's
        # ufunc-override protocol; delegating that call to an
        # immlib.math wrapper that re-evaluates `quant(a) * quant(b)`
        # (rather than calling the real-units operand's own,
        # non-recursing dunder directly) reproduced the exact same call
        # forever. This exercises both the operator form and the
        # explicit np.* function form, for both the commutative
        # (multiply) and non-commutative/unit-incompatible (add) cases.
        an = quant(np.array([1.0, 2.0, 3.0]))
        self.assertIsNone(an.units)
        e = an * a
        self.assertEqual(str(e.units), 'meter')
        self.assertTrue(np.allclose(e.m, an.m * a.m))
        e2 = np.multiply(an, a)
        self.assertEqual(str(e2.units), 'meter')
        self.assertTrue(np.allclose(e2.m, an.m * a.m))
        with self.assertRaises(Exception):
            an + a
        with self.assertRaises(Exception):
            np.add(an, a)
        # Two None-units quantities added via np.add still works (this
        # is the case the delegation exists to support in the first
        # place--Pint's own default __array_ufunc__ crashes outright for
        # a units=None operand).
        an2 = quant(np.array([4.0, 5.0, 6.0]))
        r = np.add(an, an2)
        self.assertIsNone(r.units)
        self.assertTrue(np.allclose(r.m, [5.0, 7.0, 9.0]))
        # np.equal is consistent with the == operator for a None-units
        # vs. real-units comparison (Pint's own default array_ufunc for
        # 'equal' compares bare magnitudes without regard to units at
        # all, which would otherwise silently disagree with ==).
        self.assertTrue(np.array_equal(
            np.equal(an, a), (an == a)))

        # Regression: np.matmul/`@` on two (real-units, array-backed)
        # quantities used to recurse infinitely for the same reason--
        # immlib.math.matmul is `quant(a) @ quant(b)`, and matmul was
        # previously included in the numpy delegate table, so
        # __array_ufunc__ delegated straight back into the same
        # np.matmul(self, other) call Pint's own Quantity.__matmul__
        # makes for the array-backed case. matmul now needs no
        # None-units-specific delegation at all: Pint's own default
        # __array_ufunc__ handles it correctly, including a None-units
        # operand (unlike add/multiply/etc., matmul never validates
        # dimensionality against a specific unit).
        am = quant(np.array([[1.0, 2.0], [3.0, 4.0]]), 'mm')
        bm = quant(np.array([[5.0, 6.0], [7.0, 8.0]]), 's')
        rm = am @ bm
        self.assertTrue(is_quant(rm))
        self.assertEqual(str(rm.units), 'millimeter * second')
        self.assertTrue(np.array_equal(rm.m, np.matmul(am.m, bm.m)))
        rm2 = np.matmul(am, bm)
        self.assertTrue(is_quant(rm2))
        self.assertTrue(np.array_equal(rm2.m, rm.m))

        # __array_function__ path: axis-based reductions and sequence
        # combination (which also converts compatible-but-different units
        # to the first sequence element's unit).
        a2 = quant(np.array([[1.0, 2.0], [3.0, 4.0]]), 's')
        r = np.sum(a2, axis=0)
        self.assertTrue(is_quant(r))
        self.assertEqual(str(r.units), 'second')
        self.assertTrue(np.allclose(r.m, [4.0, 6.0]))
        r = np.stack([a, quant(np.array([400.0, 500.0, 600.0]), 'cm')])
        self.assertTrue(is_quant(r))
        self.assertEqual(str(r.units), 'meter')
        self.assertTrue(np.allclose(r.m, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

        # Comparisons return a raw boolean ndarray, never a Quantity --
        # this mirrors immlib.math's equal/less/etc. (see the explicit
        # design decision that booleans are never wrapped).
        req = np.equal(a, quant(np.array([100.0, 999.0, 300.0]), 'cm'))
        self.assertFalse(is_quant(req))
        self.assertIsInstance(req, np.ndarray)
        self.assertTrue(np.array_equal(req, [True, False, True]))

        # A function that isn't part of the delegated subset (and that
        # pint itself doesn't implement for Quantity either) still fails
        # the way it always has -- immlib's dispatch override doesn't
        # swallow the failure or fall back silently to a wrong answer.
        with self.assertRaises(TypeError):
            np.fft.fft(a)

        # `out=` as a plain (non-Quantity) ndarray isn't part of the
        # delegated signature (immlib.math functions don't accept `out`),
        # so the signature-bind safety check in _numpy_math_dispatch
        # correctly falls back to pint's own pre-existing handling rather
        # than raising a spurious TypeError or silently ignoring `out`.
        out = np.empty(3)
        np.add(a, b, out=out)
        self.assertTrue(np.allclose(out, [2.0, 4.0, 6.0]))

    def test_quantity_torch_function_math_delegates(self):
        from immlib import quant, is_quant
        import torch
        # Elementary functions (_TORCH_MATH_DELEGATE_FUNCS) delegate to
        # the matching immlib.math function, which computes directly on
        # the tensor (no NumPy round-trip, so gradients keep flowing);
        # sqrt applies the units**0.5 rule.
        a2 = quant(torch.tensor([[1.0, 4.0], [9.0, 16.0]]), 'm**2')
        r = torch.sqrt(a2)
        self.assertTrue(is_quant(r))
        self.assertEqual(str(r.units), 'meter')
        self.assertTrue(
            torch.allclose(r.m, torch.tensor([[1.0, 2.0], [3.0, 4.0]])))

        # exp/log/etc. require a unit-less (units=None) quantity, exactly
        # like im.exp -- the delegate enforces this, it doesn't bypass it.
        ta = torch.tensor([[1.0, 4.0], [9.0, 16.0]])
        qn = quant(ta)
        self.assertIsNone(qn.units)
        r = torch.exp(qn)
        self.assertTrue(is_quant(r))
        self.assertIsNone(r.units)
        self.assertTrue(torch.allclose(r.m, torch.exp(ta)))
        with self.assertRaises(TypeError):
            torch.exp(a2)

        # atan2 is the one 2-argument elementary function, and its torch
        # name ('atan2') differs from immlib.math's NumPy-style name
        # ('arctan2').
        qy = quant(torch.tensor([1.0, 0.0, -1.0]))
        qx = quant(torch.tensor([1.0, 1.0, 1.0]))
        r = torch.atan2(qy, qx)
        self.assertTrue(is_quant(r))
        self.assertIsNone(r.units)
        self.assertTrue(torch.allclose(r.m, torch.atan2(qy.m, qx.m)))

        # Reduction functions (_TORCH_MATH_REDUCTION_FUNCS) translate
        # torch's own kwarg names (dim/keepdim/correction) to
        # immlib.math's NumPy-style ones (axis/keepdims/ddof) via
        # _TORCH_KWARG_TRANSLATION.
        am = quant(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), 'm')
        r = torch.var(am, dim=0, correction=0)
        self.assertTrue(is_quant(r))
        self.assertEqual(str(r.units), 'meter ** 2')
        self.assertTrue(
            torch.allclose(r.m, torch.var(am.m, dim=0, correction=0)))
        r = torch.prod(quant(torch.tensor([2.0, 3.0, 4.0]), 'm'))
        self.assertTrue(is_quant(r))
        self.assertEqual(str(r.units), 'meter ** 3')

        # Bool reductions (any/all) return the raw tensor, never a
        # Quantity -- the same "booleans aren't wrapped" rule as the
        # NumPy and Python-operator paths.
        qb = quant(torch.tensor([True, False, True]))
        r = torch.any(qb)
        self.assertFalse(is_quant(r))
        self.assertTrue(torch.is_tensor(r))
        self.assertEqual(bool(r), True)

        # amin/amax/floor/ceil/round were added to the unit-preserving
        # table in Phase 5 (previously only sum/mean/reshape/etc. were
        # covered).
        r = torch.amin(am)
        self.assertTrue(is_quant(r))
        self.assertEqual(r.u, am.u)
        r = torch.floor(quant(torch.tensor([1.2, 2.7]), 's'))
        self.assertTrue(is_quant(r))
        self.assertEqual(str(r.units), 'second')
        self.assertTrue(torch.allclose(r.m, torch.tensor([1.0, 2.0])))

        # torch.min/torch.max are deliberately NOT handled: they return
        # PyTorch's native (values, indices) tuple and take a `dim`
        # argument, which immlib.math's NumPy-style values-only min/max
        # cannot reproduce without silently changing torch's own
        # documented contract for a Quantity input (the same "materially
        # different semantics, exclude" reasoning as the dot exclusion,
        # spec 9.3); calling them on a Quantity therefore falls through
        # to torch's own error rather than returning a wrong result.
        with self.assertRaises(TypeError):
            torch.min(am)
        with self.assertRaises(TypeError):
            torch.max(am)

        # Gradient flow is preserved through the delegate path (a hard
        # requirement per spec 9.3.4): sqrt above goes through
        # immlib.math.sqrt, which computes directly on the tensor rather
        # than detaching to NumPy.
        tg = torch.tensor([4.0, 9.0], requires_grad=True)
        qg = quant(tg, 'm**2')
        rg = torch.sqrt(qg)
        rg.m.sum().backward()
        self.assertIsNotNone(tg.grad)

    def test_quantity_dunders_and_inplace(self):
        from immlib import quant
        import numpy as np
        # __int__/__float__/__complex__ on a None-units quantity convert
        # the bare magnitude directly, bypassing pint's own conversion
        # (which requires the quantity to be dimensionless, not merely
        # units=None, and would otherwise raise).
        n = quant(3.7)
        self.assertEqual(int(n), 3)
        self.assertEqual(float(n), 3.7)
        self.assertEqual(complex(n), complex(3.7))
        # The same dunders on a real (dimensionless) unit quantity are
        # untouched--still go through pint's own conversion.
        r = quant(3.7, 'dimensionless')
        self.assertEqual(int(r), 3)
        self.assertEqual(float(r), 3.7)
        # check() reports False for a None-units quantity rather than
        # raising (pint's own check() relies on .dimensionality, which is
        # undefined--and raises--for units=None).
        self.assertFalse(n.check('[length]'))
        m = quant(3.0, 'm')
        self.assertTrue(m.check('[length]'))
        self.assertFalse(m.check('[time]'))
        # ito() (in-place to()) on a None-units quantity mutates the
        # quantity itself and returns None, exactly like pint's own
        # ito() for real-unit quantities.
        q = quant(np.array([1.0, 2.0, 3.0]))
        q_id = id(q)
        result = q.ito('mm')
        self.assertIsNone(result)
        self.assertEqual(id(q), q_id)
        self.assertEqual(str(q.units), 'millimeter')
        self.assertTrue(np.array_equal(q.m, [1.0, 2.0, 3.0]))
        # ito(None) strips a real-unit quantity's units in place.
        q2 = quant(np.array([1.0, 2.0, 3.0]), 'm')
        q2.ito(None)
        self.assertIsNone(q2.units)
        self.assertTrue(np.array_equal(q2.m, [1.0, 2.0, 3.0]))
        # -= and /= (the other half of the _iadd_sub/_imul_div choke
        # points, alongside += and *= already covered in
        # test_quantity_none_arithmetic) also mutate a None-units
        # quantity in place rather than returning a new object.
        f = quant(np.array([5.0, 6.0, 7.0]))
        f_id = id(f)
        f -= quant(np.array([1.0, 1.0, 1.0]))
        self.assertEqual(id(f), f_id)
        self.assertIsNone(f.units)
        self.assertTrue(np.array_equal(f.m, [4.0, 5.0, 6.0]))
        f /= 2
        self.assertEqual(id(f), f_id)
        self.assertTrue(np.array_equal(f.m, [2.0, 2.5, 3.0]))

    def test_quantity_plain_pint_interaction(self):
        from immlib import quant, units, Quantity
        import pint
        # quant() accepts a plain pint.Quantity--one belonging to an
        # ordinary pint.UnitRegistry rather than an immlib.UnitRegistry--
        # in addition to an immlib.Quantity; this exercises "other kinds
        # of quantities" interacting with immlib's own Quantity type.
        plain_ureg = pint.UnitRegistry()
        plain_q = plain_ureg.Quantity(5.0, 'meter')
        self.assertNotIsInstance(plain_q, Quantity)
        # By default (unit=Ellipsis), a plain pint.Quantity is returned
        # exactly as given, in its own registry--even when an explicit,
        # different `ureg` is also passed--since Ellipsis short-circuits
        # before quant() ever considers re-homing it; a plain
        # pint.Quantity is never silently promoted to an immlib.Quantity
        # just because a different registry was mentioned.
        self.assertIs(quant(plain_q), plain_q)
        self.assertIs(quant(plain_q, ureg=units), plain_q)
        # Converting to an explicit unit, with no explicit `ureg`, also
        # keeps the result in the plain quantity's own (plain) registry.
        r2 = quant(plain_q, 'mm')
        self.assertNotIsInstance(r2, Quantity)
        self.assertEqual(r2.m, 5000.0)
        # A plain pint.UnitRegistry cannot represent a unit-less
        # (units=None) quantity, so requesting unit=None on a
        # plain-registry quantity raises--even though quant() is happy
        # to do the analogous thing for an immlib.UnitRegistry quantity.
        with self.assertRaises(ValueError):
            quant(plain_q, None)
        # Requesting an explicit real-unit conversion *together with* an
        # explicit, different `ureg` is the one case where a plain
        # pint.Quantity does get promoted: quant() converts within the
        # quantity's own registry first, then re-homes the result into
        # the requested registry because the two registries differ.
        r3 = quant(plain_q, 'mm', ureg=units)
        self.assertIsInstance(r3, Quantity)
        self.assertEqual(r3.m, 5000.0)
        self.assertEqual(str(r3.units), 'millimeter')
        # unit=None with an explicit immlib `ureg` also succeeds and
        # promotes, since the immlib.UnitRegistry can represent
        # unit-less quantities; the original (plain) units are simply
        # discarded, as with any other quant(..., None) call.
        r4 = quant(plain_q, None, ureg=units)
        self.assertIsInstance(r4, Quantity)
        self.assertIsNone(r4.units)
        self.assertEqual(r4.m, 5.0)

    def test_quantity_cross_registry_arithmetic(self):
        from immlib import quant
        import pint
        # Arithmetic between an immlib.Quantity and a plain pint.Quantity
        # from an *independent* pint.UnitRegistry instance is still
        # refused by pint's own cross-registry safety check--immlib's
        # None-aware overrides of _add_sub/_mul_div/etc. only special-
        # case units=None, and otherwise defer to pint's own
        # implementation (via super()), which continues to guard against
        # combining quantities from different registries.
        plain_ureg = pint.UnitRegistry()
        plain_q = plain_ureg.Quantity(3.0, 'meter')
        iq = quant(5.0, 'meter')
        with self.assertRaises(ValueError):
            iq + plain_q
        # A None-units immlib.Quantity combined with a real-unit
        # quantity from any registry (same or different) unwraps to the
        # bare magnitude for the operation (per _binop_none), so it
        # never reaches the cross-registry check at all; the bare
        # magnitude is instead treated as dimensionless and compared
        # against plain_q's real units, which raises its own
        # DimensionalityError--the same thing that happens for a
        # None-units quantity plus any other incompatible real-unit
        # quantity (see test_quantity_none_arithmetic).
        none_q = quant(5.0)
        with self.assertRaises(pint.DimensionalityError):
            none_q + plain_q
