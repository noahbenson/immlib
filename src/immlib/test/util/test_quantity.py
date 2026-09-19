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

        # Reduction functions (_TORCH_MATH_REDUCTION_FUNCS) pass torch's own
        # arguments straight through, since immlib.math takes PyTorch's
        # argument names and defaults.
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

        # torch.min/torch.max are handled now that immlib.math.min/max
        # return PyTorch's own (values, indices) pair: torch's documented
        # contract holds for a Quantity, with the values carrying units.
        r = torch.min(am)
        self.assertTrue(is_quant(r))
        self.assertEqual(float(r.m), 1.0)
        self.assertEqual(r.u, am.u)
        r = torch.max(am, dim=0)
        self.assertTrue(is_quant(r.values))
        self.assertEqual(r.values.u, am.u)
        self.assertTrue(torch.allclose(r.values.m, torch.tensor([3.0, 4.0])))
        self.assertTrue(torch.equal(r.indices, torch.tensor([1, 1])))

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
        # repr() of a None-units quantity must show the bare None literal
        # for its units, not the string "'None'" -- Pint's own __repr__
        # unconditionally quotes the units part, which, given a real
        # Python None, renders indistinguishably from a unit literally
        # named "None".
        n0 = quant(np.array([2.718281828, 7.389056099, 20.08553692]))
        r = repr(n0)
        # (Pint < 0.26 formats reprs as <Quantity(..., 'units')>; Pint 0.26
        # formats them as Quantity(..., "units").)
        self.assertTrue(r.endswith('None)>') or r.endswith('None)'))
        self.assertNotIn("'None'", r)
        self.assertNotIn('"None"', r)
        # A real-units quantity's repr is untouched.
        real = quant(5.0, 'm')
        self.assertTrue("'meter'" in repr(real) or '"meter"' in repr(real))
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
        from immlib import quant, ilquant, units, Quantity
        from immlib.util import unitregistry
        import pint
        # quant() accepts a plain pint.Quantity--one belonging to an
        # ordinary pint.UnitRegistry rather than an immlib.UnitRegistry--
        # in addition to an immlib.Quantity; this exercises "other kinds
        # of quantities" interacting with immlib's own Quantity type.
        plain_ureg = pint.UnitRegistry()
        plain_q = plain_ureg.Quantity(5.0, 'meter')
        self.assertNotIsInstance(plain_q, Quantity)
        # With no `ureg` (the default, None, which requests no particular
        # registry), a plain pint.Quantity is returned exactly as given, in
        # its own registry: quant() never silently promotes one.
        self.assertIs(quant(plain_q), plain_q)
        self.assertIs(quant(plain_q, ureg=None), plain_q)
        # Naming a registry requests one, however, so the quantity is
        # re-homed into it and thus promoted.
        r1 = quant(plain_q, ureg=units)
        self.assertIsInstance(r1, Quantity)
        self.assertEqual(r1.m, 5.0)
        self.assertEqual(str(r1.units), 'meter')
        self.assertIs(unitregistry(r1), units)
        # ureg=Ellipsis names immlib.units, and is what ilquant defaults to.
        self.assertIsInstance(quant(plain_q, ureg=Ellipsis), Quantity)
        r1b = ilquant(plain_q)
        self.assertIsInstance(r1b, Quantity)
        self.assertEqual(r1b.m, 5.0)
        self.assertEqual(str(r1b.units), 'meter')
        # ilquant(q, ureg=None) is quant(q).
        self.assertIs(ilquant(plain_q, ureg=None), plain_q)
        # An immlib.Quantity is already where ilquant wants it.
        iq = quant(5.0, 'meter')
        self.assertIs(ilquant(iq), iq)
        self.assertIs(quant(iq), iq)
        # For a non-quantity, the two functions agree.
        for f in (quant, ilquant):
            q0 = f(5.0, 'meter')
            self.assertIsInstance(q0, Quantity)
            self.assertEqual(q0.m, 5.0)
        # A unit-less quantity cannot be moved into a plain pint registry,
        # which would silently turn "no units" into dimensionless.
        with self.assertRaises(ValueError):
            quant(quant(5.0), ureg=plain_ureg)
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
        # An explicit real-unit conversion together with an explicit,
        # different `ureg` promotes as well: quant() converts within the
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


    def test_quantity_tensor_eq_ne(self):
        from immlib import quant
        import torch, pint
        # Regression test: Pint's own __eq__/__ne__ contain a "compare to
        # zero" shortcut that reduces an elementwise comparison to a
        # single Python bool via `.all()`, assuming the result is a NumPy
        # array (or another type Pint's `is_duck_array_type` recognizes).
        # It does not recognize a PyTorch tensor, so the un-reduced,
        # multi-element boolean tensor ends up in a plain Python `and`,
        # whose multi-element `Tensor.__bool__` raises RuntimeError. This
        # previously crashed for *any* comparison between two ordinary,
        # real-units, multi-element tensor-backed quantities, regardless
        # of whether either magnitude actually contained a zero.
        qa = quant(torch.tensor([1.0, 2.0, 3.0]), 'm')
        qb = quant(torch.tensor([1.0, 2.0, 3.0]), 'm')
        r = qa == qb
        self.assertTrue(torch.is_tensor(r))
        self.assertTrue(bool(r.all()))
        qc = quant(torch.tensor([1.0, 2.0, 30.0]), 'm')
        r = qa == qc
        self.assertTrue(torch.is_tensor(r))
        self.assertTrue(torch.equal(r, torch.tensor([True, True, False])))
        r = qa != qc
        self.assertTrue(torch.is_tensor(r))
        self.assertTrue(torch.equal(r, torch.tensor([False, False, True])))
        # Alike-but-different units convert before comparing, exactly as
        # for the real-units arithmetic operators.
        qcm = quant(torch.tensor([100.0, 200.0, 300.0]), 'cm')
        r = qa == qcm
        self.assertTrue(torch.is_tensor(r))
        self.assertTrue(bool(r.all()))
        # Dimensionally incompatible real units compare unequal elementwise,
        # matching Pint's behavior for NumPy magnitudes.
        qs = quant(torch.tensor([1.0, 2.0, 3.0]), 's')
        r = qa == qs
        self.assertTrue(torch.is_tensor(r))
        self.assertTrue(torch.equal(r, torch.tensor([False, False, False])))
        r = qa != qs
        self.assertTrue(torch.equal(r, torch.tensor([True, True, True])))
        # A tensor magnitude mixed with a NumPy-magnitude quantity of the
        # same real units also works (no promotion crash).
        import numpy as np
        qnp = quant(np.array([1.0, 2.0, 3.0]), 'm')
        r = qa == qnp
        self.assertTrue(bool(torch.all(torch.as_tensor(r))))
    def test_none_units_binop_rule(self):
        """Tests that ``op(quant(x, None), quant(y, u))`` is equivalent to
        ``(x, y) = promote(x, y); op(x, quant(y, u))``."""
        from immlib import quant, promote
        import numpy as np, torch, pint, operator
        xs = [np.array([0.5, 2.0, 3.0]), torch.tensor([0.5, 2.0, 3.0])]
        ys = [np.array([1.0, 2.0, 4.0]), torch.tensor([1.0, 2.0, 4.0])]
        units = ['m', 'dimensionless', 'm/km']
        ops = [operator.add, operator.sub, operator.mul, operator.truediv,
               operator.eq, operator.ne, operator.lt, operator.le,
               operator.gt, operator.ge]
        def run(f):
            try:
                return ('ok', f())
            except pint.DimensionalityError:
                return ('dimerr', None)
            except ValueError:
                # Ordering a bare value against a quantity with real
                # dimensions fails either way round, but Pint spells the
                # failure differently in the two orders: a
                # DimensionalityError for `bare < quantity` and a
                # ValueError ("Cannot compare PlainQuantity and ...") for
                # `quantity < bare`. immlib raises the DimensionalityError
                # both ways (asserted below), so the two spellings count as
                # the same outcome here.
                return ('dimerr', None)
        def same(r1, r2):
            self.assertEqual(r1[0], r2[0])
            if r1[0] != 'ok':
                return
            (a, b) = (r1[1], r2[1])
            self.assertEqual(isinstance(a, pint.Quantity),
                             isinstance(b, pint.Quantity))
            if isinstance(a, pint.Quantity):
                self.assertEqual(a.units, b.units)
                (a, b) = (a.m, b.m)
            self.assertEqual(torch.is_tensor(a), torch.is_tensor(b))
            a = np.asarray(a.detach().cpu() if torch.is_tensor(a) else a)
            b = np.asarray(b.detach().cpu() if torch.is_tensor(b) else b)
            self.assertTrue(np.allclose(a.astype(float), b.astype(float)))
        for x in xs:
            for y in ys:
                for u in units:
                    for op in ops:
                        with self.subTest(x=type(x), y=type(y), u=u, op=op):
                            (px, py) = promote(x, y)
                            expected = run(lambda: op(px, quant(py, u)))
                            got = run(lambda: op(quant(x), quant(y, u)))
                            same(got, expected)
                            # And the reflected order.
                            expected = run(lambda: op(quant(py, u), px))
                            got = run(lambda: op(quant(y, u), quant(x)))
                            same(got, expected)
        # Ordering a unit-less quantity against a dimensional one raises a
        # DimensionalityError whichever way round it is written, rather than
        # Pint's ValueError in one order and DimensionalityError in the
        # other.
        for x in xs:
            for y in ys:
                for op in (operator.lt, operator.le, operator.gt,
                           operator.ge):
                    with self.subTest(x=type(x), y=type(y), op=op):
                        with self.assertRaises(pint.DimensionalityError):
                            op(quant(x), quant(y, 'm'))
                        with self.assertRaises(pint.DimensionalityError):
                            op(quant(y, 'm'), quant(x))
        # Tensor results stay on the tensor side and are Quantities.
        r = quant(np.array([1.0, 2.0])) * quant(torch.tensor([1.0, 2.0]), 'm')
        self.assertTrue(torch.is_tensor(r.m))
        self.assertEqual(str(r.units), 'meter')
        # A unit-less value is unequal to a quantity with real dimensions.
        r = quant(torch.tensor([1.0, 2.0])) == quant(torch.tensor([1.0, 2.0]), 'm')
        self.assertTrue(torch.equal(r, torch.tensor([False, False])))
        r = quant(np.array([1.0, 2.0])) == quant(np.array([1.0, 2.0]), 'm')
        self.assertTrue(np.array_equal(r, [False, False]))
        # Addition of a unit-less value and a dimensioned one is an error.
        with self.assertRaises(pint.DimensionalityError):
            quant(np.array([1.0])) + quant(np.array([1.0]), 'm')
        # Both unit-less: the result is unit-less and tensor-backed if either
        # operand is a tensor.
        r = quant(np.array([1.0, 2.0])) + quant(torch.tensor([1.0, 2.0]))
        self.assertIsNone(r.units)
        self.assertTrue(torch.is_tensor(r.m))
        r = 2.0 ** quant(torch.tensor([1.0, 2.0]))
        self.assertIsNone(r.units)
        self.assertTrue(torch.allclose(r.m, torch.tensor([2.0, 4.0])))
    def test_quantity_pickle_hash(self):
        import pickle
        import immlib
        from immlib import quant, Quantity
        import numpy as np, torch
        for q in [quant(np.array([1.0, 2.0])),
                  quant(np.array([1.0, 2.0]), 'mm'),
                  quant(torch.tensor([1.0, 2.0])),
                  quant(torch.tensor([1.0, 2.0]), 'mm/s'),
                  quant(5.0, 'm')]:
            with self.subTest(q=q):
                r = pickle.loads(pickle.dumps(q))
                self.assertIsInstance(r, Quantity)
                self.assertIs(r._REGISTRY, immlib.units)
                self.assertEqual(r.units, q.units)
                self.assertEqual(type(r.m), type(q.m))
                if torch.is_tensor(q.m):
                    self.assertTrue(torch.equal(r.m, q.m))
                else:
                    self.assertTrue(np.array_equal(r.m, q.m))
        # Units of None hash like their magnitude.
        self.assertEqual(hash(quant(5)), hash(5))
        self.assertEqual(hash(quant(np.array(2.5))), hash(2.5))
        self.assertEqual(hash(quant(1.0, 'dimensionless')), hash(1.0))
        self.assertEqual(hash(quant(1.0, 'm')), hash(quant(100.0, 'cm')))
        self.assertEqual(hash(quant(np.array(1.0), 'm')),
                         hash(quant(1.0, 'm')))
        with self.assertRaises(TypeError):
            hash(quant(np.array([1.0, 2.0])))
    def test_alike_units_none(self):
        from immlib import quant, alike_units
        self.assertFalse(alike_units(quant(1.0), quant(1.0, 'm')))
        self.assertFalse(alike_units(quant(1.0, 'm'), quant(1.0)))
        self.assertTrue(alike_units(quant(1.0), quant(1.0, 'dimensionless')))
        self.assertTrue(alike_units(quant(1.0), quant(1.0)))
        self.assertTrue(alike_units(quant(1.0), 5))
        self.assertFalse(alike_units(quant(1.0), 'm'))
    def test_real_units_bare_tensor(self):
        """Tests arithmetic and comparisons between a quantity with real units
        and a bare tensor or a quantity of the other backend."""
        from immlib import quant
        import numpy as np, torch, pint
        t = torch.tensor([1.0, 2.0, 3.0])
        q = quant(torch.tensor([1.0, 2.0, 3.0]), 'm')
        # Adding zeros is allowed for any units (as in Pint).
        r = q + torch.zeros(3)
        self.assertEqual(str(r.units), 'meter')
        self.assertTrue(torch.equal(r.m, q.m))
        with self.assertRaises(pint.DimensionalityError):
            q + t
        with self.assertRaises(pint.DimensionalityError):
            t - q
        r = quant(t, 'dimensionless') + t
        self.assertTrue(torch.equal(r.m, 2 * t))
        r = quant(t, 'm/km') + t
        self.assertEqual(r.units, quant(1, 'm/km').units)
        self.assertTrue(torch.allclose(r.m, 1001 * t))
        # Ordering a dimensional quantity against a bare value is a
        # dimensionality failure (Pint spells this one a ValueError).
        with self.assertRaises(pint.DimensionalityError):
            q < t
        self.assertTrue(bool((q > torch.zeros(3)).all()))
        # NumPy and tensor magnitudes are promoted to tensors.
        qn = quant(np.array([1.0, 2.0, 3.0]), 'm')
        for r in (qn * t, t * qn, qn + q, q - qn, qn / q):
            self.assertTrue(torch.is_tensor(r.m))
        self.assertEqual(str((qn * q).units), 'meter ** 2')
        self.assertTrue(bool((qn <= q).all()))
        qi = quant(np.array([1.0, 2.0, 3.0]), 'm')
        qi *= t
        self.assertTrue(torch.is_tensor(qi.m))
        self.assertTrue(torch.allclose(qi.m.double(), torch.tensor([1.0, 4.0, 9.0]).double()))
        qi = quant(np.array([1.0, 2.0, 3.0]), 'm')
        qi += q
        self.assertTrue(torch.is_tensor(qi.m))
        self.assertTrue(torch.allclose(qi.m.double(), 2 * t.double()))
        qi = quant(torch.tensor([1.0, 2.0, 3.0]), 'm')
        qi -= torch.zeros(3)
        self.assertTrue(torch.equal(qi.m, t))
        # torch.cat reconciles units like immlib.math.concatenate.
        r = torch.cat([quant(t, 'm'), quant(np.array([100.0]), 'cm')])
        self.assertTrue(torch.allclose(r.m.double(), torch.tensor([1.0, 2.0, 3.0, 1.0]).double()))
        with self.assertRaises(pint.DimensionalityError):
            torch.cat([quant(t), quant(t, 'm')])
    def test_quant_magnitudes(self):
        """Tests which magnitudes quant() accepts and how it stores them."""
        from immlib import quant, units, is_array
        import numpy as np, torch, pint, scipy.sparse as sps
        # Scalars become 0-dimensional arrays.
        for (x, dt) in [(5, np.int64), (2.5, np.float64), (1j, np.complex128),
                        (np.float32(1.5), np.float32), (True, np.bool_)]:
            with self.subTest(x=x):
                q = quant(x)
                self.assertIsInstance(q.m, np.ndarray)
                self.assertEqual(q.m.ndim, 0)
                self.assertEqual(q.m.dtype, dt)
                self.assertTrue(is_array(q))
                q = quant(x, 'm')
                self.assertIsInstance(q.m, np.ndarray)
                self.assertEqual(q.m.ndim, 0)
        # Lists and tuples become arrays.
        q = quant([1, 2, 3], 'mm')
        self.assertIsInstance(q.m, np.ndarray)
        self.assertEqual(q.m.shape, (3,))
        self.assertIsInstance(quant((1.0, 2.0)).m, np.ndarray)
        # Arrays, tensors, and sparse matrices are kept as they are.
        a = np.arange(3)
        self.assertIs(quant(a).m, a)
        a0 = np.array(3.0)
        self.assertIs(quant(a0, 'm').m, a0)
        t = torch.arange(3)
        self.assertIs(quant(t).m, t)
        sp = sps.eye(3, format='csr')
        self.assertIs(quant(sp).m, sp)
        # Quantities are not re-wrapped, and unit=None converts scalars.
        pq = units.Quantity(5, 'm')
        self.assertIs(quant(pq), pq)
        q = quant(pq, None)
        self.assertIsNone(q.units)
        self.assertIsInstance(q.m, np.ndarray)
        # Non-numerical values are rejected.
        for x in ['5 mm', b'5', ['a', 'b'], [1, None], {'a': 1}, None,
                  np.array([object()]), np.array(['x']), [[1], [1, 2]]]:
            with self.subTest(x=x):
                with self.assertRaises(TypeError):
                    quant(x)
                with self.assertRaises(TypeError):
                    quant(x, 'm')
        # Strings can still be parsed by the unit registry.
        self.assertEqual(units.Quantity('5 mm'), quant(5, 'mm'))
    def test_quant_0d_behavior(self):
        """Tests that 0-dimensional magnitudes behave like scalars."""
        from immlib import quant
        import numpy as np, torch
        q = quant(5, 'm')
        # Display uses the scalar, not a matrix.
        self.assertEqual(q._repr_latex_(), '$5\\ \\mathrm{meter}$')
        self.assertEqual(f"{quant(2.5):.3f}", '2.500')
        self.assertEqual(f"{quant(2.5, 'm'):.1f~P}", '2.5 m')
        self.assertEqual(str(q), '5 meter')
        # In-place operators rebind rather than modify the array, so integer
        # magnitudes can become floats and other references are unchanged.
        m = q.m
        q *= 2.5
        self.assertEqual(float(q.m_as('m')), 12.5)
        self.assertEqual(int(m), 5)
        q = quant(5, 'm')
        q.ito('mm')
        self.assertEqual(float(q.m), 5000.0)
        self.assertEqual(str(q.units), 'millimeter')
        n = quant(5)
        n += 2.5
        self.assertIsNone(n.units)
        self.assertEqual(float(n), 7.5)
        p = quant(2, 'm')
        p **= 2
        self.assertEqual(str(p.units), 'meter ** 2')
        self.assertEqual(int(p.m), 4)
        r = quant(5, 'm')
        r -= quant(torch.tensor(1.0), 'm')
        self.assertTrue(torch.is_tensor(r.m))
        # In-place operators on arrays with dimensions still work in place.
        a = np.array([1.0, 2.0])
        qa = quant(a, 'm')
        qa *= 2
        self.assertTrue(np.array_equal(a, [2.0, 4.0]))
        # round() works.
        self.assertEqual(round(quant(2.5, 'm')), quant(2, 'm'))
        self.assertEqual(float(round(quant(2.567, 'm'), 2).m), 2.57)
        self.assertTrue(np.allclose(round(quant([2.567], 'm'), 2).m, [2.57]))
        # bool(), float(), int(), and hash() work.
        self.assertFalse(bool(quant(0)))
        self.assertTrue(bool(quant(3, 'm')))
        self.assertEqual(float(quant(2.5)), 2.5)
        self.assertEqual(int(quant(3)), 3)
        self.assertEqual(hash(quant(3)), hash(3))
    def test_numpy_dispatch_none_units(self):
        """NumPy functions and ufunc methods work for quantities whose units
        are None."""
        from immlib import quant, Quantity
        import numpy as np
        a = np.array([1.0, 2.0, 3.0])
        n = quant(a)
        r = quant(a, 'm')
        def check(res, expected):
            self.assertIsInstance(res, Quantity)
            self.assertIsNone(res.units)
            self.assertTrue(np.allclose(res.m, expected))
        check(np.add.reduce(n), 6.0)
        check(np.add.accumulate(n), np.cumsum(a))
        check(np.multiply.outer(n, n), np.multiply.outer(a, a))
        check(np.cumsum(n), np.cumsum(a))
        check(np.cumprod(n), np.cumprod(a))
        check(np.dot(n, n), 14.0)
        check(np.diff(n), np.diff(a))
        check(np.linalg.norm(n), np.linalg.norm(a))
        check(np.log1p(n), np.log1p(a))
        check(np.hypot(n, n), np.hypot(a, a))
        check(np.sort(n), a)
        (frac, whole) = np.modf(n)
        check(frac, np.modf(a)[0])
        check(whole, np.modf(a)[1])
        out = quant(np.zeros(3))
        self.assertIs(np.add(n, n, out=out), out)
        check(out, 2 * a)
        # Boolean and index results are plain.
        self.assertIsInstance(np.isfinite(n), np.ndarray)
        self.assertNotIsInstance(np.argmax(n), Quantity)
        self.assertTrue(np.allclose(n, n))
        # With real units present, unit-less values act as bare values.
        res = np.dot(n, r)
        self.assertEqual(str(res.units), 'meter')
        self.assertEqual(float(res.m), 14.0)
        import pint
        with self.assertRaises(pint.DimensionalityError):
            np.hypot(n, r)
        # Real dimensionless quantities keep their units.
        res = np.cumsum(quant(a, 'dimensionless'))
        self.assertEqual(res.units, quant(1, 'dimensionless').units)
    def test_quantity_persist(self):
        """Tests Quantity.persist, is_persistent, and the refusals they
        imply."""
        import copy, pickle
        import numpy as np, torch, pint
        from immlib import quant, Quantity
        for make in (lambda: quant(np.array([1.0, 2.0, 3.0]), 'mm'),
                     lambda: quant(torch.tensor([1.0, 2.0, 3.0]), 'mm')):
            q = make()
            # A new quantity is not persistent, and persist returns the
            # same object rather than a copy.
            self.assertFalse(q.is_persistent)
            self.assertIs(q.persist(), q)
            self.assertTrue(q.is_persistent)
            # Persisting twice is fine.
            self.assertIs(q.persist(), q)
            # The in-place unit conversions are refused.
            for name in ('ito', 'ito_root_units', 'ito_base_units'):
                with self.assertRaises(TypeError):
                    if name == 'ito':
                        q.ito('cm')
                    else:
                        getattr(q, name)()
            # So is item assignment, and attribute assignment or deletion.
            with self.assertRaises(TypeError):
                q[0] = quant(5.0, 'mm')
            with self.assertRaises(TypeError):
                q._magnitude = None
            with self.assertRaises(TypeError):
                q._units = None
            with self.assertRaises(TypeError):
                q._persistent = False
            with self.assertRaises(TypeError):
                q.newattr = 10
            with self.assertRaises(TypeError):
                del q._units
            # None of that changed the quantity.
            self.assertEqual(str(q.units), 'millimeter')
            self.assertTrue(np.allclose(
                np.asarray(q.m), np.array([1.0, 2.0, 3.0])))
            # Reading and computing all still work; in particular the
            # dimensionality property, which pint computes lazily and
            # caches on the quantity itself.
            self.assertEqual(q.dimensionality, make().dimensionality)
            self.assertEqual(q.dimensionality, make().dimensionality)
            self.assertEqual(float(q.sum().m), 6.0)
            self.assertTrue(np.allclose(np.asarray((q + q).m),
                                        np.array([2.0, 4.0, 6.0])))
            self.assertAlmostEqual(float(q.to('cm').m[0]), 0.1, places=5)
            # The in-place operators return new quantities instead of
            # mutating, as they do for an int.
            for op in ('__iadd__', '__isub__'):
                r = getattr(q, op)(quant(1.0, 'mm'))
                self.assertIsNot(r, q)
                self.assertFalse(r.is_persistent)
            for op in ('__imul__', '__itruediv__', '__ipow__'):
                r = getattr(q, op)(2)
                self.assertIsNot(r, q)
                self.assertFalse(r.is_persistent)
            # (//= and %= are meaningful only without real units.)
            nq = quant(q.m, None).persist()
            for op in ('__ifloordiv__', '__imod__'):
                r = getattr(nq, op)(2)
                self.assertIsNot(r, nq)
                self.assertFalse(r.is_persistent)
            self.assertTrue(np.allclose(
                np.asarray(q.m), np.array([1.0, 2.0, 3.0])))
            # The statement form rebinds the name and leaves q alone.
            p = q
            p += quant(1.0, 'mm')
            self.assertIsNot(p, q)
            self.assertEqual(float(q.m[0]), 1.0)
            # Copies of a persistent quantity are persistent; a quantity
            # rebuilt from the magnitude and units is not.
            self.assertTrue(copy.copy(q).is_persistent)
            self.assertTrue(copy.deepcopy(q).is_persistent)
            self.assertTrue(pickle.loads(pickle.dumps(q)).is_persistent)
            self.assertFalse(quant(q.m, q.u).is_persistent)
            # A mutable quantity is unaffected by any of this.
            m = make()
            m += quant(1.0, 'mm')
            self.assertEqual(float(m.m[0]), 2.0)
            m.ito('cm')
            self.assertEqual(str(m.units), 'centimeter')
            m[0] = quant(5.0, 'cm')
            self.assertEqual(float(m.m[0]), 5.0)
        # A unit-less quantity persists in the same way.
        qn = quant(np.array([1.0, 2.0])).persist()
        self.assertTrue(qn.is_persistent)
        self.assertIsNone(qn.units)
        with self.assertRaises(TypeError):
            qn[0] = 5.0
        self.assertTrue(np.allclose((qn + qn).m, np.array([2.0, 4.0])))
        # A NumPy out= argument naming a persistent quantity is refused,
        # whether the quantity has units or not, and whether the call
        # goes through the ufunc protocol or the function protocol.
        out = quant(np.zeros(2)).persist()
        with self.assertRaises(TypeError):
            np.add(qn, qn, out=out)
        self.assertTrue(np.allclose(out.m, np.zeros(2)))
        rout = quant(np.zeros(2), 'mm').persist()
        rq = quant(np.array([1.0, 2.0]), 'mm')
        with self.assertRaises(TypeError):
            np.add(rq, rq, out=rout)
        with self.assertRaises(TypeError):
            np.cumsum(rq, out=rout)
        self.assertTrue(np.allclose(rout.m, np.zeros(2)))
        # A mutable out= still works.
        mout = quant(np.zeros(2))
        self.assertIs(np.add(qn, qn, out=mout), mout)
        # The contents of the magnitude are not frozen by persist; that is
        # what immlib.freezearray is for.
        from immlib import freezearray
        qm = quant(np.array([1.0, 2.0]), 'mm').persist()
        qm.m[0] = 99.0
        self.assertEqual(float(qm.m[0]), 99.0)
        arr = np.array([1.0, 2.0])
        freezearray(arr)
        qf = quant(arr, 'mm').persist()
        with self.assertRaises(ValueError):
            qf.m[0] = 99.0
        # Persistence does not interfere with gradient tracking.
        g = quant(torch.tensor([1.0, 2.0], requires_grad=True), 'mm')
        g.persist()
        (g * g).sum().m.backward()
        self.assertTrue(np.allclose(g.m.grad.numpy(), np.array([2.0, 4.0])))
    def test_quantity_inplace_floordiv_mod(self):
        """Tests the in-place //= and %= operators on mutable quantities.

        These have three paths: a unit-less operand, a 0-dimensional
        magnitude (which quant stores as a 0-d array, and which Pint's own
        in-place code cannot handle), and everything else, which Pint
        handles.
        """
        import numpy as np, torch, pint
        from immlib import quant
        # A unit-less quantity divides and takes remainders like its own
        # magnitude, and does it in place.
        for mk in (lambda: quant(np.array([5.0, 7.0])),
                   lambda: quant(torch.tensor([5.0, 7.0]))):
            q = mk()
            r = q
            r //= 2
            self.assertIs(r, q)
            self.assertIsNone(r.units)
            self.assertTrue(np.allclose(np.asarray(r.m), [2.0, 3.0]))
            q = mk()
            r = q
            r %= 2
            self.assertIs(r, q)
            self.assertIsNone(r.units)
            self.assertTrue(np.allclose(np.asarray(r.m), [1.0, 1.0]))
        # The same for a 0-dimensional magnitude, which is what quant makes
        # of a scalar.
        q = quant(5.0)
        r = q
        r //= 2
        self.assertIs(r, q)
        self.assertEqual(float(r.m), 2.0)
        q = quant(5.0)
        r = q
        r %= 2
        self.assertIs(r, q)
        self.assertEqual(float(r.m), 1.0)
        # /= on a 0-d magnitude goes through the same machinery and keeps
        # real units.
        q = quant(5.0, 'mm')
        r = q
        r /= 2
        self.assertIs(r, q)
        self.assertEqual(float(r.m), 2.5)
        self.assertEqual(str(r.units), 'millimeter')
        # With real units, // and % are not meaningful, and Pint's error is
        # what comes out, whether the magnitude is 0-d or not.
        for q in (quant(5.0, 'mm'), quant(np.array([5.0, 7.0]), 'mm')):
            with self.assertRaises(pint.DimensionalityError):
                q //= 2
            with self.assertRaises(pint.DimensionalityError):
                q %= 2
        # A real dimensionless quantity, however, is fine, and that is the
        # path that Pint itself handles.
        q = quant(np.array([5.0, 7.0]), 'dimensionless')
        r = q
        r //= 2
        self.assertIs(r, q)
        self.assertTrue(np.allclose(r.m, [2.0, 3.0]))
    def test_quantity_reflected_divmod_and_pow(self):
        """Tests divmod and ** with the quantity on the right."""
        import numpy as np
        from immlib import quant, Quantity
        # divmod(x, q) for a unit-less q.
        for q in (quant(2.0), quant(np.array([2.0, 4.0]))):
            (d, m) = divmod(7, q)
            self.assertIsInstance(d, Quantity)
            self.assertIsNone(d.units)
            self.assertTrue(np.allclose(np.asarray(d.m), 7 // np.asarray(q.m)))
            self.assertTrue(np.allclose(np.asarray(m.m), 7 % np.asarray(q.m)))
        # x ** q for a unit-less q gives a bare value, since the exponent
        # carries no units and neither does the base.
        self.assertEqual(2 ** quant(3.0), 8.0)
        # A real dimensionless quantity is Pint's own path.
        self.assertEqual(2 ** quant(3.0, 'dimensionless'), 8.0)
        # **= on a quantity with real units is Pint's path too, and it
        # changes the units.
        q = quant(np.array([2.0, 3.0]), 'mm')
        q **= 2
        self.assertEqual(str(q.units), 'millimeter ** 2')
        self.assertTrue(np.allclose(q.m, [4.0, 9.0]))
    def test_quantity_round_and_complex(self):
        """Tests __round__ for each kind of magnitude, and __complex__."""
        import numpy as np, torch, pint
        from immlib import quant, Quantity
        # A 0-d array magnitude, which is what quant makes of a scalar.
        r = round(quant(1.2345, 'mm'), 2)
        self.assertAlmostEqual(float(r.m), 1.23)
        self.assertEqual(str(r.units), 'millimeter')
        # An array magnitude.
        r = round(quant(np.array([1.2345, 2.3456]), 'mm'), 2)
        self.assertTrue(np.allclose(r.m, [1.23, 2.35]))
        # A tensor magnitude, which Pint would have converted to an array.
        r = round(quant(torch.tensor([1.2345, 2.3456]), 'mm'), 2)
        self.assertTrue(torch.is_tensor(r.m))
        self.assertTrue(np.allclose(r.m.numpy(), [1.23, 2.35]))
        # With no digits, rounding is to the nearest integer.
        r = round(quant(torch.tensor([1.6, 2.4]), 'mm'))
        self.assertTrue(np.allclose(r.m.numpy(), [2.0, 2.0]))
        # A plain Python number magnitude, which a Quantity built directly
        # (rather than through quant) can have.
        q = Quantity(1.2345, 'mm')
        self.assertIsInstance(q._magnitude, float)
        self.assertAlmostEqual(float(round(q, 2).m), 1.23)
        # complex() works for a unit-less quantity and for a real
        # dimensionless one, and raises for anything with dimensions.
        self.assertEqual(complex(quant(1.0)), 1 + 0j)
        self.assertEqual(complex(quant(1.0, 'dimensionless')), 1 + 0j)
        with self.assertRaises(pint.DimensionalityError):
            complex(quant(1.0, 'mm'))
    def test_quantity_torch_function_declines(self):
        """Tests the cases in which __torch_function__ declines to handle a
        call, and the unary-operator functions that it does handle."""
        import numpy as np, torch
        from immlib import quant, Quantity
        q = quant(torch.tensor([1.0, -2.0]), 'mm')
        # The unary operators are handled by calling the quantity's own
        # dunder method, so units are preserved.
        for (fn, want) in ((torch.neg, [-1.0, 2.0]),
                           (torch.negative, [-1.0, 2.0]),
                           (torch.abs, [1.0, 2.0]),
                           (torch.absolute, [1.0, 2.0]),
                           (torch.positive, [1.0, -2.0])):
            r = fn(q)
            self.assertIsInstance(r, Quantity)
            self.assertEqual(str(r.units), 'millimeter')
            self.assertTrue(np.allclose(r.m.numpy(), want))
        # A unary call with keyword arguments is declined, and PyTorch then
        # reports that no implementation accepted it.
        with self.assertRaises(TypeError):
            torch.neg(q, out=None)
        # So is a delegated math function with the wrong number of arguments
        # or with no quantity among them.
        with self.assertRaises(TypeError):
            torch.exp(q, out=None)
        self.assertTrue(torch.is_tensor(torch.exp(torch.tensor([1.0]))))
        # A unit-preserving function and a reduction both need a quantity as
        # their first argument; with a plain tensor they are ordinary torch
        # calls.
        self.assertTrue(torch.is_tensor(torch.squeeze(torch.tensor([[1.0]]))))
        self.assertTrue(torch.is_tensor(torch.sum(torch.tensor([1.0]))))
        # And with a quantity they keep the units.
        self.assertEqual(str(torch.sum(q).units), 'millimeter')
        self.assertEqual(str(torch.squeeze(q).units), 'millimeter')
    def test_quantity_cross_registry_none_units(self):
        """Tests a unit-less quantity combined with a quantity from another
        registry, which must come back in this registry's Quantity class."""
        import pint
        from immlib import quant, Quantity, units
        other = pint.UnitRegistry()
        r = quant(2.0) * other.Quantity(3.0, 'mm')
        # The result is an immlib Quantity in immlib's registry, not the
        # other registry's class.
        self.assertIsInstance(r, Quantity)
        self.assertIs(r._REGISTRY, units)
        self.assertEqual(str(r.units), 'millimeter')
        self.assertEqual(float(r.m), 6.0)
    def test_quantity_nonmultiplicative_tensor_eq(self):
        """Tests comparing a quantity in a non-multiplicative unit to zero.

        Degrees Celsius has an offset, so comparing it to a bare zero is
        ambiguous: Pint raises unless the registry is set to convert offset
        units to base units automatically, in which case it compares in base
        units. Rule 1 requires the tensor path to do the same as the array
        path in both settings; it used to convert unconditionally, so the
        array comparison raised while the tensor comparison answered False.
        """
        import numpy as np, torch, pint
        from immlib import quant, UnitRegistry
        # By default the comparison is ambiguous and raises, for both
        # backends and both operators.
        for mag in (np.array(0.0), torch.tensor(0.0),
                    np.array([0.0, 100.0]), torch.tensor([0.0, 100.0])):
            q = quant(mag, 'degC')
            with self.assertRaises(pint.OffsetUnitCalculusError):
                q == 0
            with self.assertRaises(pint.OffsetUnitCalculusError):
                q != 0
        # With autoconvert_offset_to_baseunit, both backends compare in base
        # units and give the same answer: 0 degC is 273.15 K, not zero.
        ureg = UnitRegistry(autoconvert_offset_to_baseunit=True)
        for (mag_np, mag_t) in ((np.array(0.0), torch.tensor(0.0)),
                                (np.array([0.0, 100.0]),
                                 torch.tensor([0.0, 100.0]))):
            qn = ureg.Quantity(mag_np, 'degC')
            qt = ureg.Quantity(mag_t, 'degC')
            self.assertTrue(
                np.array_equal(np.asarray(qn == 0), (qt == 0).numpy()))
            self.assertTrue(
                np.array_equal(np.asarray(qn != 0), (qt != 0).numpy()))
            self.assertFalse(np.any(np.asarray(qn == 0)))
    def test_promote_mixed_quantities(self):
        """Tests promote with several quantities, only one of which is a
        tensor."""
        import numpy as np, torch
        from immlib import promote, quant, Quantity
        args = promote(
            quant(torch.tensor([1.0]), 'mm'),
            quant(np.array([2.0]), 'mm'),
            quant(3.0, 'mm'),
            4.0)
        # Everything becomes a tensor, and the quantities stay quantities
        # with their units.
        for a in args[:3]:
            self.assertIsInstance(a, Quantity)
            self.assertTrue(torch.is_tensor(a.m))
            self.assertEqual(str(a.units), 'millimeter')
        self.assertTrue(torch.is_tensor(args[3]))
    def test_quantwrap_basics(self):
        """Tests that quantwrap converts arguments into quantities."""
        import numpy as np
        from immlib import quantwrap, quant, Quantity
        # Every argument becomes a quantity; one that was not a quantity
        # gets units of None.
        @quantwrap
        def f(a, b):
            return (a, b)
        (a, b) = f(1.0, quant(2.0, 'mm'))
        self.assertIsInstance(a, Quantity)
        self.assertIsInstance(b, Quantity)
        self.assertIsNone(a.units)
        self.assertEqual(str(b.units), 'millimeter')
        # The decorated function keeps its name and documentation.
        @quantwrap
        def documented(a):
            "A docstring."
            return a
        self.assertEqual(documented.__name__, 'documented')
        self.assertEqual(documented.__doc__, "A docstring.")
        # Naming arguments touches only those. What the function is given
        # has to be checked inside it: a caller who passes no quantities is
        # answered with magnitudes, whatever the function returns.
        seen = {}
        def record(a, b):
            seen['a'] = isinstance(a, Quantity)
            seen['b'] = isinstance(b, Quantity)
        quantwrap('a')(record)(1.0, 2.0)
        self.assertTrue(seen['a'])
        self.assertFalse(seen['b'])
        # The same names can be given after the function instead.
        seen.clear()
        quantwrap(record, 'a')(1.0, 2.0)
        self.assertTrue(seen['a'])
        self.assertFalse(seen['b'])
        # And the decorator can be written with empty parentheses, which
        # touches every argument.
        seen.clear()
        quantwrap()(record)(1.0, 2.0)
        self.assertTrue(seen['a'])
        self.assertTrue(seen['b'])
        # A first argument that is neither a name nor a callable is an
        # error.
        with self.assertRaises(TypeError):
            quantwrap(10)
    def test_quantwrap_return_rules(self):
        """Tests how quantwrap decides the units of the return value."""
        import numpy as np
        from immlib import quantwrap, quant, Quantity
        @quantwrap
        def add(a, b):
            return a + b
        # A caller who passes no quantities is answered with a magnitude.
        r = add(1.0, 2.0)
        self.assertNotIsInstance(r, Quantity)
        self.assertEqual(float(r), 3.0)
        # A caller who passes one is answered with a quantity.
        r = add(quant(1.0, 'mm'), quant(2.0, 'mm'))
        self.assertIsInstance(r, Quantity)
        self.assertEqual(str(r.units), 'millimeter')
        # A bare argument becomes a quantity with no units, which is not
        # the same as a dimensionless one: adding it to a length is still
        # the error it always was.
        import pint
        with self.assertRaises(pint.DimensionalityError):
            add(quant(1.0, 'mm'), 2.0)
        # Multiplying by one is fine, as it is for a bare number.
        @quantwrap
        def scale(a, b):
            return a * b
        r = scale(quant(2.0, 'mm'), 3.0)
        self.assertEqual(str(r.units), 'millimeter')
        self.assertAlmostEqual(float(r.m), 6.0)
        # runit converts the result, and says the caller cares about units,
        # so the plain-numbers rule does not apply.
        @quantwrap(runit='mm')
        def ident(a):
            return a
        r = ident(quant(1.0, 'm'))
        self.assertIsInstance(r, Quantity)
        self.assertEqual(str(r.units), 'millimeter')
        self.assertAlmostEqual(float(r.m), 1000.0)
        r = ident(5.0)
        self.assertIsInstance(r, Quantity)
        self.assertEqual(str(r.units), 'millimeter')
        # An incompatible unit is an error; a bare magnitude is not, since
        # runit assumes the magnitude is already in the named unit.
        with self.assertRaises(Exception):
            ident(quant(1.0, 's'))
        # runit=None gives the result no units.
        @quantwrap(runit=None)
        def unitless(a):
            return a
        self.assertIsNone(unitless(quant(1.0, 'mm')).units)
        # return_quant is applied after runit.
        @quantwrap(runit='mm', return_quant=False)
        def mm_mag(a):
            return a
        r = mm_mag(quant(1.0, 'm'))
        self.assertNotIsInstance(r, Quantity)
        self.assertAlmostEqual(float(r), 1000.0)
        # return_quant=True wraps whatever is not a quantity.
        @quantwrap(return_quant=True)
        def always_q(a):
            return a
        self.assertIsInstance(always_q(1.0), Quantity)
        self.assertIsNone(always_q(1.0).units)
        # return_quant=False strips whatever is.
        @quantwrap(return_quant=False)
        def never_q(a):
            return a
        self.assertNotIsInstance(never_q(quant(1.0, 'mm')), Quantity)
    def test_quantwrap_units_and_requirements(self):
        """Tests the units and require_units options."""
        import numpy as np
        from immlib import quantwrap, quant, Quantity
        # units converts an argument into the named unit, and accepts both a
        # bare value and a quantity.
        @quantwrap(units={'x': 'mm'}, return_quant=True)
        def f(x):
            return x
        self.assertEqual(str(f(10).units), 'millimeter')
        self.assertAlmostEqual(float(f(quant(1.0, 'm')).m), 1000.0)
        self.assertEqual(str(f(quant(1.0, 'm')).units), 'millimeter')
        with self.assertRaises(Exception):
            f(quant(1.0, 's'))
        # require_units insists on a quantity in a compatible unit, and
        # converts it.
        @quantwrap(require_units={'x': 'mm'}, return_quant=True)
        def g(x):
            return x
        self.assertAlmostEqual(float(g(quant(1.0, 'm')).m), 1000.0)
        with self.assertRaises(TypeError):
            g(10)
        with self.assertRaises(ValueError):
            g(quant(1.0, 's'))
        # A required unit of None means the argument must be a quantity with
        # no units, which is not the same as dimensionless.
        @quantwrap(require_units={'x': None}, return_quant=True)
        def h(x):
            return x
        self.assertIsNone(h(quant(1.0)).units)
        with self.assertRaises(ValueError):
            h(quant(1.0, 'dimensionless'))
        with self.assertRaises(ValueError):
            h(quant(1.0, 'mm'))
        # Naming an argument in units also asks for it to be touched, even
        # when other arguments are named positionally.
        seen = {}
        @quantwrap('a', units={'b': 'mm'})
        def k(a, b, c):
            seen.update(a=a, b=b, c=c)
        k(1.0, 2.0, 3.0)
        self.assertIsInstance(seen['a'], Quantity)
        self.assertIsNone(seen['a'].units)
        self.assertEqual(str(seen['b'].units), 'millimeter')
        self.assertNotIsInstance(seen['c'], Quantity)
        # Decoration-time errors.
        with self.assertRaises(ValueError):
            quantwrap('nosucharg')(lambda a: a)
        with self.assertRaises(ValueError):
            quantwrap(units={'nosucharg': 'mm'})(lambda a: a)
        with self.assertRaises(ValueError):
            quantwrap(units={'a': 'mm'}, require_units={'a': 'mm'})(
                lambda a: a)
        with self.assertRaises(TypeError):
            quantwrap(units=10)(lambda a: a)
    def test_quantwrap_require_runit(self):
        """Tests the require_runit option."""
        from immlib import quantwrap, quant, Quantity
        @quantwrap(require_runit='mm', return_quant=True)
        def f(a):
            return quant(1.0, 'm')
        r = f(1.0)
        self.assertEqual(str(r.units), 'millimeter')
        self.assertAlmostEqual(float(r.m), 1000.0)
        # A function that returns the wrong dimension is an error.
        @quantwrap(require_runit='mm')
        def g(a):
            return quant(1.0, 's')
        with self.assertRaises(ValueError):
            g(1.0)
        # As is one that returns something with no units at all.
        @quantwrap(require_runit='mm')
        def h(a):
            return quant(1.0)
        with self.assertRaises(ValueError):
            h(1.0)
        # A tuple requirement applies one unit per element, and the lengths
        # must match.
        @quantwrap(require_runit=('mm', 's'), return_quant=True)
        def tup(a):
            return (quant(1.0, 'm'), quant(2.0, 'ms'))
        (x, y) = tup(1.0)
        self.assertAlmostEqual(float(x.m), 1000.0)
        self.assertAlmostEqual(float(y.m), 0.002)
        @quantwrap(require_runit=('mm', 's'))
        def short(a):
            return (quant(1.0, 'm'),)
        with self.assertRaises(ValueError):
            short(1.0)
        # A mapping requirement applies one unit per key, and the keys must
        # match.
        @quantwrap(require_runit={'x': 'mm'}, return_quant=True)
        def dct(a):
            return {'x': quant(1.0, 'm')}
        self.assertAlmostEqual(float(dct(1.0)['x'].m), 1000.0)
        @quantwrap(require_runit={'x': 'mm'})
        def wrongkeys(a):
            return {'y': quant(1.0, 'm')}
        with self.assertRaises(ValueError):
            wrongkeys(1.0)
        # A requirement shaped differently from the return value is an
        # error rather than a silent mismatch.
        @quantwrap(require_runit=('mm', 's'))
        def notuple(a):
            return quant(1.0, 'm')
        with self.assertRaises(TypeError):
            notuple(1.0)
    def test_quantwrap_tuples_and_mappings(self):
        """Tests quantwrap's handling of tuple and mapping return values."""
        import numpy as np
        from immlib import quantwrap, quant, Quantity
        from pcollections import pdict, ldict, lazy
        # Every quantity in a returned tuple is stripped when the caller
        # passed none, and items that are not quantities are left alone.
        @quantwrap
        def tup(a):
            return (a, 'text', 5)
        r = tup(1.0)
        self.assertNotIsInstance(r[0], Quantity)
        self.assertEqual(r[1], 'text')
        self.assertEqual(r[2], 5)
        self.assertIsInstance(tup(quant(1.0, 'mm'))[0], Quantity)
        # The same for a mapping, which keeps its own type.
        @quantwrap
        def dct(a):
            return {'x': a, 'y': 5}
        r = dct(1.0)
        self.assertIsInstance(r, dict)
        self.assertNotIsInstance(r['x'], Quantity)
        self.assertEqual(r['y'], 5)
        self.assertIsInstance(dct(quant(1.0, 'mm'))['x'], Quantity)
        @quantwrap
        def pd(a):
            return pdict(x=a, y=5)
        r = pd(1.0)
        self.assertIsInstance(r, pdict)
        self.assertNotIsInstance(r['x'], Quantity)
        # A lazy dictionary survives too, and the values that did not change
        # are left lazy.
        @quantwrap
        def ld(a):
            return ldict({'x': a, 'y': lazy(lambda: 5)})
        r = ld(1.0)
        self.assertIsInstance(r, ldict)
        self.assertNotIsInstance(r['x'], Quantity)
        self.assertEqual(r['y'], 5)
        # A mapping whose keys are not strings cannot be rebuilt with
        # keyword arguments, so quantwrap falls back to rebuilding it from a
        # mapping.
        @quantwrap
        def intkeys(a):
            return {1: a, 2: 5}
        r = intkeys(1.0)
        self.assertNotIsInstance(r[1], Quantity)
        self.assertEqual(r[2], 5)
        # runit applies to each element of a tuple, and a tuple of units
        # applies one to each.
        @quantwrap(runit='mm')
        def tup2(a):
            return (a, a)
        self.assertTrue(all(str(u.units) == 'millimeter' for u in tup2(1.0)))
        @quantwrap(runit=('mm', 'm'))
        def tup3(a):
            return (a, a)
        (x, y) = tup3(quant(1.0, 'm'))
        self.assertEqual(str(x.units), 'millimeter')
        self.assertEqual(str(y.units), 'meter')
    def test_quantwrap_variadic(self):
        """Tests quantwrap on *args and **kwargs parameters."""
        from immlib import quantwrap, quant, Quantity
        # What the function receives is checked inside it, since a nested
        # tuple or mapping is one element to quantwrap and is not
        # traversed.
        seen = {}
        # The unit named for a *args parameter applies to all of them.
        @quantwrap(units={'rest': 'mm'})
        def f(a, *rest):
            seen.update(a=a, rest=rest)
        f(1.0, 2.0, quant(3.0, 'm'))
        self.assertIsNone(seen['a'].units)
        self.assertEqual([str(u.units) for u in seen['rest']],
                         ['millimeter', 'millimeter'])
        self.assertAlmostEqual(float(seen['rest'][1].m), 3000.0)
        # And the one named for a **kwargs parameter to all of its values.
        @quantwrap(units={'kw': 's'})
        def g(a, **kw):
            seen.update(a=a, kw=kw)
        g(1.0, b=2.0, c=quant(3.0, 'ms'))
        self.assertEqual(str(seen['kw']['b'].units), 'second')
        self.assertAlmostEqual(float(seen['kw']['c'].m), 0.003)
        # Empty variadic parameters are fine.
        f(1.0)
        self.assertEqual(seen['rest'], ())
        g(1.0)
        self.assertEqual(seen['kw'], {})
        # A quantity anywhere, including in the variadic arguments, means
        # the result keeps its units.
        @quantwrap
        def h(*vals):
            return sum(vals[1:], vals[0])
        self.assertNotIsInstance(h(1.0, 2.0), Quantity)
        self.assertIsInstance(h(quant(1.0, 'mm'), quant(2.0, 'mm')),
                              Quantity)
    def test_quantwrap_registries(self):
        """Tests quantwrap's ureg option and its registry checks."""
        import pint
        from immlib import quantwrap, quant, Quantity, units, UnitRegistry
        other = UnitRegistry()
        # By default everything is re-homed into immlib's default registry.
        @quantwrap(return_quant=True)
        def f(a):
            return a
        r = f(other.Quantity(1.0, 'mm'))
        self.assertIs(r._REGISTRY, units)
        # Arguments that disagree about their registry are an error when no
        # registry was named, since there is then no unambiguous answer.
        @quantwrap
        def g(a, b):
            return a
        with self.assertRaises(ValueError):
            g(quant(1.0, 'mm'), other.Quantity(2.0, 'mm'))
        # Naming one resolves it.
        @quantwrap(ureg=units, return_quant=True)
        def h(a, b):
            return a
        r = h(quant(1.0, 'mm'), other.Quantity(2.0, 'mm'))
        self.assertIs(r._REGISTRY, units)
        # A plain pint registry cannot represent units of None, so it is
        # refused, as is ureg=None, which asks for no registry at all.
        with self.assertRaises(TypeError):
            quantwrap(ureg=pint.UnitRegistry())(lambda a: a)
        with self.assertRaises(ValueError):
            quantwrap(ureg=None)(lambda a: a)
        with self.assertRaises(TypeError):
            quantwrap(ureg=10)(lambda a: a)
    def test_quantwrap_rules(self):
        """Tests that quantwrap obeys immlib.math's two rules.

        Rule 1: the wrapper gives equal results for equal arrays and
        tensors, with the result's type following the input's. Rule 2: it
        does not break PyTorch's gradient tracking, which means it must
        never convert a magnitude, and unit conversion must stay a
        multiplication.
        """
        import numpy as np, torch
        from immlib import quantwrap, quant, Quantity
        @quantwrap(units={'a': 'mm'}, runit='m')
        def f(a):
            return a * 2
        rn = f(np.array([1000.0, 2000.0]))
        rt = f(torch.tensor([1000.0, 2000.0]))
        # Rule 1: same units, same values, types following the inputs.
        self.assertEqual(str(rn.units), str(rt.units))
        self.assertTrue(np.allclose(np.asarray(rn.m), rt.m.numpy()))
        self.assertIsInstance(rn.m, np.ndarray)
        self.assertTrue(torch.is_tensor(rt.m))
        # Rule 2: a tensor that requires grad survives the wrapper, the
        # unit conversion, and the return.
        x = torch.tensor([1.0, 2.0], requires_grad=True)
        @quantwrap
        def g(a):
            return a * 2
        r = g(x)
        self.assertTrue(torch.is_tensor(r))
        self.assertIsNotNone(r.grad_fn)
        r.sum().backward()
        self.assertTrue(np.allclose(x.grad.numpy(), [2.0, 2.0]))
        # And through a conversion, which is a multiply.
        x = torch.tensor([1.0], requires_grad=True)
        @quantwrap(runit='mm')
        def h(a):
            return a
        r = h(quant(x, 'm'))
        self.assertEqual(str(r.units), 'millimeter')
        self.assertIsNotNone(r.m.grad_fn)
        r.m.sum().backward()
        self.assertTrue(np.allclose(x.grad.numpy(), [1000.0]))
        # The magnitude is never copied or moved between backends.
        y = torch.tensor([1.0, 2.0])
        @quantwrap
        def ident(a):
            return a.m
        self.assertIs(ident(y), y)
    def test_quantwrap_shape_mismatches(self):
        """Tests quantwrap's errors when a unit spec and a return value have
        different shapes, and the paths that leave a value alone."""
        from immlib import quantwrap, quant, Quantity
        from pcollections import pdict
        # A mapping of units against a returned tuple, and a tuple of units
        # against a returned mapping, are both errors rather than silent
        # mismatches.
        @quantwrap(runit={'x': 'mm'})
        def a(v):
            return (v,)
        with self.assertRaises(TypeError):
            a(1.0)
        @quantwrap(runit=('mm',))
        def b(v):
            return {'x': v}
        with self.assertRaises(TypeError):
            b(1.0)
        # A single unit applies to every value of a returned mapping.
        @quantwrap(runit='mm')
        def c(v):
            return {'x': v, 'y': v}
        r = c(1.0)
        self.assertEqual([str(u.units) for u in r.values()],
                         ['millimeter', 'millimeter'])
        # A tuple or mapping that holds no quantities is returned as it is.
        @quantwrap
        def d(v):
            return ('a', 'b')
        self.assertEqual(d(1.0), ('a', 'b'))
        @quantwrap
        def e(v):
            return pdict(x=1, y=2)
        self.assertEqual(e(1.0), pdict(x=1, y=2))
        # So is a return value that is not a quantity at all.
        @quantwrap
        def f(v):
            return 'text'
        self.assertEqual(f(1.0), 'text')
        # return_quant=True cannot wrap something that is not numerical.
        @quantwrap(return_quant=True)
        def g(v):
            return 'text'
        with self.assertRaises(TypeError):
            g(1.0)
        # A mapping type that can be rebuilt from neither keyword arguments
        # nor a mapping is reported rather than silently mishandled.
        class OddMap(dict):
            def __init__(self, *args, **kw):
                raise RuntimeError("cannot rebuild me")
        @quantwrap(return_quant=False)
        def h(v):
            m = OddMap.__new__(OddMap)
            dict.__init__(m, x=quant(1.0, 'mm'))
            return m
        with self.assertRaises(TypeError):
            h(1.0)
