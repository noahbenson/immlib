# -*- coding: utf-8 -*-
###############################################################################
# immlib/workflow/_plantype.py


# Dependencies ################################################################

import inspect
from functools import wraps
from copy import copy

from pcollections import (pdict, ldict, lazy)

from ..doc import docwrap
from ..util import (is_str, is_pdict, is_tdict, is_ldict, assoc, merge)
from ._core import (calc, plan, plandict, tplandict, is_calcfn)


# #plantype ###################################################################

class plantype(type):
    """A metaclass that allows one to create lazy types from calculation plans.

    The ``plantype`` metaclass handles classes with the base-class
    ``planobject``.  In general, one should create a plan-object by inheriting
    from ``planobject``, not by providing the ``plantype`` metaclass, but
    passing ``plantype`` has the same effect (all classes created with
    metaclass ``plantype`` will inherit from ``planobject``).

    See ``planobject`` for more information.
    """
    __slots__ = ()
    class planobject_base:
        """The base-class for the ``immlib.planobject`` class.

        ``plantype.planobject_base`` is a simple class that implements the
        basic features of the ``planobject`` class. The separation for certain
        methods from the ``planobject`` type itself is required due to details
        of how the ``planobject`` class, which has the ``plantype`` meta-class,
        gets initialized while the ``plantype.__new__`` method depends on
        methods in the ``planobject`` class (which hasn't been
        initialized/defined at the time that the ``planobject.__new__`` method
        is called. This class shouldn't be used directly and shouldn't be
        inherited. Use the ``planobject`` class instead.
        """
        __slots__ = ('_plandict_',)
        def __getattr__(self, k):
            pd = self._plandict_
            r = pd.get(k, pd)
            if r is pd:
                raise AttributeError
            else:
                return r
        def __setattr__(self, k, v):
            pd = self._plandict_
            pdtype = type(pd)
            plan = type(self).plan
            if pdtype is dict or pdtype is tplandict:
                if k not in plan.inputs:
                    msg = "only planobject inputs may be mutated in "
                    if pdtype is dict:
                        msg += "the __init__ method"
                    else:
                        msg += "transient planobjects"
                    raise ValueError(msg)
                else:
                    pd[k] = v
            else:
                raise TypeError(f"type {type(self)} is immutable")
        def __delattr__(self, k):
            pd = self._plandict_
            if type(pd) is dict:
                raise TypeError(
                    f"cannot delete attributes from plantype: {type(self)}")
            else:
                raise TypeError(f"type {type(self)} is immutable")
        def __new__(cls, *args, **kwargs):
            # Start by creating the object itself and setting up its slots.
            obj = object.__new__(cls)
            object.__setattr__(obj, '_plandict_', None)
            # Once the __init__ function is done running, the plandict will be
            # cleaned up (this is guaranteed by the plantype meta-class).
            return obj
        def __dir__(self):
            pd = self._plandict_
            l = object.__dir__(self)
            l.extend(pd.keys())
            l.sort()
            return l
        def __init__(self, *args, **kwargs):
            for (k,v) in merge(*args, **kwargs).items():
                setattr(self, k, v)
        @staticmethod
        def _init_wrapper(cls, self, inittrans, *args, **kwargs):
            """Manages the initialization (``__init__``) for ``planobject``
            types.

            The ``planobject`` type, and any types that inherit from it, is
            generally immutable; however, When a ``planobject`` is first
            created, it is allowed to set its inputs, as if they were mutable
            attributes, during the ``__init__()`` method. In order to
            facilitate this, a reorganization of the initialization code for
            each ``planobject`` subclass is performed when that class is
            defined. The class's true ``__init__`` method is stored in the
            method ``__planobject_init__``, while the
            ``plantype.planobject_base._init_wrapper`` method is stored in the
            type's ``__init__`` method. This method calls the type's
            ``__planobject_init__`` method then makes the initialized object
            immutable.

            This method should not be called directly by the user.
            """
            # If the plandict is not a mutable dictionary, we have already been
            # initialized.
            pd = self._plandict_
            if is_pdict(pd) or is_tdict(pd):
                raise RuntimeError(
                    "_init_wrapper method called on an already-initialized"
                    " planobject")
            elif pd is not None:
                # We're already in the middle of initializing; one of the
                # __init__ methods probably just called a parent class's
                # __init__ method. We can just run it and return.
                return cls.__planobject_init__(self, *args, **kwargs)
            # Otherwise, pd is None, meaning that we're the first initializer.
            # Note that we are now in the process of initializing...
            pd = {}
            object.__setattr__(self, '_plandict_', pd)
            # This method is the real initializer for the class (what the
            # class's actual code wrote as the __init__ method).
            cls.__planobject_init__(self, *args, **kwargs)
            # Postprocess the argument.
            theplan = type(self).plan
            params = dict(theplan.defaults, **pd)
            if params.keys() != theplan.inputs:
                raise ValueError(
                    f"bad parameters for plantype {type(self)};"
                    f" expected {tuple(theplan.inputs)} but found"
                    f" {tuple(params.keys())}")
            if inittrans:
                pd = tplandict(theplan, params)
            else:
                pd = theplan(params)
            object.__setattr__(self, '_plandict_', pd)
        # For the pickle module:
        def __getstate__(self):
            return dict(self._plandict_.inputs)
        def __setstate__(self, inputs):
            plan = type(self).plan
            object.__setattr__(self, '_plandict_', plan(inputs))
    def __new__(cls, name, bases, attrs, **kwargs):
        sup = super(plantype, cls)
        # Before we go too far, let's extract the valid args from kwargs.
        initplan = kwargs.pop('plan', None)
        inittrans = kwargs.pop('transient', False)
        if len(kwargs) > 0:
            ks = tuple(kwargs.keys())
            raise ValueError(f"unsupported type options: {ks}")
        # We want to go over the attributes and make a few changes:
        # (1) We want to make the basic updates.
        kvs = [
            ('__getattr__', plantype.planobject_base.__getattr__),
            ('__setattr__', plantype.planobject_base.__setattr__),
            ('__new__',     plantype.planobject_base.__new__),
            ('__dir__',     plantype.planobject_base.__dir__)]
        for (k,v) in kvs:
            if k in attrs:
                raise ValueError(f"plantype classes may not define {k}")
            else:
                attrs[k] = v
        # (2) We want to save the init function and update it to our version.
        init = attrs.get('__init__', plantype.planobject_base.__init__)
        attrs['__planobject_init__'] = init
        def _initfn(self, *args, **kwargs):
            return plantype.planobject_base._init_wrapper(
                _initfn.cls, self, inittrans, *args,
                **kwargs)
        attrs['__init__'] = wraps(init)(_initfn)
        # (3) Go through the bases: see if there are planobject bases already,
        #     and if not, add planobject in. As we go, collect calculations.
        calcs = {k:v for (k,v) in attrs.items() if is_calcfn(v)}
        found_planobj = False
        for b in bases:
            if not issubclass(b, plantype.planobject_base): continue
            found_planobj = True
            for (k,v) in inspect.getmembers(b):
                if k not in calcs and is_calcfn(v):
                    calcs[k] = v
        if not found_planobj:
            bases.append(plantype.planobject_base)
        # (4) We now want to create the plan from these calculations and make
        #     sure it's part of the class.
        if initplan is None:
            attrs['plan'] = plan(calcs)
        else:
            attrs['plan'] = plan(initplan, **calcs)
        # (5) Return the type with all the updated attributes.
        cls = sup.__new__(cls, name, bases, attrs)
        _initfn.cls = cls
        return cls


# #planobject #################################################################

class planobject(plantype.planobject_base, metaclass=plantype):
    """Base class for objects that are based on lazy calculation plans.

    ``planobject`` is the base-class for all objects that use ``immlib.plan``
    objects as their base type. Objects that inherit from ``planobject`` (which
    uses metaclass ``plantype``) are defined in the same way that calculation
    plans are defined. Any attributes of the class (including those inherited
    from base classes) that are calculations (see ``immlib.calc`` and
    ``immlib.plan``) are turned into a plan. The inputs of the plan are the
    required parameters for the class, and the outputs of the plan become the
    attributes of the class, which are resolved lazily like in ``plandict`` s.

    The ``__init__`` function of a ``planobject`` is special. During the
    ``__init__`` function only, the parameters of a ``planobject`` function can
    be set using the usual ``setattr`` interface. All ``planobject``s are
    immutable once they have been initialized, however. At the end of the
    ``__init__`` function, the object must have all of its parameters set,
    otherwise an error is raised. If no ``__init__`` function is defined, then
    the ``planobject`` default init function calls ``merge`` on its arguments
    and keywords; the resulting dict must be a dictionary of the class's
    parameters.

    ``planobject`` types must not overload the following methods, as they are
    used by the ``planobject`` / ``plantype`` system. These are:
    * ``__new__``
    * ``__setattr__``
    * ``__getattr__``
    * ``__dir__``
    """
    __slots__ = ()
    def __str__(self):
        pd = object.__getattribute__(self, '_plandict_')
        p = pd.plan
        param_str = ", ".join(
            f"{k}={pd[k] if pd.is_ready(k) else '<lazy>'}"
            for k in p.inputs)
        rest_str = ", ".join(
            f"{k}={pd[k] if pd.is_ready(k) else '<lazy>'}"
            for k in p.outputs if not k.startswith('_'))
        cls = type(self)
        if rest_str:
            return f"{cls.__name__}({param_str}; {rest_str})"
        else:
            return f"{cls.__name__}({param_str})"
    def __repr__(self):
        pd = object.__getattribute__(self, '_plandict_')
        p = pd.plan
        param_str = ", ".join(
            f"{k}={pd[k]}" for k in p.inputs)
        rest_str = ", ".join(
            f"{k}={repr(pd.getlazy(k))}" for k in p.outputs)
        cls = type(self)
        if rest_str:
            return f"{cls.__module__}.{cls.__name__}({param_str}; {rest_str})"
        else:
            return f"{cls.__module__}.{cls.__name__}({param_str})"
    def __eq__(self, other):
        if type(self) is not type(other):
            return False
        return self._plandict_.inputs == other._plandict_.inputs
    def __ne__(self, other):
        if type(self) is not type(other):
            return True
        return self._plandict_.inputs != other._plandict_.inputs
    def __hash__(self):
        return hash((type(self), self._plandict_.inputs))
    def copy(self, **kwargs):
        """Creates a copy of the planobject with optional parameter updates.
        """
        pd = plandict(self._plandict_, **kwargs)
        obj = object.__new__(self.__class__)
        object.__setattr__(obj, '_plandict_', pd)
        return obj
    def transient(self):
        """Returns a transient copy of the planobject."""
        pd = self._plandict_
        if pd is None or isinstance(pd, dict):
            raise RuntimeError(
                f"cannot create transient copy during initialization")
        c = copy(self)
        object.__setattr__(c, '_plandict_', tplandict(pd))
        return c
    def persistent(self):
        """Returns a persistent copy of the planobject. If the planobject is
        already persistent, it is returned unchanged.
        """
        pd = self._plandict_
        if pd is None or isinstance(pd, dict):
            raise RuntimeError(
                f"cannot create persistent copy during initialization")
        elif isinstance(pd, plandict):
            # Already persistent.
            return self
        elif not isinstance(pd, tplandict):
            raise ValueError(f"unknown plandict type: {type(pd)}")
        c = copy(self)
        object.__setattr__(c, '_plandict_', pd.persistent())
        return c
    def is_persistent(self):
        """Returns ``True`` if the planobject is persistent and ``False`` if it
        is transient.
        """
        pd = self._plandict_
        if pd is None or isinstance(pd, dict):
            return type(self).init_transient
        else:
            return isinstance(pd, plandict)


# Utilities ###################################################################

@docwrap
def is_planobject(obj):
    '''Determines if an object is an instance of a ``immlib.plantype`` object.
    
    ``is_planobject(obj)`` returns ``True`` if ``obj`` is an instance of a
    ``immlib.plantype`` class and ``False`` otherwise.

    See also: ``plantype``, ``is_plantype``
    '''
    return isinstance(obj, planobject)
@docwrap
def is_plantype(obj):
    '''Determines if an object is a ``immlib.plantype``.
    
    ``is_plantype(obj)`` returns ``True`` if ``obj`` is a ``immlib``
    ``plantype`` class and ``False`` otherwise. Note that this works for the
    type but not instances of the type, for which you should use
    ``is_planobject``.

    See also: ``is_planobject``, ``plantype``
    '''
    return isinstance(obj, plantype)
