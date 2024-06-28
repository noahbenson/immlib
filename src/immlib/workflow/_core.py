# -*- coding: utf-8 -*-
################################################################################
# immlib/workflow/_core.py


# Dependencies #################################################################

import copy
from collections.abc import Callable
from collections import (defaultdict, namedtuple)
from functools import (reduce, lru_cache, wraps)
from inspect import getfullargspec

import numpy as np

from pcollections import (pdict, ldict, lazy, pset, plist)

from ..doc import (docwrap, docproc, make_docproc)
from ..util import (
    is_pdict, is_str, is_number, is_tuple, is_dict, is_ldict,
    is_array, is_integer, strisvar, is_amap, merge, valmap)


# Utility Functions ############################################################

@docwrap
def to_pathcache(obj):
    """Returns a joblib.Memory object that corresponds to the given path object.

    `to_pathcache(obj)` converts the given object `obj` into a `joblib.Memory`
    cache manager. The object may be any of the following:
     * a `joblib.Memory` object;
     * a filename or pathlib object pointing to a directory; or
     * a tuple containing a filename or pathlib object followed by a dict-like
       object of options to `joblib.Memory`.

    If the `obj` is `None`, then `None` is returned. However, a `joblib.Memory`
    object whose location parameter is `None` can be created by using the
    object `(None, opts)` where `opts` may be `None` or an empty dict.
    """
    from joblib import Memory
    from pathlib import Path
    # If we have been given a Memory object, just return it.
    if isinstance(obj, Memory): return obj
    # Otherwise, check if we have been given options first.
    if is_tuple(obj):
        if   len(obj) == 1: (obj,opts) = (obj[0], {})
        elif len(obj) == 2: (obj,opts) = obj
        else: raise ValueError("only 1- or 2-tuples can become memcaches")
        if opts is None: opts = {}
    else:
        opts = {}
    # Whether there were or were not any options, then we now have either a
    # string or pathlib path that we want to pass to the memory constructor.
    if isinstance(obj, Path) or isinstance(obj, str) or obj is None:
        return Memory(obj, **opts)
    else:
        raise TypeError(f"location must be path, str, or None")
@docwrap
def to_lrucache(obj):
    """Returns an lru_cache function appropriate for the given object.

    `to_lrucache(obj)` converts the given object `obj` into either `None`, the
    `lru_cache` function, or a function returned by `lru_cache`. The object may
    be any of the following:
     * `lru_cache` itself, in which case it is just returned;
     * `None` or 0, indicating that no caching should be used (`None` is
        returned in these cases);
     * `inf`, indicating that an infinite cache should be returned; or
     * a positive integer indicating the number of most recently used items
       to keep in the cache.
    """
    if   obj is lru_cache: return obj
    elif obj is None: return None
    elif is_number(obj):
        if   obj == 0: return None
        elif obj == np.inf: return lru_cache(maxsize=None)
        elif not is_integer(obj): raise TypeError("lru_cache size must be int")
        elif obj < 1: raise ValueError("lrucache maxsize must be > 0")
        else: return lru_cache(maxsize=obj)
    elif callable(obj): return obj
    else: raise TypeError(f"bad type for to_lrucache: {type(obj)}")


################################################################################
# The calc, plan, and plandict classes

#TODO: make cache_memory and cache_path work with plans:
# if cache_path=True or cache_memory=True, then that caching is done at the plan
# level, with the plan's cache_path giving the calc a cache location, and the
# plan's cache_memory giving the calc a lru_cache if requested.

# calc #########################################################################
class calc:
    '''Class that represents a single calculation in a calc-plan.
    
    The `calc` class encapsulates data regarding the calculation of a single set
    of data from a separate set of input data. The input parameters are
    sometimes referred to as afferent values and the output variables are
    sometimes referred to as efferent values.

    `@calc` is typically used as a decorator that indicates that the function
    that follows is a calculation component; calculation components can be
    combined to form `plan` objects, which can encapsulate a flexible workflow
    of simple Python computations. When `@calc` is used as a decorator by
    itself, then the calc is considered to have a single output value whose name
    is the same as that of the function it decorates.
    
    `@calc(names...)` accepts a string or strings that name the output values of
    the calc function. In this case, the decorated function must return either a
    tuple of thes values in the order they are given or a dictionary in which
    the keys are the same as the given names.
    
    `@calc(None)` is a special instance in which the lazy argument is ignored
    (it is forced to be `False`), no output values are expected, and the
    calculation is always run when the input parameters are updated.
    
    The `calc` class starts by runnings its input through the `immlib.docwrap`
    function in order to gather input information about it. The `'Inputs'` and
    `'Outputs'` sections are tracked as documentation of the parameters and the
    data that are produced. Users of calculation objects should run their
    functions through `docwrap` manually themselves, however (if desired),
    because the docwrap used by `calc` does not put function documentation in
    the immlib namespace.

    Parameters
    ----------
    fn : callable
        The function that performs the calculation.
    outputs : tuple-like of strings
        A list or tuple or the names of the output variables. The names must all
        be valid variable names (see `strisvar`).
    name : None or str, optional
        The name of the function. The default, `None`, uses `fn.__name__`.
    lazy : boolean, optional
        Whether the calculation unit should be calculated lazily (`True`) or
        eagerly (`False`) when a plandict is created. The default is `True`.
    cache_memory : int, optional
        The number of recently calculated results to cache. If this value is 0,
        then no memoization is done (the default). If `cache_memory` is an
        integer greater than 0, then an LRU cache is used with a maxsize of
        `cache_memory`. If `cache_memory` is `inf`, then all values are cached
        indefinitely. Note that this cache is performed at the level of the
        calculation using Python's `functools` caching decorators.
    cache_path : None or directory-name, optional
        If `cache_path` is not `None` (the default), then cached results are
        also cached to the given directory when possible. The `cache_path`
        option may also be either a `pathlib.Path` object or a 2-tuple
        containing a path followed by options to the `joblib.Memory`
        constructor; see `to_pathcache` for more information.
    indent : int, optional
        The indentation level of the function's docstring. The default is 4.

    Attributes
    ----------
    name : str
        The name of the calculation function.
    base_function : callable
        The original function, prior to decoration for caching.
    cache_memory : None or lru_cache-like function
        Either `None`, indicating that no in-memory cache is being used by the
        calculation directly, or the function used to wrap the `base_function`
        for caching.
    cache_path : None or joblib.Memory object
        Either `None`, indicating that no filesystem cache is being used by the
        calculation directly, or the `joblib.Memory` object that handles the
        caching for the calculation.
    function : callable
        The function itself.
    argspec : FullArgSpec
        The argspec of `fn`, as returned from `inspect.getfullargspec(fn)`. This
        may not precisely match the argspec of `function` at any given time, but
        it remains correct as far as the calculation object requires (changes
        are due to translation calls). The `argspec` also differs from a true
        `argspec` object in that its members are all persistent objects such as
        `tuple`s instead of `list`s.
    inputs : pset of strs
        The names of the input parameters for the calculation.
    outputs : tuple of str
        The names of the output values of the calculation.
    defaults : mapping
        A persistent dictionary whose keys are input parameter names and whose
        values are the default values for the associated parameters.
    lazy : boolean
        Whether the calculation is intended as a lazy (`True`) or eager
        (`False`) calculation.
    input_docs : mapping
        A pdict object whose keys are input names and whose values are the
        documentation for the associated input parameters.
    output_docs : mapping
        A pdict object whose keys are output names and whose values are
        the documentation for the associated output values.

    '''
    __slots__ = ('name', 'base_function', 'cache_memory', 'cache_path',
                 'function', 'argspec', 'inputs', 'outputs', 'defaults', 'lazy',
                 'input_docs', 'output_docs')
    @staticmethod
    def _dict_persist(arg):
        if arg is None: return arg
        else: return pdict(arg)
    @staticmethod
    def _argspec_persist(spec):
        from inspect import FullArgSpec
        return FullArgSpec(
            args=tuple(spec.args),
            varargs=spec.varargs,
            varkw=spec.varkw,
            defaults=spec.defaults,
            kwonlyargs=tuple(spec.kwonlyargs),
            kwonlydefaults=calc._dict_persist(spec.kwonlydefaults),
            annotations=calc._dict_persist(spec.annotations))
    @staticmethod
    def _apply_caching(base_fn, cache_memory, cache_path):
        # We assume that cache and cache_path have already been appropriately
        # filtered by the to_lrucache and to_pathcache functions.
        if cache_path is None:
            fn = base_fn
        else:
            fn = cache_path.cache(base_fn)
        if cache_memory is not None:
            fn = cache_memory(fn)
        return fn
    @classmethod
    def _new(cls, fn, outputs,
             name=None, lazy=True, indent=None,
             cache_memory=0, cache_path=None):
        # Check the name.
        if name is None:
            name = fn.__module__ + '.' + fn.__name__
        # Okay, let's run the fn through docwrap to get the input and output
        # documentation.
        if (hasattr(fn, '__doc__') and
            fn.__doc__ is not None and fn.__doc__.strip() != '' and
            name is not None):
            fndoc = fn.__doc__
            dp = make_docproc()
            fn = docwrap('fn', indent=indent, proc=dp)(fn)
            input_docs  = {k[10:]: doc
                           for (k,doc) in dp.params.items()
                           if k.startswith('fn.inputs.')}
            output_docs = {k[11:]: doc
                           for (k,doc) in dp.params.items()
                           if k.startswith('fn.outputs.')}
            input_docs  = pdict(input_docs)
            output_docs = pdict(output_docs)
        else:
            input_docs = pdict()
            output_docs = pdict()
            fndoc = None
        # We make a new class that is a subtype of calc and that runs this
        # specific function when called. This lets us update the documentation.
        class LambdaClass(cls):
            @wraps(fn)
            def __call__(self, *args, **kw):
                return cls.__call__(self, *args, **kw)
        LambdaClass.__doc__ = fndoc
        # Go ahead and allocate the object we're creating.
        self = object.__new__(LambdaClass)
        # Set some attributes.
        object.__setattr__(self, 'name', name)
        # Save the base_function before we do anything to it.
        object.__setattr__(self, 'base_function', fn)
        # If there's a caching strategy here, use it.
        cache_memory = to_lrucache(cache_memory)
        object.__setattr__(self, 'cache_memory', cache_memory)
        # If there's a cache path, note it.
        cache_path = to_pathcache(cache_path)
        object.__setattr__(self, 'cache_path', cache_path)
        # Now save the function.
        cache_fn = calc._apply_caching(fn, cache_memory, cache_path)
        object.__setattr__(self, 'function', cache_fn)
        # Get the argspec for the calculation function.
        spec = getfullargspec(fn)
        if spec.varargs is not None:
            raise ValueError("calculations do not support varargs")
        if spec.varkw is not None:
            raise ValueError("calculations do not support varkw")
        # Save this for future use.
        spec = calc._argspec_persist(spec)
        object.__setattr__(self, 'argspec', spec)
        # Figure out the inputs from the argspec.
        inputs = pset(spec.args + spec.kwonlyargs)
        object.__setattr__(self, 'inputs', inputs)
        # Check that the outputs are okay.
        outputs = tuple(outputs)
        for out in outputs:
            if not strisvar(out):
                raise ValueError(f"calc output '{out}' is not a valid varname")
        object.__setattr__(self, 'outputs', outputs)
        # We need to grab the defaults also.
        dflts = {}
        for (arglst,argdfs) in [(spec.args, spec.defaults),
                                (spec.kwonlyargs, spec.kwonlydefaults)]:
            if not arglst or not argdfs: continue
            arglst = arglst[-len(argdfs):]
            dflts.update(zip(arglst, argdfs))
        object.__setattr__(self, 'defaults', pdict(dflts))
        # Save the laziness status and the documentations.
        object.__setattr__(self, 'lazy', bool(lazy))
        object.__setattr__(self, 'input_docs', input_docs)
        object.__setattr__(self, 'output_docs', output_docs)
        # That is all for the constructor.
        return self
    def __new__(cls, *args,
                name=None, lazy=True,
                cache_memory=0, cache_path=None, indent=None):
        kw = dict(name=name, lazy=lazy, cache_memory=cache_memory,
                  cache_path=cache_path, indent=indent)
        if len(args) == 0:
            # @calc(k1=v1...) :: calc(k1=v1...)(fn)
            # Special case where we are getting the output name from the
            # function's name directly.
            def calc_noarg(f):
                return cls._new(f, (f.__name__,), **kw)
            return calc_noarg
        elif len(args) == 1 and not is_str(args[0]):
            if args[0] is None:
                # @calc(None, k1=v1...) :: calc(None, k1=v1...)(fn)
                # Call to @calc(None), which forces a no-outputs version.
                def calc_none(f):
                    return cls._new(f, None, **kw)
                return cls_none
            else:
                # @calc :: calc(fn) or calc(fn, k1=v1...)
                # Call to @calc without arguments: use the function name.
                f = args[0]
                return cls._new(f, (f.__name__,), **kw)
        else:
            # @calc(out1..., k1=v1...) :: calc(out1..., k1=v1...)(fn)
            # We have been given a list of output variable names.
            def calc_outputs(f):
                return cls._new(f, args, **kw)
            return calc_outputs
    def eager_call(self, *args, **kwargs):
        """Eagerly calls the given calculation using the arguments.

        `c.eager_call(...)` returns the result of calling the calculation
        `c(...)` directly. Using the `eager_call` method is different from
        calling the `__call__` method only in that the `eager_call` method
        ignored the `lazy` member and always returns the direct results of
        calling the calculation; using the `__call__` method will result in
        `eager_call` being run if the calculation is not lazy and in `lazy_call`
        being run if the calculation is lazy.

        See also `calc.eager_mapcall`, `calc.lazy_call`, and
        `calc.lazy_mapcall`.
        """
        # The function is being called; we just pass this along (the function
        # itself has been given the caching code via decorators already).
        res = self.function(*args, **kwargs)
        # Now interpret the result.
        outs = self.outputs
        if not outs:
            # We ignore the output and just return an empty lazydict in this
            # case.
            return ldict({})
        n = len(outs)
        if is_amap(res) and len(res) == n and all(k in res for k in outs):
            pass
        elif is_tuple(res) and len(res) == n:
            res = {k:v for (k,v) in zip(outs, res)}
        elif len(self.outputs) == 1:
            res = {outs[0]: res}
        elif not self.outputs and not res:
            res = {}
        else:
            raise ValueError(f'return value from function call ({self.name}):'
                             ' did not match efferents')
        # We always convert lazys into values by returning a lazydict.
        return ldict(res)
    def lazy_call(self, *args, **kwargs):
        """Returns a lazy-dict of the results of calling the calculation.

        `calc.lazy_call(...)` is equivalent to `calc(...)` except that the
        `lazydict` that it returns encapsulates the running of the calculation
        itself, so that `calc(...)` is not run until one of the lazy values is
        requested.

        See also `calc.mapcall` annd `calc.lazy_mapcall`.
        """
        # First, create a lazy for the actual call:
        calldel = lazy(self.eager_call, *args, **kwargs)
        # Then make a lazy map of all the outputs, each of which pulls from this
        # delay object to get its values.
        return ldict({k: lazy(lambda k: calldel()[k], k)
                      for k in self.outputs})
    def __call__(self, *args, **kwargs):
        if self.lazy: return self.lazy_call(*args, **kwargs)
        else:         return self.eager_call(*args, **kwargs)
    def call(self, *args, **kwargs):
        """Calls the calculation and returns the results dictionary.

        `c.call(...)` is an alias for `c(...)`.

        See also `calc.mapcall`, `calc.eager_call`, and `calc.lazy_call`.
        """
        if self.lazy: return self.lazy_call(*args, **kwargs)
        else:         return self.eager_call(*args, **kwargs)
    def _maps_to_args(self, args, kwargs):
        opts = merge(self.defaults, *args, **kwargs)
        args = []
        kwargs = {}
        miss = False
        for name in self.argspec.args:
            if name not in opts:
                miss = True
                continue
            if miss:
                kwargs[name] = opts[name]
            else:
                args.append(opts[name])
        for name in self.argspec.kwonlyargs:
            if name in opts:
                kwargs[name] = opts[name]
        return (args, kwargs)
    def eager_mapcall(self, *args, **kwargs):
        """Calls the given calculation using the parameters in mappings.

        `c.eager_mapcall(map1, map2..., key1=val1, key2=val2...)` returns the
        result of calling the calculation `c(...)` using the parameters found in
        the provided mappings and key-value pairs. All arguments of `mapcall`
        are merged left-to-right using `immlib.merge` then passed to
        `c.function` as required by it.
        """
        (args, kwargs) = self._maps_to_args(args, kwargs)
        return self.eager_call(*args, **kwargs)
    def lazy_mapcall(self, *args, **kwargs):
        """Calls the given calculation lazily using the parameters in mappings.

        `c.lazy_mapcall(map1, map2..., key1=val1, key2=val2...)` returns the
        result of calling the calculation `c(...)` using the parameters found in
        the provided mappings and key-value pairs. All arguments of `mapcall`
        are merged left-to-right using `immlib.merge` then passed to
        `c.function` as required by it.

        The only difference between `calc.mapcall` and `calc.lazy_mapcall` is
        that the lazydict returned by the latter method encapsulates the calling
        of the calculation itself, so no call to the calculation is made until
        one of the values of the lazydict is requested.

        See also `calc.eager_mapcall`, `calc.lazy_call`, and `calc.eager_call`.
        """
        # Note that all the args must be dictionaries, so we make copies of them
        # if they're not persistent dictionaries. This prevents later
        # modifications from affecting the results downstream.
        args = [d if is_pdict(d) else dict(d) for d in args]
        # First, create a lazy for the actual call:
        calldel = lazy(self.eager_mapcall, *args, **kwargs)
        # Then make a lazy map of all the outputs, each of which pulls from this
        # lazy object to get its values.
        return ldict({k: lazy(lambda k: calldel()[k], k)
                      for k in self.outputs})
    def mapcall(self, *args, **kwargs):
        """Calls the calculation and returns the results dictionary.

        `c.mapcall(map1, map2..., key1=val1, key2=val2...)` returns the result
        of calling the calculation `c(...)` using the parameters found in the
        provided mappings and key-value pairs. All arguments of `mapcall` are
        merged left-to-right using `immlib.merge` then passed to `c.function` as
        required by it.

        See also `calc.lazy_mapcall`, `calc.eager_mapcall`, and `calc.call`.
        """
        if self.lazy: return self.lazy_mapcall(*args, **kwargs)
        else:         return self.eager_mapcall(*args, **kwargs)
    def __setattr__(self, k, v):
        raise TypeError('calc objects are immutable')
    def __delattr__(self, k):
        raise TypeError('calc objects are immutable')
    @staticmethod
    def _tr_map(tr, m):
        if m is None: return None
        is_ld = is_ldict(m)
        it = (m.to_pdict() if is_ld else m).items()
        d = {tr.get(k,k): v for (k,v) in it}
        if is_ld: return ldict(d)
        else: return pdict(d)
    @staticmethod
    def _tr_tup(tr, t):
        return None if t is None else tuple(tr.get(k,k) for k in t)
    @staticmethod
    def _tr_set(tr, t):
        return None if t is None else pset(tr.get(k,k) for k in t)
    def tr(self, *args, **kwargs):
        """Returns a copy of the calculation with translated inputs and outputs.
        
        `calc.tr(...)` returns a copy of `calc` in which the input and output
        values of the function have been translated. The translation is found
        from merging the list of 0 or more dict-like arguments given
        left-to-right followed by the keyword arguments.
        """
        d = merge(*args, **kwargs)
        # The reversed version of d.
        r = {v:k for (k,v) in d.items()}
        # Make a copy.
        tr = object.__new__(calc)
        # Simple changes first.
        object.__setattr__(tr, 'name', self.name + f'.tr{hex(id(tr))}')
        object.__setattr__(tr, 'base_function', self.base_function)
        object.__setattr__(tr, 'cache_memory', self.cache_memory)
        object.__setattr__(tr, 'cache_path', self.cache_path)
        object.__setattr__(tr, 'argspec', self.argspec)
        object.__setattr__(tr, 'inputs', calc._tr_set(d, self.inputs))
        object.__setattr__(tr, 'outputs', calc._tr_tup(d, self.outputs))
        object.__setattr__(tr, 'defaults', calc._tr_map(d, self.defaults))
        object.__setattr__(tr, 'lazy', self.lazy)
        object.__setattr__(tr, 'input_docs', calc._tr_map(d, self.input_docs))
        object.__setattr__(tr, 'output_docs', calc._tr_map(d, self.output_docs))
        # Translate the argspec.
        from inspect import FullArgSpec
        spec = self.argspec
        spec = FullArgSpec(
            args=calc._tr_tup(d, spec.args),
            varargs=None, varkw=None,
            defaults=calc._tr_tup(d, spec.defaults),
            kwonlyargs=calc._tr_tup(d, spec.kwonlyargs),
            kwonlydefaults=calc._tr_map(d, spec.kwonlydefaults),
            annotations=calc._tr_map(d, spec.annotations))
        object.__setattr__(tr, 'argspec', spec)
        # The function also needs a wrapper.
        from functools import wraps
        fn = self.function
        def _tr_fn_wrapper(*args, **kwargs):
            # We may need to untranslate some of the keys.
            kwargs = {r.get(k,k):v for (k,v) in kwargs.items()}
            res = fn(*args, **kwargs)
            if is_amap(res):
                return calc._tr_map(d, res)
            else:
                return res
        object.__setattr__(tr, 'function', wraps(fn)(_tr_fn_wrapper))
        return tr
    def with_cache_memory(self, new_cache):
        "Returns a copy of a calc with a different in-memory cache strategy."
        new_cache = to_lrucache(new_cache)
        new_calc = copy.copy(self)
        object.__setattr__(new_calc, 'cache_memory', new_cache)
        new_fn = calc._apply_caching(self.base_function, new_cache,
                                     self.cache_path)
        object.__setattr__(new_calc, 'function', new_fn)
        return new_calc
    def with_cache_path(self, new_path):
        """Returns a copy of a calc with a different cache directory."""
        new_cache = to_pathcache(new_path)
        new_calc = copy.copy(self)
        object.__setattr__(new_calc, 'cache_path', new_cache)
        new_fn = calc._apply_caching(self.base_function,
                                     self.cache_memory, new_cache)
        object.__setattr__(new_calc, 'function', new_fn)
        return new_calc
    def is_filter(self):
        """Determines if the given `calc` object is a valid filter.

        A `calc` is a valid filter if it has exactly one input and one output
        both of which are the same.
        """
        return (len(self.inputs) == 1
                and next(iter(self.inputs)) == self.outputs[0])
@docwrap
def is_calc(arg):
    """Determines if an object is a `calc` instance.

    `is_calc(x)` returns `True` if `x` is a function that was decorated with an
    `@calc` directive and `Falseq otherwise.
    """
    return isinstance(arg, calc)


# plan #########################################################################

class plan(pdict):
    '''Represents a directed acyclic graph of calculations.
    
    The `plan` class encapsulates individual functions that require parameters
    as inputs and produce outputs in the form of named values. Plan objects can
    be called as functions with a dictionary and/or a list of keyword parameters
    specifying their parameters; they always return a dictionary of the values
    they calculate, even if they calculate only a single value.

    Superficially, a `plan` is a `pdict` object whose values must all be `calc`
    objects. However, under the hood, every `plan` object maintains a directed
    acyclic graph of dependencies of the inputs and outputs of the calculation
    objects such that it can create `plandict` objects that reify the outputs of
    the various calculations lazily.

    The keys that are used in a plan must be strings and must obey the following
    rules:
      * Any key that begins with `'filter_'` must end with the name of an input
        parameter to the plan, and its value must a function of one parameter or
        a calc object whose only output is the value in question. Such an entry
        is treated as a filter for the relevant input parameter.
      * Any key that begins with `'default_'` must end with the name of an input
        parameter to the plan, and its value is taken to be the default value of
        the given parameter if no other default is found in the plan and if no
        explicit value is provided.
    Keys are not otherwise restricted, but `'calc_'` is suggested as a prefix
    for standard calculations.

    For a plan `p = plan(calc_key1=calc1, calc_key2=calc2, ...)`, a `plandict`
    can be instantiated using the following syntax:
      `pd = p(param1=val1, param2=val2, ...)`.
    This `plandict` is an enhanced `lazydict` that evaluates components of the
    plan as requested based on laziness requirements of the calculations in the
    plan and on dictionary lookups of plan outputs.

    Attributes
    ----------
    inputs : pset of strs
        A pset of the input parameter names, as defined by the plan's
        calculations.
    outputs : pset of strs
        A pset of the output parameter names, as defined by the plan's
        calculations.
    calcs : tuple of str
        A tuple of the caclulation names of the normal (non-filter and
        non-default) calculations in the plan.
    defaults : pdict
        A pdict whose keys are input parameter names and whose values are
        the default values of the associated parameter. Inputs that don't have
        default values are not included.
    calc_defaults : pdict
        A pdict similar to `defaults` but limited to the defaults listed
        in the `defaults` of the calculations (i.e., the default values in the
        argument lists of the calculations). When two calculations at the same
        layer in the plan (see `plan.layers`) have different default values for
        the same input parameter, then one of them is chosen, but which is not
        defined.
    dependencies : pdict
        A pdict whose keys are each of the output value names from the
        plan's calculations and whose values are tuples of the input value names
        that the associated output value depends on.
    dependants : pdict
        A pdict whose keys are each of the input value names from the
        plan's calculations and whose values are tuples of the output value
        names that depend on the associated input value.
    filters : pdict
        A pdict whose keys are the input value names of all inputs that
        have defined filters in the plan and whose values are the filter calc
        for the associated input.
    requirements : pdict
        A pdict whose keys are the required calculations of the plan and
        whose values are tuples of the names of the input parameters they
        require.
    layers : tuple of plan.Layer tuples
        A tuple of layers in the calculation. Each layer is a named tuple of
        type `plan.Layer`, which has fields `('inputs', 'calcs', 'outputs')`.
        The `inputs` are input parameters available to calculations at that
        layer of the plan; the `calcs` are the calculations that can run using
        those inputs; and the `outputs` are the output values that are newly
        available after that layer of the plan is calculated.
    input_docs : pdict
        A dictionary whose keys are input parameter names and whose values are
        the combined documentation for the associated parameter across all
        calculations in the plan.
    output_docs : pdict
        A dictionary whose keys are output value names and whose values are the
        combined documentation for the associated outputs across all
        calculations in the plan.

    '''
    Layer = namedtuple('PlanLayer', ('inputs', 'calcs', 'outputs'))
    __slots__ = (
        'inputs', 'outputs', 'defaults', 'calc_defaults',
        'dependencies', 'dependants',
        'calc_dependencies', 'dependant_calcs', 'calc_sources',
        'calcs', 'filters', 'requirements', 'layers',
        'input_docs', 'output_docs')
    @staticmethod
    def _trclosure(edges):
        clos = set((k,u) for (k,v) in edges.items() for u in v)
        while True:
            s = set((u1,v2) for (u1,v1) in clos for (u2,v2) in clos if v2 == u1)
            if clos.issuperset(s): break
            clos |= s
        res = defaultdict(lambda:set([]))
        for (u,v) in clos:
            res[u].add(v)
        return res
    @staticmethod
    def _tc_params_to_rparams(tc_params):
        tc_rparams = defaultdict(lambda:[])
        for (k,v) in tc_params.items():
            for u in v:
                tc_rparams[u].append(k)
        tc_rparams = {k: tuple(v) for (k,v) in tc_rparams.items()}
        return tc_rparams
    # Constructor
    def __init__(self, *args, **kwargs):
        # We ignore the arguments because they are handled by pdict's __new__
        # method.  We need to start by building up the graph of dependencies.
        deps = defaultdict(lambda:set([]))
        inputs = set([])
        outputs = set([])
        filts = {}
        reqs = set([])
        defaults = {}
        # calc_src: value name |-> source calc
        calc_src = {}
        # We also start building up the layers here (this is mostly done below;
        # see comments a few paragraphs down for more info).
        normal_calcs = set([])
        noinput_calcs = set([])
        noinput_calc_outputs = set([])
        for (k,v) in self.items():
            if k.startswith('filter_'):
                nm = k[7:]
                if not is_calc(v):
                    v = calc(nm)(v)
                if not v.is_filter():
                    raise ValueError(f"{k} calc is not a valid filter")
                filts[nm] = v
                inputs.add(nm)
                continue
            elif k.startswith('default_'):
                nm = k[8:]
                defaults[nm] = v
                continue
            elif not is_calc(v):
                raise TypeError("plans require calculation objects as values")
            # If we have reached this point, then v is a calc that is part of
            # the calc plan (not a filter or default value).
            if not v.lazy:
                reqs.add(k)
            # Note the edge-data for each input/output.
            if v.outputs:
                for kk in v.outputs:
                    if kk in calc_src:
                        raise ValueError(f"two calcs output value {kk}:"
                                         f" {k} and {calc_src[kk]}")
                    else:
                        calc_src[kk] = k
                outputs.update(v.outputs)
            if v.inputs:
                inputs.update(v.inputs)
                # All of the calc's outputs depend on its inputs.
                if v.outputs:
                    for output in v.outputs:
                        deps[output].update(v.inputs)
                # We note that this is a normal calc.
                normal_calcs.add(k)
            else:
                # This calculation has no inputs, so it's part of layer 0. We
                # convert this set of layer-0 calcs into a proper layer
                # structure in the section on layers, below.
                noinput_calcs.add(k)
                if v.outputs:
                    noinput_calc_outputs.update(v.outputs)
        # Now that we have all inputs and all outputs, we can find the input
        # parameters by removing the outputs from the inputs.
        params = inputs - outputs
        # Next, we want to make the layers of the calculation. Layers represent
        # the required order of execution of the calculation plan. Each layer is
        # a tuple of three psets: (input_names, calc_names,
        # new_output_names). The calc_names are the names of the calculations
        # that require input from only the layers above this layer while the
        # new_output_names are the values that are available after this layer of
        # calculations has been run. The first layer, layers[0], is special in
        # that it is always (pset([]), noinput_calcs, inputs |
        # noinput_calc_outputs) where noinput_calcs are the calculations that
        # don't require any input parameters, and noinput_calc_outputs are the
        # outputs of those calcs.
        params_sofar = params | noinput_calc_outputs
        params = pset(params)
        noinput_calcs = pset(noinput_calcs)
        calc_defaults = {}
        layers = [plan.Layer(pset([]),
                             noinput_calcs,
                             pset(params_sofar))]
        allcalcs = noinput_calcs | normal_calcs
        while len(normal_calcs) > 0:
            (calcs, vals) = (set([]), set([]))
            # See which calcs can run now that we have this layer of values.
            for k in normal_calcs:
                calc = self[k]
                if params_sofar.issuperset(calc.inputs):
                    calcs.add(k)
                    if calc.outputs:
                        vals.update(calc.outputs)
            if len(vals) == 0:
                raise ValueError("calculation graph contains unreachable nodes")
            if not params_sofar.isdisjoint(vals):
                isect = tuple(params_sofar.intersection(vals))
                raise ValueError(f"dependency loop detected: {isect}")
            layer = plan.Layer(pset(params_sofar),
                               pset(calcs),
                               pset(vals))
            layers.append(layer)
            # We have some postprocessing to do on the calcs that were added to
            # this layer.
            for k in calcs:
                # If this calc has defaults that aren't already in the defaults
                # dict for calcs, add them in.
                c = self[k]
                dflts = {} if c.defaults is None else c.defaults
                for (kk,dflt) in dflts.items():
                    calc_defaults.setdefault(kk, dflt)
                # Remove these calculations from the normal calcs dict.
                normal_calcs.remove(k)
            # Now we go on to look for calculations that depend on only the
            # input params so far.
            params_sofar |= vals
        # Once we've gone through the calculations themselves, we can find the
        # transitive closure of the dependency graph.
        tc = plan._trclosure(deps)
        # Ultimately, what we care about are the params in the transitive
        # closure.
        tc_params = {k: tuple(v & params) for (k,v) in tc.items()}
        tc_rparams = plan._tc_params_to_rparams(tc_params)
        # Make the same tc_params and tc_rparams for the calcs.
        calctc_params = defaultdict(lambda:set([]))
        for k in allcalcs:
            c = self[k]
            cdeps = set([])
            if c.inputs:
                for inp in c.inputs:
                    cdeps.update(tc_params.get(inp, ()))
            calctc_params[k] = pset(cdeps)
        calctc_rparams = plan._tc_params_to_rparams(calctc_params)
        # Finally, collect all the input and output documentations.
        input_docs = {k: [] for k in params} # Just the params, not all inputs
        output_docs = {k: [] for k in outputs}
        # For the params, start with their filters.
        for (k,filt) in filts.items():
            if k not in params:
                msg = f"filter for {k}, which is not in params: {params}"
                raise ValueError(msg)
            try: doc = filt.__doc__
            except Exception: doc = None
            if doc:
                input_docs[k].append("filter: " + filt.name + "\n" + doc)
        # Now go through the layers and append the documentation to each.
        for layer in layers:
            for k in layer.calcs:
                c = self[k]
                for (inp,doc) in c.input_docs.items():
                    if not doc: continue
                    s = f"{k} input: {inp}"
                    if len(s) > 80: s = s[77] + '...'
                    if inp in params: input_docs[inp].append(s + '\n' + doc)
                    else:             output_docs[inp].append(s + '\n' + doc)
                for (out,doc) in c.output_docs.items():
                    if not doc: continue
                    s = f"{k} output: {out}"
                    if len(s) > 80: s = s[77] + '...'
                    output_docs[out].append(s + '\n' + doc)
        connectfn = lambda v: '\n---\n'.join(v)
        input_docs = pdict(valmap(connectfn, input_docs))
        output_docs = pdict(valmap(connectfn, output_docs))
        # Okay, set everything in the object.
        object.__setattr__(self, 'inputs', pset(params))
        object.__setattr__(self, 'outputs', pset(outputs))
        object.__setattr__(self, 'defaults', merge(calc_defaults, defaults))
        object.__setattr__(self, 'calc_defaults', pdict(calc_defaults))
        object.__setattr__(self, 'dependencies', pdict(tc_params))
        object.__setattr__(self, 'dependants', pdict(tc_rparams))
        object.__setattr__(self, 'calc_dependencies', pdict(calctc_params))
        object.__setattr__(self, 'dependant_calcs', pdict(calctc_rparams))
        object.__setattr__(self, 'calc_sources', pdict(calc_src))
        object.__setattr__(self, 'calcs', tuple(allcalcs))
        object.__setattr__(self, 'filters', pdict(filts))
        object.__setattr__(self, 'requirements', pset(reqs))
        object.__setattr__(self, 'layers', tuple(layers))
        object.__setattr__(self, 'input_docs', input_docs)
        object.__setattr__(self, 'output_docs', output_docs)
        # That's it; we should be constructed now!
    def filtercall(self, *args, **kwargs):
        """Calls the plan object, but filters out args that aren't in the plan.

        `plan_obj.filtercall(dict1, dict2, ..., k1=v1, k2=v2, ...)` is
        equivalent to calling `plan_obj(dict1, dict2, ..., k1=v1, k2=v2, ...)`
        except that any keys in the argument list to `filtercall` that aren't
        in the parameter list of `plan_obj` are automatically filtered out.
        """
        params = merge(*args, **kwargs)
        for k in params.keys():
            if k not in self.inputs:
                params = params.delete(k)
        return self.__call__(params)
    def __call__(self, *args, **kwargs):
        # Make and return a plandict with these parameters.
        return plandict(self, *args, **kwargs)
    def __str__(self):
        return f"plan(<{len(self.calcs)} calcs>, <{len(self.inputs)} params>)"
    def __repr__(self):
        return f"plan(<{len(self.calcs)} calcs>, <{len(self.inputs)} params>)"
@docwrap
def is_plan(arg):
    """Determines if an object is a `plan` instance.

    `is_plan(x)` returns `True` if `x` is a calculation `plan` and `False`
    otherwise.
    """
    return isinstance(arg, plan)


# #plandict ####################################################################

class plandict(ldict):
    """A persistent dict type that manages the outputs of executing a plan.

    `plandict(plan, params)` instantiates a plan object with the given dict-like
    object of parameters, `params`.

    `plandict(plan, params, k1=v1, k2=v2, ...)` additional merges all keyword
    arguments into parameters.

    `plandict(plan, k1=v1, k2=v2, ...)` uses only the keyword arguments as the
    plan parameters.

    Note that `plandict(plan, args...)` is equivalent to `plan(args...)`.

    `plandict` is a subclass of `lazydict`, but it has some unique behavior,
    primarily in that only the parameters of a `plandict` may be updated; the
    rest of the items are consequences of the plan and parameter.

    Parameters
    ----------
    plan : plan
        The `plan` object that is to be instantiated.
    params : dict-like, optional
        The dict-like object of the parameters of the `plan`. All and only
        `plan` parameters must be provided, after the `params` argument is
        merged with the `kwargs` options. This may be a `lazydict`, and this
        dict's laziness is respected as much as possible.
    kwargs : optional keywords
        Optional keywords that are merged into `params` to form the set of
        parameters for the plan.
    
    Attributes
    ----------
    plan : plan
        The plan object on which this plandict is based.
    inputs : pdict
        The parameters that fulfill the plan. Note that these are the only keys
        in the `plandict` that can be updated using methods like `set` and
        `setdefault`.
    calcs : pdict
        A `lazydict` whose keys are the calculation names from `plan` and whose
        values are the `lazydict`s that result from running the associated
        calculation with this `plandict`'s parameters.
    """
    __slots__ = ('plan', 'inputs', 'calcs')
    @staticmethod
    def _run_calc(plan, calc, mutvals):
        d = {}
        for k in calc.inputs:
            u = mutvals[k]
            if isinstance(u, lazy): u = u()
            d[k] = u
        return calc.eager_mapcall(d)
    @staticmethod
    def _run_filt(filt, params, k):
        res = filt.eager_mapcall({k: params[k]})
        return res[k]
    @staticmethod
    def _get_val(calcdicts, calcnm, output):
        return calcdicts[calcnm][output]
    def __new__(cls, *args, **kwargs):
        # There are two valid ways to call plandict(): plandict(planobj, params)
        # and plandict(plandictobj, new_params). We call different classmethods
        # for each version.
        if len(args) == 0:
            raise TypeError(
                "plandict() requires 1 argument that is a plan or plandict")
        (obj, args) = (args[0], args[1:])
        if is_plan(obj):
            return cls._new_from_plan(obj, *args, **kwargs)
        elif isinstance(obj, plandict):
            return cls._new_from_plandict(obj, *args, **kwargs)
        else:
            raise TypeError(
                "plandict(obj, ...) requires that obj be a plan or plandict")
    @classmethod
    def _new_from_plan(cls, plan, *args, **kwargs):
        # First, merge from left-to-right, respecting laziness. Then, run them
        # (lazily) through the filters.
        params = merge(plan.defaults, *args, **kwargs)
        if plan.filters:
            new_params = {k: (lazy(plandict._run_filt, filt, params, k)
                              if filt.lazy else
                              plandict._run_filt(filt, params, k))
                          for (k,filt) in plan.filters.items()}
            params = merge(params, ldict(new_params))
        # We must have all the parameters and only the parameters.
        if (len(params) != len(plan.inputs) or
            not all(k in params for k in plan.inputs)):
            (params, inp) = (set(params.keys()), set(plan.inputs))
            missing = inp - params
            if missing: raise ValueError(f"missing inputs: {tuple(missing)}")
            extras = params - inp
            raise ValueError(f"extra inputs: {tuple(extras)}")
        # We need to make a dict of the (lazy) calculations first. To do this,
        # we need to be able to pass the calcs the delays that we will make
        # after them, so we use a mutable dict as a hack.
        mut_values = {}
        calcs = ldict(
            {k: lazy(plandict._run_calc, plan, plan[k], mut_values)
             for k in plan.calcs})
        # Go ahead and run the 
        # The outputs come from these.
        outputs = {
            k: lazy(plandict._get_val, calcs, plan.calc_sources[k], k)
            for k in plan.outputs}
        outputs = ldict(outputs)
        # We now want to run all the required calculations.
        # We now have everything we need--go ahead and instantiate the lazydict.
        values = merge(params, outputs)
        values = values.to_pdict() if is_ldict(values) else values
        valitems = values.items()
        mut_values.update(valitems)
        self = super(plandict, cls).__new__(cls, valitems)
        # And set our special member-values.
        object.__setattr__(self, 'plan', plan)
        object.__setattr__(self, 'inputs', params)
        object.__setattr__(self, 'calcs', calcs)
        # Finally, now that we have the object entirely initialized, we can run
        # the required calculations.
        for r in plan.requirements:
            tmp = plan[r]
            for k in tmp.inputs:
                self[k]
            tmp = calcs[r]
            for k in tmp.keys():
                self[k]
        # That's all!
        return self
    @classmethod
    def _new_from_plandict(cls, pd, *args, **kwargs):
        plan = pd.plan
        # First, merge from left-to-right, respecting laziness. Then, run them
        # (lazily) through the filters.
        params = merge(plan.defaults, *args, **kwargs)
        if plan.filters:
            new_params = {
                k: (lazy(plandict._run_filt, filt, params, k)
                    if filt.lazy else
                    plandict._run_filt(filt, params, k))
                for (k,filt) in plan.filters.items()
                if k in params}
            params = merge(params, ldict(new_params))
        # There must only be parameters here.
        if not all(k in plan.inputs for k in params.keys()):
            (params, inp) = (set(params.keys()), set(plan.inputs))
            extras = params - inp
            raise ValueError(f"extra inputs: {tuple(extras)}")
        # Figure out all the dependant calcs and outputs of these params.
        depcalcs = set([])
        depouts = set([])
        for k in params.keys():
            depcalcs.update(plan.dependant_calcs[k])
            depouts.update(plan.dependants[k])
        # Like in _new_from_plan above, we need to make a dict of the (lazy)
        # calculations first, and we do so using a mutable dict as a hack.
        mut_values = {}
        new_calcs = {
            k: lazy(plandict._run_calc, plan, plan[k], mut_values)
            for k in depcalcs}
        new_calcs = ldict(new_calcs)
        new_outs = {
            k: lazy(
                plandict._get_val, new_calcs,
                plan.calc_sources[k], k)
            for k in depouts}
        new_outs = ldict(new_outs)
        params = merge(pd.params, new_params)
        values = merge(pd, new_params, new_outs)
        calcs = merge(pd.calcs, new_calcs)
        valitems = (values.to_pdict() if is_ldict(values) else values).items()
        mut_values.update(valitems)
        # We now have everything we need--go ahead and instantiate the lazydict.
        self = super(plandict, cls).__new__(cls, valitems)
        # And set our special member-values.
        object.__setattr__(self, 'plan', plan)
        object.__setattr__(self, 'inputs', params)
        object.__setattr__(self, 'calcs', calcs)
        # Finally, now that we have the object entirely initialized, we can
        # run the required calculations.
        for r in plan.requirements:
            # Just extract their lazydicts / forces evaluation.
            tmp = plan[r]
            for k in tmp.inputs:
                self[k]
            tmp = new_calcs.get(r, {})
            for k in tmp.keys():
                self[k]
        # That's all!
        return self
    def set(self, k, v):
        return plandict(self, {k:v})
    def setdefault(self, k, v=None):
        # All possible keys to set are already set in a plandict, so just pass
        # this through to set.
        return self.set(k, v)
    def delete(self, k):
        raise TypeError("cannot delete from a plandict")
@docwrap
def is_plandict(arg):
    """Determines if an object is a `plandict` instance.

    `is_plandict(x)` returns `True` if `x` is a `plandict` object and `False`
    otherwise.
    """
    return isinstance(arg, plandict)
