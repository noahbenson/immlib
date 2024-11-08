# -*- coding: utf-8 -*-
################################################################################
# immlib/workflow/_core.py


# Dependencies #################################################################

import copy
from collections.abc import Callable
from collections import (defaultdict, namedtuple)
from functools import (reduce, lru_cache, wraps, partial)
from inspect import getfullargspec

import numpy as np

from pcollections import (pdict, tdict, ldict, lazy, pset, tset, plist)

from ..doc import (docwrap, docproc, make_docproc)
from ..util import (
    is_pdict, is_str, is_number, is_tuple, is_dict, is_ldict,
    is_array, is_integer, strisvar, is_amap, merge, rmerge, valmap)


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
def identfn(x):
    "The identify function; `identfn(x)` returns `x`."
    return x

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
    __slots__ = (
        'name', 'base_function', 'cache_memory', 'cache_path',
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
    def _tr_map(tr, m, is_input):
        if m is None:
            return None
        tup_ii = int(not is_input)
        is_ld = is_ldict(m)
        it = (m.to_pdict() if is_ld else m).items()
        d = tdict()
        for (k,v) in it:
            kk = tr.get(k,k)
            if isinstance(kk, tuple):
                kk = kk[tup_ii]
            d[kk] = v
        return ldict(d) if is_ld else pdict(d)
    @staticmethod
    def _tr_tup(tr, t, is_input):
        if t is None:
            return None
        tup_ii = int(not is_input)
        res = []
        for k in t:
            k = tr.get(k,k)
            if isinstance(k, tuple):
                k = k[tup_ii]
            res.append(k)
        return tuple(res)
    @staticmethod
    def _tr_set(tr, t, is_input):
        if t is None:
            return None
        tup_ii = int(not is_input)
        res = tset()
        for k in t:
            k = tr.get(k, k)
            if isinstance(k, tuple):
                k = k[tup_ii]
            res.add(k)
        return res.persistent()
    def tr(self, *args, **kwargs):
        """Returns a copy of the calculation with translated inputs and outputs.
        
        `calc.tr(...)` returns a copy of `calc` in which the input and output
        values of the function have been translated. The translation is found
        from merging the list of 0 or more dict-like arguments given
        left-to-right followed by the keyword arguments into a single
        dictionary. The keys of this dictionary are translated into their
        associated values in the returned dictionary.

        If any of the values of the merged dictionary are 2-tuples, then they
        are interpreted as `(input_tr, output_tr)`. In this case, then the key
        must be associated with a name that appears in both the calculation's
        input list and its output list, and the two names are translated
        differently.
        """
        d = merge(*args, **kwargs)
        # Make a copy.
        tr = object.__new__(calc)
        # Simple changes first.
        object.__setattr__(tr, 'name', self.name + f'.tr{hex(id(tr))}')
        object.__setattr__(tr, 'base_function', self.base_function)
        object.__setattr__(tr, 'cache_memory', self.cache_memory)
        object.__setattr__(tr, 'cache_path', self.cache_path)
        object.__setattr__(tr, 'argspec', self.argspec)
        object.__setattr__(tr, 'inputs', calc._tr_set(d, self.inputs, True))
        object.__setattr__(tr, 'outputs', calc._tr_tup(d, self.outputs, False))
        object.__setattr__(tr, 'defaults', calc._tr_map(d, self.defaults, True))
        object.__setattr__(tr, 'lazy', self.lazy)
        object.__setattr__(
            tr, 'input_docs', calc._tr_map(d, self.input_docs, True))
        object.__setattr__(
            tr, 'output_docs', calc._tr_map(d, self.output_docs, False))
        # Translate the argspec.
        from inspect import FullArgSpec
        spec = self.argspec
        spec = FullArgSpec(
            args=calc._tr_tup(d, spec.args, True),
            varargs=None, varkw=None,
            defaults=calc._tr_tup(d, spec.defaults, True),
            kwonlyargs=calc._tr_tup(d, spec.kwonlyargs, True),
            kwonlydefaults=calc._tr_map(d, spec.kwonlydefaults, True),
            annotations=calc._tr_map(d, spec.annotations, True))
        object.__setattr__(tr, 'argspec', spec)
        # The function also needs a wrapper.
        from functools import wraps
        # The reversed version of d (for inputs).
        r = {v:(k if isinstance(k, str) else k[0]) for (k,v) in d.items()}
        fn = self.function
        def _tr_fn_wrapper(*args, **kwargs):
            # We may need to untranslate some of the keys.
            kwargs = {r.get(k,k):v for (k,v) in kwargs.items()}
            res = fn(*args, **kwargs)
            if is_amap(res):
                return calc._tr_map(d, res, False)
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
        calculations. Note that the union of the inputs and the outputs is
        equivalent to the keys in any plan-dictionary.
    outputs : pset of strs
        A pset of the output parameter names, as defined by the plan's
        calculations. Note that the union of the inputs and the outputs is
        equivalent to the keys in any plan-dictionary.
    defaults : pdict
        A dictionary whose keys consist of a subset of the inputs to the plan
        and whose values are the default values those parameters should take if
        they are not provided explicitly to the plan.
    calcs : pdict
        A persistent dictionary whose keys are the names of the various
        calculations in the plan and whose values are the calculation objects
        themselves.
    input_docs : pdict
        A dictionary whose keys are input parameter names and whose values are
        the combined documentation for the associated parameter across all
        calculations in the plan.
    output_docs : pdict
        A dictionary whose keys are output value names and whose values are the
        combined documentation for the associated outputs across all
        calculations in the plan.
    requirements : pset
        A `pset` of the names of the required calculations of the plan (i.e.,
        those with option `lazy=False`).
    '''
    # Subclasses ---------------------------------------------------------------
    CalcData = namedtuple(
        'CalcData',
        ('names', 'calcs', 'args', 'sources', 'index'))
    DepData = namedtuple(
        'DepData',
        ('inputs', 'calcs'))
    # Static Methods -----------------------------------------------------------
    @staticmethod
    def _filter_sort(kv):
        return len(kv[1].inputs)
    @staticmethod
    def _iter_layer_filters(layer):
        "Iterates over the calcs in layer that filter their inputs."
        calcs = sorted(layer.calcs.items(), key=plan._filter_sort)
        for (nm,c) in calcs:
            filts = set(c.inputs)
            filts &= set(c.outputs)
            if len(filts) > 0:
                yield (nm, c, filts)
    @staticmethod
    def _separate_filters(layer):
        "Separates a layer into a sequence of layers with ordered filters."
        # Go through each calc in this layer that filters its inputs and figure
        # out if we need a separate layer for it.
        names = []
        ins = layer.inputs
        for (nm,c,filts) in plan._iter_filters(layer):
            outs = ins.transient()
            outs |= set(c.outputs)
            outs = outs.persistent()
            yield plan.Layer(ins, pdict(nm=c), outs)
            ins = outs
            names.append(nm)
        if len(names) == 0:
            # We didn't return any separate layers, so we just yield the
            # original layer.
            yield layer
        else:
            yield plan.Layer(ins, layer.calcs.dropall(names), layer.outputs)
    @staticmethod
    def _find_trname(valnames, k, suffix=None):
        "Returns a new unique value name appropriate for internal translation."
        if suffix is not None:
            k = f'{k}__{suffix}'
            if k not in valnames:
                return k
        k0 = k + '_'
        ii = 1
        k = f'{k0}1'
        while k in valnames:
            ii += 1
            k = f'{k0}{ii}'
        return k
    @staticmethod
    def _source_lookup(inputtup, calctup, src):
        if isinstance(src, tuple):
            (cidx, oidx) = src
            lazycalc = calctup[cidx]
            val = lazycalc()[oidx]
        else:
            val = inputtup[src]()
        return val
    @staticmethod
    def _lookup(calcdata, inputtup, calctup, key):
        return plan._source_lookup(inputtup, calctup, calcdata.sources[key])
    @staticmethod
    def _call_calc(inputtup, calctup, c, args):
        args = map(partial(plan._source_lookup, inputtup, calctup), args)
        kwonlyargs = c.argspec.kwonlyargs
        nkwonly = len(kwonlyargs)
        if nkwonly == 0:
            r = c.eager_call(*args)
        else:
            args = tuple(args)
            kw = dict(zip(kwonlyargs, args[-nkwonly:]))
            args = args[:nkwonly]
            r = c.eager_call(*args, **kw)
        if is_amap(r):
            return tuple(map(r.__getitem__, c.outputs))
        else:
            return tuple(r)
    @staticmethod
    def _make_calctup(calcdata, inputtup):
        f = plan._call_calc
        # We take advantage of Python's weak closures here:
        calctup = ()
        calctup = tuple(
            lazy(lambda c,args: f(inputtup, calctup, c, args), c, args)
            for (c,args) in zip(calcdata.calcs, calcdata.args))
        return calctup
    @staticmethod
    def _update_calctup(calcdata, inputtup, calctup, cidx):
        f = plan._call_calc
        calctup[cidx] = lazy(
            plan._call_calc,
            inputtup, calctup,
            calcdata.calcs[cidx],
            calcdata.args[cidx])
        return calctup
    @staticmethod
    def _make_srcs_args(names, calcs, params):
        args = []
        srcs = {k:ii for (ii,k) in enumerate(params)}
        for (cidx,(nm,c)) in enumerate(zip(names, calcs)):
            # Wire up the inputs/arguments:
            a = []
            for ii in c.inputs:
                a.append(srcs[ii])
            args.append(tuple(a))
            # And the outputs/sources.
            for (oidx, oo) in enumerate(c.outputs):
                assert oo not in srcs, "filter detected in flattened plan"
                srcs[oo] = (cidx, oidx)
        # At this point...
        # args is the list (in calc order) of where to find the inputs of
        # each calc.
        args = tuple(args)
        # srcs is the dictionary whose keys are plan value names and whose
        # values are the indices telling us where to find that value.
        srcs = pdict(srcs)
        # That's all.
        return (srcs, args)
    @staticmethod
    def _transitive_closure(edges):
        clos = set(edges)
        while True:
            s = set((u1,v2) for (u1,v1) in clos for (u2,v2) in clos if v2 == u1)
            if clos.issuperset(s): break
            clos |= s
        res = defaultdict(lambda:set())
        for (u,v) in clos:
            res[u].add(v)
        return res
    # Construction -------------------------------------------------------------
    __slots__ = (
        'inputs', 'outputs', 'defaults', 'requirements',
        'input_docs', 'output_docs'
        'calcdata', 'valsources', 'dependants')
    def __init__(self, *args, **kwargs):
        # We ignore the arguments because they are handled by pdict's __new__
        # method.
        # We can start by gathering up the calcs that deal with each of the
        # plan's values. The val2calc dict maps each value name in the plan to a
        # tuple of (output, filter, input) calcs that process the value. The
        # output calcs are those that produce the value as an output but don't
        # requie it as an input; the filter calcs are those that require the
        # value as an input and that produce the value as an output; the input
        # calcs are those that require the calc as an input but don't produce it
        # as output.
        val2calc = defaultdict(lambda:([],[],[]))
        filters = set()
        params = []
        reqs = tset()
        for (nm,c) in self.items():
            # Examine the inputs and outputs:
            for oo in c.outputs:
                if oo not in c.inputs:
                    val2calc[oo][0].append(nm)
                else:
                    filters.add(nm)
                    val2calc[oo][1].append(nm)
            for ii in c.inputs:
                if ii not in c.outputs:
                    val2calc[ii][2].append(nm)
            # If it's not a lazy calculation, it goes on the requirements list:
            if not c.lazy:
                reqs.add(nm)
        nval = len(val2calc)
        for (k,(outs,filts,ins)) in val2calc.items():
            nouts = len(outs)
            if nouts == 0:
                # If it's an output of zero calcs, then it's a parameter of the
                # overall plan.
                params.append(k)
            elif nouts > 1:
                # If any of the values are produced by more than 1 output, then
                # that is a violation of the graph rules.
                raise ValueError(
                    f"value {k} is an output of {nouts} calcs: {outs}")
        # Now that we have a list of the params for the plan, we can start
        # putting the calcs in a calculation order. The order must guarantee
        # that any calc at position p has inputs that are drawn only from plan
        # parameters and the outputs of calcs whose position is less than p. If
        # we can make such an ordering, then we can make a DAG.
        params = pset(params)
        inputs = params.transient()
        calcs = set(self.keys())
        calcorder = []
        input_docs = defaultdict(lambda:[])
        output_docs = defaultdict(lambda:[])
        # We're going to be selecting filters and we want to do so
        # preferentially based on the number of inputs they require.
        filts = tset(sorted(filters, key=lambda f:-len(self[f].inputs)))
        filters = pset(filts)
        is_ready = lambda f: self[f].inputs <= inputs
        while len(calcs) > 0:
            # We start by greedily selecting filters.
            if len(filts) > 0:
                nextcalc = next(filter(is_ready, filts), None)
            else:
                nextcalc = None
            if nextcalc is None:
                # If we get here, we didn't find a filter, so we look for any
                # other calc we can run!
                nextcalc = next(filter(is_ready, calcs), None)
                if nextcalc is None:
                    raise ValueError(
                        f"unreachable calcs: {tuple(calcs)}; this is likely due"
                        f" to a circular dependency")
            else:
                filts.discard(nextcalc)
            # We have a next calculation in the order, so we add it.
            calcorder.append(nextcalc)
            c = self[nextcalc]
            inputs.addall(c.outputs)
            calcs.discard(nextcalc)
            # While we're going through the calcs in order, we process the docs:
            # Process the documentation:
            for (inp,doc) in c.input_docs.items():
                if not doc:
                    continue
                s = f"{nextcalc} input: {inp}"
                if len(s) > 80:
                    s = s[77] + '...'
                if inp in params:
                    input_docs[inp].append(s + '\n' + doc)
                else:
                    output_docs[inp].append(s + '\n' + doc)
            for (out,doc) in c.output_docs.items():
                if not doc:
                    continue
                s = f"{nextcalc} output: {out}"
                if len(s) > 80:
                    s = s[77] + '...'
                output_docs[out].append(s + '\n' + doc)
        doc_connectfn = lambda v: '\n---\n'.join(v)
        input_docs = pdict(valmap(doc_connectfn, input_docs))
        output_docs = pdict(valmap(doc_connectfn, output_docs))
        ncalcs = len(calcorder)
        calcidx = pdict(zip(calcorder, range(ncalcs)))
        outputs = inputs
        outputs -= params
        outputs = outputs.persistent()
        inputs = params
        # We now have a calculation ordering that we can use to turn the plan's
        # filtered values into sequential values.  For example, if the variable
        # 'x' is filtered through calculations 'f', 'g', and 'h', in that order,
        # then we update the inputs/outputs of the functions to force an
        # ordering. First, the initial parameter will be renamed to 'x.', then f
        # is changed to take 'x.' as an input in place of 'x' and to produce
        # 'x.f'. Then g is changed so that it takes 'x.f' instead of x and
        # produces 'x.g'. Then h is changed so that it takes 'x.h' as instead of
        # 'x' and produces the output 'x'.  The actual internal names don't use
        # periods (they stay as valid variable names that are potentially
        # randomly chosen using the _find_trname staticmethod).
        names = tuple(calcorder)
        calcs = tuple(self[k] for k in calcorder)
        if len(filters) == 0:
            # We don't actually have any filters to put in order, so we have the
            # straightforward job of wiring things up as-is. We already know
            # that there aren't any cycles in the graph (they would have
            # appeared earlier).
            (srcs, args) = plan._make_srcs_args(names, calcs, params)
            calcdata = plan.CalcData(names, calcs, args, srcs, calcidx)
            valsources = srcs
        else:
            # We need to do two things: (1) make a plan out of the unfiltered
            # calcs (i.e., separate inputs and outputs of each filter calc into
            # different variables and link them up across calculations), and (2)
            # make a translation for the output values.
            tr = {}
            valnames = set(val2calc.keys())
            new_calcs = []
            for (nm,c) in zip(names, calcs):
                filts = c.inputs & set(c.outputs)
                calctr = {}
                for f in filts:
                    k = tr.get(f, f)
                    new_k = plan._find_trname(valnames, k, nm)
                    valnames.add(new_k)
                    tr[f] = new_k
                    calctr[f] = (k, new_k)
                new_calcs.append(c.tr(tr, calctr))
            (srcs, args) = plan._make_srcs_args(names, new_calcs, params)
            calcdata = plan.CalcData(names, new_calcs, args, srcs, calcidx)
            valsrcs = tdict()
            for k in valnames:
                valsrcs[k] = srcs[tr.get(k, k)]
            valsources = valsrcs.persistent()
        # Go through the calc ordering and find the earliest default value for
        # each of the keys.
        defaults = rmerge(*(c.defaults for c in calcs if c.defaults))
        # Note the requirements.
        requirements = pset(reqs)
        # One final thing we need to do is to make the dependants graph; this is
        # basically the graph of calculations and outputs that need to be
        # updated / reset any time a parameter is changed.
        depset = set()
        for (cidx,c) in enumerate(calcdata.calcs):
            for k in c.inputs:
                depset.add((k, cidx))
            for k in c.outputs:
                depset.add((cidx, k))
        depgraph = plan._transitive_closure(depset)
        deps = tdict()
        for k in c.inputs:
            odeps = []
            cdeps = []
            for d in depgraph[k]:
                if isinstance(d, str):
                    odeps.append(d)
                else:
                    cdeps.append(d)
            deps[k] = plan.DepData(tuple(odeps), tuple(cdeps))
        dependants = deps.persistent()
        # Now set all the variables, and we're done!
        object.__setattr__(self, 'inputs', inputs)
        object.__setattr__(self, 'outputs', outputs)
        object.__setattr__(self, 'defaults', defaults)
        object.__setattr__(self, 'requirements', requirements)
        object.__setattr__(self, 'input_docs', input_docs)
        object.__setattr__(self, 'output_docs', output_docs)
        object.__setattr__(self, 'calcdata', calcdata)
        object.__setattr__(self, 'valsources', valsources)
        object.__setattr__(self, 'dependants', dependants)
    # Methods ------------------------------------------------------------------
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
    __slots__ = ('plan', 'inputs', '_calcdata', '_inputdata')
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
        given_params = set(params.keys())
        # Are we missing any parameters?
        missing_params = plan.inputs - given_params
        if len(missing_params) > 0:
            raise ValueError(f"missing inputs: {tuple(missing_params)}")
        # Do we have extra inputs?
        extra_params = given_params - plan.inputs
        if len(extra_params) > 0:
            raise ValueError(f"extra inputs: {tuple(extra_params)}")
        # Okay, we have the correct parameters. We can make the input tuple.
        inp = []
        pparams = params.to_pdict() if isinstance(params, ldict) else params
        for k in plan.inputs:
            v = pparams[k]
            if isinstance(v, lazy):
                inp.append(v)
            else:
                inp.append(lazy(identfn, v))
        inputtup = tuple(inp)
        # We can also make a lazy object per calc for the calctup.
        calctup = plan._make_calctup(plan.calcdata, inputtup)
        # Go ahead and do the initialization.
        items = ldict.empty.transient()
        for k in plan.inputs:
            # If this input gets filtered, we need a lazy lookup:
            src = plan.valsources[k]
            if isinstance(src, tuple):
                v = lazy(plan._source_lookup, inputtup, calctup, src)
            else:
                v = pparams[k]
            items[k] = v
        for k in plan.outputs:
            src = plan.valsources[k]
            items[k] = lazy(plan._source_lookup, inputtup, calctup, src)
        self = super(plandict, cls).__new__(cls, items)
        # And set our special member-values.
        object.__setattr__(self, 'plan', plan)
        object.__setattr__(self, 'inputs', params)
        object.__setattr__(self, '_calcdata', calctup)
        object.__setattr__(self, '_inputdata', inputtup)
        # Finally, now that we have the object entirely initialized, we can run
        # the required calculations.
        for r in plan.requirements:
            cidx = plan.calcdata.index[r]
            calctup[cidx]()
        # That's all!
        return self
    @classmethod
    def _new_from_plandict(cls, pd, *args, **kwargs):
        plan = pd.plan
        calcdata = plan.calcdata
        if len(args) == 0 and len(kwargs) == 0:
            return pd
        # First, merge from left-to-right, respecting laziness.
        params = merge(pd.inputs, *args, **kwargs)
        paramkeys = set(params.keys())
        # There must only be parameters here.
        extras = paramkeys - plan.inputs
        if len(extras) > 0:
            raise ValueError(f"unrecognized inputs: {tuple(extras)}")
        # Make a new inputtup and calctup.
        inputtup = list(pd._inputdata)
        pparams = params.to_pdict() if isinstance(params, ldict) else params
        allvals = set()
        allcals = set()
        for (ii,k) in enumerate(plan.inputs):
            if k in pparams:
                v = pparams[k]
                inputtup[ii] = v if isinstance(v, lazy) else lazy(identfn, v)
                (vals, cidcs) = plan.dependants[k]
                allcals.update(cidcs)
                allvals.update(vals)
                allvals.add(k)
        inputtup = tuple(inputtup)
        calctup = list(pd._calcdata)
        for cidx in allcals:
            plan._update_calctup(calcdata, inputtup, calctup, cidx)
        calctup = tuple(calctup)
        items = pd.transient()
        for k in allvals:
            src = plan.valsources[k]
            if isinstance(src, tuple):
                v = lazy(plan._source_lookup, inputtup, calctup, src)
            else:
                v = pparams[k]
            items[k] = v
        # Allocate the lazy dict.
        self = super(plandict, cls).__new__(cls, items)
        # And set our special member-values.
        object.__setattr__(self, 'plan', plan)
        object.__setattr__(self, 'inputs', params)
        object.__setattr__(self, '_calcdata', calctup)
        object.__setattr__(self, '_inputdata', inputtup)
        # Finally, now that we have the object entirely initialized, we can run
        # the required calculations.
        for r in plan.requirements:
            cidx = plan.calcdata.index[r]
            calctup[cidx]()
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
