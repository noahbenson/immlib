# -*- coding: utf-8 -*-
###############################################################################
# immlib/workflow/_core.py


# Dependencies ################################################################

import copy, textwrap
from collections.abc import (Callable, Mapping)
from collections import (defaultdict, namedtuple)
from functools import (reduce, wraps, partial, update_wrapper)
from inspect import (signature, Parameter)
from joblib import Memory
from pathlib import Path

import numpy as np
from pcollections import (
    pdict, tdict, ldict, tldict,
    lazy, holdlazy,
    pset, tset,
    plist)

from ..doc import (docwrap, make_docproc, reindent, detect_indentation)
from ..util import (
    is_pdict, is_str, is_number, is_tuple, is_dict, is_ldict,
    is_array, is_integer, strisvar, is_amap, is_pcoll, to_pcoll,
    to_pathcache, to_lrucache, identfn,
    merge, rmerge, valmap)


# calc ########################################################################

class calc:
    '''Decorator type that represents a single calculation in a calc-plan.
    
    The ``calc`` class encapsulates data regarding the calculation of a single
    set of output values from a separate set of input values: a calculation
    component that can be fit together with other such components to make a
    calculation plan.

    ``@calc`` by itself can be used as a decorator to indicate that the
    function that follows is a calculation component; calculation components
    can be combined to form ``plan`` objects, which can encapsulate a flexible
    workflow of Python computations. When ``@calc`` is used as a decorator by
    itself, then the calc is considered to have a single output value whose
    name is the same as that of the function it decorates.
    
    ``@calc(names...)`` accepts a string or strings that name the output values
    of the calc function. In this case, the decorated function must return
    either a tuple of thes values in the order they are given or a dictionary
    in which the keys are the same as the given names.
    
    ``@calc(None)`` is a special instance which indicates that the lazy
    argument is to be ignored (it is forced to be ``False``), no output values
    are to be produced by the function, and the calculation must always run
    when the input parameters are updated.
    
    The ``calc`` class parses its inputs and outputs through the
    ``immlib.docwrap`` function in order to collect documentation (see the
    ``input_docs`` and ``output_docs`` attributes, below). The ``'Inputs'`` and
    ``'Outputs'`` sections are tracked as the documentation of the parameters,
    and are required to be formatted using [NumPy's documentation
    style](https://numpydoc.readthedocs.io/en/latest/format.html) in order for
    the parameter documentation to be properly extracted. Users of calculation
    objects should decorate their functions using ``docwrap`` manually
    themselves, however (if desired), because decorating a function with
    ``calc`` alone does not cause the function's documentation to be available
    to other functions that use ``@docwrap`` to format their docstrings.

    Caching for calculations requires some care. First, the ``calc``- and
    ``plan``-based workflow system in ``immlib`` is designed to work best with
    ``calc`` objects that are pure functions. A function ``f(*args, **kw)`` is
    pure if it has no side-effects and if ``f(*args1, **kw1) == f(*args2,
    **kw2)`` is true whenever ``args1 == args2 and kw1 == kw2``. That is, ``f``
    always produces the same outputs when given the same inputs. Plans that
    contain unpure functions can work fine in many contexts, but unpure
    ``calc`` objects will break caching because the return value of a cached
    unpure calculation will always be the same value. (In other words, the
    value that is calculated and cached by the function the first time it is
    called.)

    Second, the ``calc`` type has an option, ``pathcache``, which can be set to
    an explicit path to which all calculations run by the created ``calc``
    object will be cached and later uncached if re-requested. This is
    occasionally appropriate for a particular compute environment, but a better
    approach is typically to grant control of caching and cache paths to the
    user who creates the ``plandict`` object downstream of the creation of the
    ``calc`` objects. To enable this behavior, one should instead use the
    option ``pathcache=True``, which enables caching of calculations to a
    specific cache path when provided by the user during the creation of the
    ``plandict`` (the default is ``False``, which disables path caching for the
    calculation).

    Parameters
    ----------
    outputs : strings
        The positional arguments to ``@calc()`` provide the names of the output
        variables. The names must all be valid variable names (see
        ``immlib.strisvar``).
    name : None or str, optional
        The name of the function. The default, ``None``, uses ``fn.__name__``.
    lazy : bool, optional
        Whether the calculation unit should be calculated lazily (``True``) or
        eagerly (``False``) when a plandict is created. The default is
        ``True``.
    lrucache : int, optional
        The number of recently calculated results to cache. If this value is 0,
        then no memoization is done (the default). If ``lrucache`` is an
        integer greater than 0, then an LRU cache is used with a maximum size
        of `lrucache`. If ``lrucache`` is ``inf``, then all values are cached
        indefinitely. Note that this cache is performed at the level of the
        calculation using Python's ``functools`` caching decorators.
    pathcache : None, bool, or path-like, optional
        If ``pathcache`` is a path-like object (typically a ``pathlib.Path``
        orstring) that references a directory, then the results are cached in
        files in the given directory whenever possible. The ``pathcache``
        option may also a 2-tuple containing a path followed by options to the
        ``joblib.Memory`` constructor; see ``immlib.util.to_pathcache`` for
        more information.
    indent : int or None, optional
        The indentation level of the function's docstring. The default is
        ``None``, which indicates that the indentation level should be deduced
        from the docstring itself.

    Attributes
    ----------
    name : str
        The name of the calculation function.
    base_function : callable
        The original function, prior to decoration for caching.
    lrucache : None or lrucache-like
        The in-memory cache being used. If this value is ``None``then no
        in-memory cache is being used. If it is an integer, this indicates the
        number of least recently used objects being stored in the
        cache. Otherwise, ``lrucache`` will be a function used to wrap the
        ``base_function`` of the calculation for caching. The ``lrucache``
        parameter is filtered by the ``immlib.util.to_lrucache`` function in
        order to convert it into a valid ``functools.lru_cache`` object.
    pathcache : None or pathcache-like
        The file-system-based cache being used. If this value is ``None`` or
        ``False``, then no filesystem cache is being used by the calculation
        directly. If this value is a path object, then that path is the
        directory in which cache files are saved/loaded. If ``pathcache`` is a
        ``joblib.Memory`` object, then this object handles the caching for the
        calculation. Otherwise, the value will be ``True``, indicating that
        caching should be performed automatically using the ``cache_path``
        input to the calc. If ``cache_path`` was not already one of the inputs,
        it is added as an input with the default value ``None``. When automatic
        caching is performed, the ``cache_path`` is automatically converted
        into a ``joblib.Memory`` object using the ``immlib.util.to_pathcache``
        function.
    function : callable
        The function itself.
    signature : inspect.Signature
        The signature of ``fn``, as returned from ``inspect.signature(fn)``.
    inputs : pcollections.pset of str
        The names of the input parameters for the calculation.
    outputs : tuple of str
        The names of the output values of the calculation.
    defaults : pcollections.pdict
        A persistent dictionary whose keys are input parameter names and whose
        values are the default values for the associated parameters.
    lazy : bool
        Whether the calculation is intended as a lazy (``True``) or eager
        (``False``) calculation.
    input_docs : pcollections.pdict
        A ``pdict`` object whose keys are input names and whose values are the
        documentation for the associated input parameters.
    output_docs : pcollections.pdict
        A ``pdict`` object whose keys are output names and whose values are
        the documentation for the associated output values.
    '''
    __slots__ = (
        'name', 'base_function', 'lrucache', 'pathcache',
        'function', 'signature', 'inputs', 'outputs', 'defaults', 'lazy',
        'input_docs', 'output_docs')
    @staticmethod
    def _dict_persist(arg):
        return None if arg is None else pdict(arg)
    @classmethod
    def _interpret_pathcache(cls, pathcache):
        if pathcache is None or pathcache is False:
            return None
        elif pathcache is True:
            return True
        else:
            return to_pathcache(pathcache)
    @staticmethod
    def _pathcache_woutsig(base_fn, *args, **kw):
        if 'cache_path' in kw:
            cache_path = kw.pop('cache_path')
        elif len(args) > 0:
            cache_path = args[-1]
            args = args[:-1]
        else:
            cache_path = None
        if cache_path is None or cache_path is False:
            return base_fn(*args, **kw)
        cp = to_pathcache(cache_path)
        cache_fn = cp.cache(base_fn)
        return cache_fn(*args, **kw)
    @staticmethod
    def _pathcache_withsig(base_fn, sig, *args, **kw):
        ba = sig.bind(*args, **kw)
        ba.apply_defaults()
        cp = ba.arguments['cache_path']
        if cp is None or cp is False:
            return base_fn(*args, **kw)
        cp = to_pathcache(cp)
        cache_fn = cp.cache(base_fn)
        return cache_fn(*args, **kw)
    @staticmethod
    def _apply_caching(base_fn, sig, lrucache, pathcache):
        # We assume that cache and pathcache have already been appropriately
        # filtered by the to_lrucache and to_pathcache functions.
        newsig = None
        if pathcache is None or pathcache is False:
            # No caching requested, either for plandicts or globally.
            fn = base_fn
        elif pathcache is True:
            # This means we are caching into the cache_path input, which may be
            # implicitly given. We make a special function if we need to ignore
            # (not pass along) the cache_path argument.
            if 'cache_path' in sig.parameters:
                fn = partial(calc._pathcache_withsig, base_fn, sig)
            else:
                fn = partial(calc._pathcache_woutsig, base_fn)
                params = list(sig.parameters.values())
                params.append(
                    Parameter(
                        'cache_path',
                        Parameter.KEYWORD_ONLY,
                        default=None))
                newsig = sig.replace(parameters=params)
        elif isinstance(pathcache, Memory):
            fn = pathcache.cache(base_fn)
        else:
            fn = Memory(pathcache).cache(base_fn)
        if lrucache is not None:
            fn = lrucache(fn)
        # We want to wrap base_fn but use the signature sig.
        wrapfn = fn if fn is base_fn else wraps(base_fn)(fn)
        if newsig is not None:
            wrapfn.__signature__ = newsig
        return wrapfn
    @classmethod
    def _new(cls, fn, outputs,
             name=None, lazy=True, indent=None,
             lrucache=0, pathcache=None):
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
            input_docs = tdict()
            output_docs = tdict()
            for (k,doc) in dp.params.items():
                if k.startswith('fn.inputs.'):
                    input_docs[k[10:]] = doc
                elif k.startswith('fn.parameters.'):
                    input_docs[k[14:]] = doc
                elif k.startswith('fn.outputs.'):
                    output_docs[k[11:]] = doc
            input_docs  = pdict(input_docs)
            output_docs = pdict(output_docs)
        else:
            input_docs = pdict()
            output_docs = pdict()
            fndoc = None
        # Go ahead and allocate the object we're creating.
        self = object.__new__(cls)
        # Setting function to Ellipsis is a signal to the setattr method that
        # the object is being initialized; until we set function to something
        # else at the end of this function, setattr is allowed (i.e., the calc
        # becomes immutable once this function returns.)
        object.__setattr__(self, 'function', Ellipsis)
        # Set some attributes.
        self.name = name
        # Save the base_function before we do anything to it.
        self.base_function = fn
        # If there's a caching strategy here, use it.
        lrucache = to_lrucache(lrucache)
        self.lrucache = lrucache
        # If there's a cache path, note it.
        pathcache = self._interpret_pathcache(pathcache)
        self.pathcache = pathcache
        # Get the argspec for the calculation function.
        sig = signature(fn)
        for p in sig.parameters.values():
            if p.kind == p.VAR_POSITIONAL:
                raise ValueError("calculations do not support varargs")
            elif p.kind == p.VAR_KEYWORD:
                raise ValueError("calculations do not support varkw")
        # Figure out the inputs from the argspec; we set them below, after we
        # have checked the pathcache.
        inputs = pset(sig.parameters.keys())
        # Check that the outputs are okay.
        outputs = tuple(outputs)
        for out in outputs:
            if not strisvar(out):
                raise ValueError(f"calc output '{out}' is not a valid varname")
        self.outputs = outputs
        # We need to grab the defaults also.
        dflts = {}
        for p in sig.parameters.values():
            if p.default is not p.empty:
                dflts[p.name] = p.default
        # If pathcache is True, then cache_path is an implicit argument if not
        # already included; add that here if necessary. This won't screw up the
        # arguments when the eager_call is eventually made because the
        # _apply_caching function handles this.
        if pathcache is True and 'cache_path' not in inputs:
            inputs = inputs.add('cache_path')
            dflts['cache_path'] = None
        self.inputs = inputs
        self.defaults = pdict(dflts)
        # Save the laziness status and the documentations.
        self.lazy = bool(lazy)
        self.input_docs = input_docs
        self.output_docs = output_docs
        # Last thing is to set the function, which signals that construction is
        # done and the calc is now immutable.
        cache_fn = self._apply_caching(fn, sig, lrucache, pathcache)
        # At this point we get the signature for cache_fn because it's possible
        # that cache_fn added a cache_path parameter.
        self.signature = signature(cache_fn)
        self.function = cache_fn
        # That is all for the constructor. However, what we actually return
        # from a calc decorator/call is a function with a
        # `calc` field.
        try:
            fn.calc = self
        except Exception:
            # If the above fails, it's probably because fn isn't a normal
            # function and doesn't allow a field to be set. We can hack that.
            func = fn
            @wraps(fn)
            def fn_wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            fn_wrapper.calc = self
            fn = fn_wrapper
        return fn
    def __new__(cls, *args,
                name=None, lazy=True,
                lrucache=0, pathcache=None, indent=None):
        kw = dict(name=name, lazy=lazy, lrucache=lrucache,
                  pathcache=pathcache, indent=indent)
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
    def update_function(self, fn):
        """Updates the function and its calc object.

        On occasion, a function decorated with ``@calc`` is later decorated
        with another feature, such as a decorator that causes its inputs to be
        promoted. Such a decorator, when it comes after the ``@calc`` decorator
        (i.e., on a line prior to the ``@calc``), will not update the
        calculation object and thus the calculation object, when invoked, will
        not call the fully decorated version of its function. To fix this, any
        ``calc`` object whose ``base_function`` member variable is identical to
        the function given to a ``plan`` object (i.e., ``f is not
        to_calc(f).base_function``), then this method is called to return a
        ``calc`` object whose ``base_function`` has been updated. If possible,
        it also updates the `fn` argument to use the new ``calc`` object.

        In general, this function should not be called directly by the user;
        rather, it gets run automatically when a calc is added to a new
        ``plan`` or ``planobject``.
        """
        # First, make sure the right calc was passed this function.
        if to_calc(fn, update=False) is not self:
            raise ValueError(
                "calcobj.update_function(f) called, but to_calc(f) is not"
                " calcobj")
        # Next make sure we aren't already up-to-date.
        if self.base_function is fn:
            return self
        # If fn.__wrapped__ doesn't exist or isn't the self.function, then the
        # function was either poorly wrapped or it wasn't made from
        # base_function and we need to raise an error.
        wrapped = fn
        wset = set([fn])
        while wrapped is not None:
            wrapped = getattr(fn, '__wrapped__', None)
            if wrapped is self.base_function:
                break
            wid = id(wrapped)
            if wid in wset:
                raise ValueError("loop in __wrapped__ attributes")
            wset.add(wid)
        if wrapped is None:
            raise ValueError(
                "calcobj.update_function(f) called, but f is not made from"
                " calcobj.base_function")
        # At this point, we have verified that this is an appropriate update,
        # so we can go ahead and make a duplicate calc with the new function.
        new_fn = calc._new(
            fn, self.outputs,
            name=self.name, lazy=self.lazy,
            lrucache=self.lrucache,
            pathcache=self.pathcache)
        return new_fn.calc
    def eager_call(self, *args, **kwargs):
        """Eagerly calls the given calculation using the arguments.

        ``c.eager_call(...)`` returns the result of calling the calculation
        ``c(...)`` directly. Using the ``eager_call`` method is different from
        calling the ``__call__`` method only in that the ``eager_call`` method
        ignores the ``lazy`` member and always returns the direct results of
        calling the calculation; using the ``__call__`` method will result in
        ``eager_call`` being run if the calculation is not lazy and in
        ``lazy_call`` being run if the calculation is lazy.

        See Also
        --------
        calc.eager_mapcall, calc.lazy_call, calc.lazy_mapcall
        """
        # Now we just pass these arguments along (the function itself has been
        # given the caching code via decorators already).
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

        ``calc.lazy_call(...)`` is equivalent to ``calc(...)`` except that the
        ``lazydict`` that it returns encapsulates the running of the
        calculation itself, so that ``calc(...)`` is not run until one of the
        lazy values is requested.

        See Also
        --------
        calc.mapcall, calc.lazy_mapcall
        """
        # First, create a lazy for the actual call:
        lazycall = lazy(self.eager_call, *args, **kwargs)
        # Then make a lazy map of all the outputs, each of which pulls from
        # this lazy object to get its values.
        return ldict(
            {k: lazy(lambda k: lazycall()[k], k)
             for k in self.outputs})
    def __call__(self, *args, **kwargs):
        if self.lazy:
            return self.lazy_call(*args, **kwargs)
        else:
            return self.eager_call(*args, **kwargs)
    def call(self, *args, **kwargs):
        """Calls the calculation and returns the results dictionary.

        ``c.call(...)`` is an alias for ``c(...)``.

        See also ``calc.mapcall``, ``calc.eager_call``, and ``calc.lazy_call``.
        """
        if self.lazy:
            return self.lazy_call(*args, **kwargs)
        else:
            return self.eager_call(*args, **kwargs)
    def _maps_to_args(self, args, kwargs):
        opts = merge(self.defaults, *args, **kwargs)
        args = []
        kwargs = {}
        for (name,p) in self.signature.parameters.items():
            if name not in opts:
                raise ValueError(f"required argument {name} not found")
            if p.kind == p.POSITIONAL_ONLY:
                args.append(opts[name])
            else:
                kwargs[name] = opts[name]
        return (args, kwargs)
    def eager_mapcall(self, *args, **kwargs):
        """Calls the given calculation using the parameters in mappings.

        ``c.eager_mapcall(map1, map2..., key1=val1, key2=val2...)`` returns the
        result of calling the calculation ``c(...)`` using the parameters found
        in the provided mappings and key-value pairs. All arguments of
        ``mapcall`` are merged left-to-right using ``immlib.merge`` then passed
        to ``c.function`` as required by it.
        """
        (args, kwargs) = self._maps_to_args(args, kwargs)
        return self.eager_call(*args, **kwargs)
    def lazy_mapcall(self, *args, **kwargs):
        """Calls the given calculation lazily using the parameters in mappings.

        ``c.lazy_mapcall(map1, map2..., key1=val1, key2=val2...)`` returns the
        result of calling the calculation ``c(...)`` using the parameters found
        in the provided mappings and key-value pairs. All arguments of
        ``mapcall`` are merged left-to-right using ``immlib.merge`` then passed
        to ``c.function`` as required by it.

        The only difference between ``calc.mapcall`` and ``calc.lazy_mapcall``
        is that the lazydict returned by the latter method encapsulates the
        calling of the calculation itself, so no call to the calculation is
        made until one of the values of the lazydict is requested.

        See Also
        --------
        calc.eager_mapcall, calc.lazy_call, calc.eager_call
        """
        # Note that all the args must be dictionaries, so we make copies of
        # them if they're not persistent dictionaries. This prevents later
        # modifications from affecting the results downstream.
        args = [d if is_pdict(d) else dict(d) for d in args]
        # First, create a lazy for the actual call:
        calldel = lazy(self.eager_mapcall, *args, **kwargs)
        # Then make a lazy map of all the outputs, each of which pulls from
        # this lazy object to get its values.
        fn = lambda k: calldel()[k]
        return ldict({k: lazy(fn, k) for k in self.outputs})
    def mapcall(self, *args, **kwargs):
        """Calls the calculation and returns the results dictionary.

        ``c.mapcall(map1, map2..., key1=val1, key2=val2...)`` returns the
        result of calling the calculation ``c(...)`` using the parameters found
        in the provided mappings and key-value pairs. All arguments of
        ``mapcall`` are merged left-to-right using ``immlib.merge`` then passed
        to ``c.function`` as required by it.

        See Also
        --------
        calc.lazy_mapcall, calc.eager_mapcall, calc.call
        """
        if self.lazy: return self.lazy_mapcall(*args, **kwargs)
        else:         return self.eager_mapcall(*args, **kwargs)
    def __setattr__(self, k, v):
        if self.function is Ellipsis:
            # We're still initializing, so setattr is allowed.
            return object.__setattr__(self, k, v)
        else:
            raise TypeError('calc objects are immutable')
    def __delattr__(self, k):
        raise TypeError('calc objects are immutable')
    @staticmethod
    def _tr_map(tr, m, is_input):
        if m is None:
            return None
        tup_ii = int(not is_input)
        is_ld = is_ldict(m)
        it = holdlazy(m).items()
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
    def rename_keys(self, *args, **kwargs):
        """Returns a copy of the calculation with inputs and outputs renamed.
        
        ``calc.rename_keys(...)`` returns a copy of ``calc`` in which the input
        and output values of the function have been translated. The translation
        is found from merging the list of 0 or more dict-like arguments given
        left-to-right followed by the keyword arguments into a single
        dictionary. The keys of this dictionary are translated into their
        associated values in the returned dictionary.

        If any of the values of the merged dictionary are 2-tuples, then they
        are interpreted as ``(input_tr, output_tr)``. In this case, then the
        key must be associated with a name that appears in both the
        calculation's input list and its output list, and the two names are
        translated differently.
        """
        d = merge(*args, **kwargs)
        # Make a copy.
        tr = object.__new__(calc)
        # Simple changes first.
        trhash = np.fromiter(map(hash, d.items()), dtype=np.intp)
        trhash = np.sum(trhash.astype(np.uintp))
        object.__setattr__(tr, 'name', self.name + f'.rename{hex(trhash)}')
        object.__setattr__(tr, 'base_function', self.base_function)
        object.__setattr__(tr, 'lrucache', self.lrucache)
        object.__setattr__(tr, 'pathcache', self.pathcache)
        object.__setattr__(tr, 'lazy', self.lazy)
        object.__setattr__(tr, 'inputs', calc._tr_set(d, self.inputs, True))
        object.__setattr__(tr, 'outputs', calc._tr_tup(d, self.outputs, False))
        object.__setattr__(
            tr, 'defaults', calc._tr_map(d, self.defaults, True))
        object.__setattr__(
            tr, 'input_docs', calc._tr_map(d, self.input_docs, True))
        object.__setattr__(
            tr, 'output_docs', calc._tr_map(d, self.output_docs, False))
        # Translate the argspec.
        params = []
        for (k,v) in self.signature.parameters.items():
            name = d.get(k, k)
            if name != k:
                name = name if isinstance(name, str) else name[0]
                v = v.replace(name=name)
            params.append(v)
        newsig = self.signature.replace(parameters=params)
        object.__setattr__(tr, 'signature', newsig)
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
        wrapfn = wraps(self.function)(_tr_fn_wrapper)
        object.__setattr__(tr, 'function', wrapfn)
        return tr
    def with_lrucache(self, new_cache):
        "Returns a copy of a calc with a different in-memory cache strategy."
        new_cache = to_lrucache(new_cache)
        if new_cache is self.lrucache:
            return self
        new_calc = copy.copy(self)
        object.__setattr__(new_calc, 'lrucache', new_cache)
        fn = self.base_function
        new_fn = calc._apply_caching(fn, new_cache, self.pathcache)
        if fn is not new_fn:
            object.__setattr__(new_calc, 'function', new_fn)
        return new_calc
    def with_pathcache(self, new_path):
        """Returns a copy of a calc with a different cache directory."""
        new_path = self._interpret_pathcache(new_path)
        if new_cache is self.pathcache:
            return self
        new_calc = copy.copy(self)
        object.__setattr__(new_calc, 'pathcache', new_cache)
        fn = self.base_function
        new_fn = calc._apply_caching(fn, self.lrucache, new_cache)
        if fn is not new_fn:
            object.__setattr__(new_calc, 'function', new_fn)
        return new_calc
@docwrap('immlib.workflow.is_calc')
def is_calc(obj, /):
    """Determines if an object is a ``calc`` instance.

    ``is_calc(obj)`` returns ``True`` if `obj` is a ``calc`` object.

    .. Warning:: ``is_calc(obj)`` returns ``False`` if `obj` is a function that
        was decorated with the ``@calc`` decorator. This is because ``calc``
        does not turn its decorated functions into ``calc`` objects; rather it
        attaches a field ``calc`` to the decorated function. To see
        whether a function is was decorated by ``calc``, use ``is_calcfn``.

    See Also
    --------
    calc, to_calc, is_calcfn
    """
    return isinstance(obj, calc)
@docwrap('immlib.is_calcfn')
def is_calcfn(obj, /):
    """Determines if an object is function that was decorated by ``@calc``.

    ``is_calcfn(obj)`` returns ``True`` if `obj` is a function that was
    decorated with an ``@calc`` decorator or if `obj` is a ``calc`` object, and
    it returns ``False`` otherwise.

    Functions decorated with ``@calc`` are not changed but rather are given
    some metadata, which is stored in the member field ``calc``. For
    such functions, this field contains an object of type ``calc``.

    See Also
    --------
    calc, to_calc, is_calc
    """
    return isinstance(getattr(obj, 'calc', None), calc)
@docwrap('immlib.workflow.to_calc')
def to_calc(obj, /, update=True):
    """Converts an object into a ``calc`` object or raises a ``TypeError``.

    ``to_calc(obj)`` returns `obj` if `obj` is already a ``calc``
    object. Otherwise, if `obj` has the attribute ``calc``, then that attribute
    is returned.

    .. Note:: When a function is decorated by ``@calc``, the calculation data
        is stored in a ``calc`` object that is saved to the ``calc``
        field of the function, which is why the above works.
    """
    if isinstance(obj, calc):
        return obj
    c = getattr(obj, 'calc', None)
    if isinstance(c, calc):
        if update:
            c = c.update_function(obj)
        return c
    raise TypeError(f"to_calc received non-calc object of type {type(obj)}")


# plan ########################################################################

class plan(pdict):
    '''Represents a directed acyclic graph of calculations.
    
    The ``plan`` class encapsulates individual functions that require
    parameters as inputs and produce outputs in the form of named values. Plan
    objects can be called as functions with a dictionary and/or a keyword
    arguments providing the plan's parameters; they always return a type of
    lazy dictionary called a ``plandict`` of the values they calculate, even if
    they calculate only a single value.

    Superficially, a ``plan`` is a ``pdict`` object whose values must all be
    ``calc`` objects. However, under the hood, every ``plan`` object maintains
    a directed acyclic graph of dependencies of the inputs and outputs of the
    calculation objects such that it can create ``plandict`` objects that reify
    the outputs of the various calculations lazily.

    The keys that are used in a plan must be strings but are not otherwise
    restricted.

    For a plan ``p = plan(calc_key1=calc1, calc_key2=calc2, ...)``, a
    ``plandict`` can be instantiated using the following syntax::

        pd = p(param1=val1, param2=val2, ...)

    This ``plandict`` is an enhanced ``ldict`` that evaluates components of the
    plan as requested based on laziness requirements of the calculations in the
    plan and on dictionary lookups of plan outputs. (``ldict`` is the lazy
    dictionary type from the ``pcollections`` library.)

    All plans implicitly contains the parameter ``'cache_path'`` with the
    default value of ``None``. This parameter is used by the plan's
    ``plandict`` objects, to cache the outputs of calculations that were
    constructed with the option ``pathcache=True``.

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
        A ``pset`` of the names of the required calculations of the plan (i.e.,
        those with option ``lazy=False``).
    __doc__ : str
        Every ``plan`` object is given a set of documentation which includes
        sections for the inputs and outputs as well as a listing of all the
        calculation steps.
    '''
    # Subclasses --------------------------------------------------------------
    CalcData = namedtuple(
        'CalcData',
        ('names', 'calcs', 'args', 'sources', 'index'))
    DepData = namedtuple(
        'DepData',
        ('inputs', 'calcs'))
    # Static Methods ----------------------------------------------------------
    @staticmethod
    def _filter_sort(kv):
        return len(kv[1].inputs)
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
        argvals = map(partial(plan._source_lookup, inputtup, calctup), args)
        args = []
        kwargs = {}
        c = to_calc(c)
        for (p,arg) in zip(c.signature.parameters.values(), argvals):
            if p.kind == p.POSITIONAL_ONLY:
                args.append[arg]
            else:
                kwargs[p.name] = arg
        r = c.eager_call(*args, **kwargs)
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
        calctup[cidx] = lazy(
            plan._call_calc,
            inputtup, calctup,
            calcdata.calcs[cidx],
            calcdata.args[cidx])
        return calctup
    def _update_dictdata(self, inputtup, calctup, updates):
        calcdata = self.calcdata
        sources = calcdata.sources
        dependants = self.dependants
        valsources = self.valsources
        calc_updates = set()
        # We're outputting/building-up new_inputtup and new_calctup from the
        # inputtup and calctup values.
        # We use some sleight of hand in this function:
        # The a = lazy(f, arg) construct makes a closure over the value of arg.
        # The b = lambda: f(arg) construct makes a closure over the symbol arg.
        # This means that if arg is updated after both of these lines, then
        # a() will return fn(original_value) and b() will return fn(new_value).
        # We can use this to change calctup as we go.
        new_inputtup = list(inputtup)
        new_calctup = list(calctup)
        items = tldict.empty()
        srcget = plan._source_lookup
        updates = holdlazy(updates)
        for (k,v) in updates.items():
            iidx = sources[k]
            lv = v if isinstance(v, lazy) else lazy(identfn, v)
            new_inputtup[iidx] = lv
            calc_updates.update(dependants[k].calcs)
            # Create a new plandict item for the new input value.
            src = valsources[k]
            if isinstance(src, tuple):
                # This input gets filtered, so we need a lazy lookup using a
                # lambda that makes a closure over the new_calctup symbol,
                # which will get updated as we go.
                lv = lazy(
                    lambda src: srcget(new_inputtup, new_calctup, src),
                    src)
            items[k] = lv
        new_inputtup = tuple(new_inputtup)
        output_updates = set()
        for cidx in calc_updates:
            calc = calcdata.calcs[cidx]
            output_updates.update(calc.outputs)
            new_calctup[cidx] = lazy(
                lambda c,a: plan._call_calc(new_inputtup, new_calctup, c, a),
                calc,
                calcdata.args[cidx])
        new_calctup = tuple(new_calctup)
        outputs = self.outputs
        for k in output_updates:
            if k not in outputs:  # Skip the internal/translated outputs.
                continue
            src = self.valsources[k]
            items[k] = lazy(srcget, new_inputtup, new_calctup, src)
        return (new_inputtup, new_calctup, items.persistent())
    @staticmethod
    def _make_srcs_args(names, calcs, params):
        args = []
        srcs = {k:ii for (ii,k) in enumerate(params)}
        for (cidx,(nm,c)) in enumerate(zip(names, calcs)):
            c = to_calc(c)
            # Wire up the inputs/arguments:
            a = []
            for k in c.inputs:
                a.append(srcs[k])
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
            s = set(
                (u1,v2)
                for (u1,v1) in clos
                for (u2,v2) in clos
                if u2 == v1
                if u1 != v2)
            if clos.issuperset(s):
                break
            clos |= s
        res = defaultdict(lambda:set())
        for (u,v) in clos:
            res[u].add(v)
        return res
    # Construction ------------------------------------------------------------
    __slots__ = (
        'inputs', 'outputs', 'defaults', 'requirements',
        'input_docs', 'output_docs', 'docstr',
        'calcdata', 'valsources', 'dependants')
    def __new__(cls, *args, **kwargs):
        # We overload new just to parse the input arguments and convert any
        # values into calc objects. We then pass these down to pdict.
        calcs = {}
        nargs = len(args)
        if nargs == 1:
            inplan = args[0]
            if isinstance(inplan, Mapping):  # (plan is a pdict/map of calcs)
                calcs.update(inplan)
            else:
                raise TypeError(
                    f"x in plan(x) must be a Mapping; found {type(inplan)}")
        elif nargs > 1:
            raise ValueError(
                f"plan expects 0 or 1 positional arguments; found {nargs}")
        calcs.update(kwargs)
        for (k,v) in calcs.items():
            u = to_calc(v)
            calcs[k] = u
        return pdict.__new__(cls, calcs)
    def __init__(self, *args, **kwargs):
        # We ignore the arguments because they are handled by __new__.
        # We can start by gathering up the calcs that deal with each of the
        # plan's values. The val2calc dict maps each value name in the plan to
        # a tuple of (output, filter, input) calcs that process the value. The
        # output calcs are those that produce the value as an output but don't
        # requie it as an input; the filter calcs are those that require the
        # value as an input and that produce the value as an output; the input
        # calcs are those that require the calc as an input but don't produce
        # it as output.
        val2calc = defaultdict(lambda:([],[],[]))
        filters = set()
        params = []
        reqs = tset()
        for (nm,c) in self.items():
            c = to_calc(c)
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
        filts = tset(sorted(
            filters, key=lambda
            f:-len(to_calc(self[f]).inputs)))
        filters = pset(filts)
        is_ready = lambda f: to_calc(self[f]).inputs <= inputs
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
                        f"unreachable calcs: {tuple(calcs)}; this is likely"
                        f" due to a circular dependency")
            else:
                filts.discard(nextcalc)
            # We have a next calculation in the order, so we add it.
            calcorder.append(nextcalc)
            c = to_calc(self[nextcalc])
            inputs.addall(c.outputs)
            calcs.discard(nextcalc)
            # While we're going through the calcs in order, we process docs:
            for (inp,doc) in c.input_docs.items():
                if not doc:
                    continue
                lns = doc.split('\n')
                nameln = lns[0]
                if ':' in nameln:
                    tag = ' :' + ':'.join(nameln.split(':')[1:])
                else:
                    tag = ''
                doc = reindent(
                    '\n'.join(lns[1:]), 4,
                    skip_first=False, final_endline=False)
                doc = f"    **``{nextcalc}``** input: ``{inp}``{tag}  \n{doc}"
                if inp in params:
                    input_docs[inp].append(doc)
                else:
                    output_docs[inp].append(doc)
            for (out,doc) in c.output_docs.items():
                if not doc:
                    continue
                lns = doc.split('\n')
                nameln = lns[0]
                if ':' in nameln:
                    tag = ' :' + ':'.join(nameln.split(':')[1:])
                else:
                    tag = ''
                doc = reindent(
                    '\n'.join(lns[1:]), 4,
                    skip_first=False, final_endline=False)
                doc = f"    **``{nextcalc}``** output: ``{out}``{tag}  \n{doc}"
                output_docs[out].append(doc)
        doc_connectfn = lambda v: '\n\n'.join(v)
        input_docs = pdict(valmap(doc_connectfn, input_docs))
        output_docs = pdict(valmap(doc_connectfn, output_docs))
        ncalcs = len(calcorder)
        calcidx = pdict(zip(calcorder, range(ncalcs)))
        outputs = inputs
        outputs -= params
        outputs = outputs.persistent()
        inputs = params
        # Before we move on to the calculation graph, let's make the docstring.
        # the 12 here must match the indentation of the docstring that
        # follows.
        code_indent = 12
        code_head = ' ' * code_indent
        sep = '\n' + code_head
        # The calculation substring is the most complex part; we make it
        # out of the list of calculations and their inputs/outputs.
        calcstr = []
        for (name,c) in self.items():
            cname = c.base_function if c.name is None else c.name
            calcstr.append(f"* ``{name}``: ``{cname}``  ")
            if len(inputs) == 0:
                inps = "None"
            else:
                inps = textwrap.wrap('``'+'``, ``'.join(c.inputs)+'``', 68)
                inps = (' '*11).join(inps)
            calcstr.append(f"  Inputs:  {inps}  ")
            if len(outputs) == 0:
                outs = "None"
            else:
                outs = textwrap.wrap('``'+'``, ``'.join(c.outputs)+'``', 68)
                outs = (' '*11).join(outs)
            calcstr.append(f"  Outputs: {outs}  ")
        calcstr = '\n'.join(calcstr)
        calcstr = reindent(
            calcstr, code_indent + 1,
            skip_first=False,
            final_endline=False)
        calcstr = calcstr[code_indent + 1:]
        # Make up the input and output strings too.
        inputstr = '\n'.join(
            [reindent(
                k + '\n' + s, code_indent,
                skip_first=False, final_endline=False)
             for (k,s) in input_docs.items()])
        outputstr = '\n'.join(
            [reindent(
                k + '\n' + s, code_indent,
                skip_first=False, final_endline=False)
             for (k,s) in output_docs.items()])
        if len(inputstr) > 0:
            inputstr = (
                f"{sep}Inputs{sep}"
                f"------\n"
                f"{inputstr}{sep}")
        if len(outputstr) > 0:
            outputstr = (
                f"{sep}Outputs{sep}"
                f"-------\n"
                f"{outputstr}{sep}")
        docstr = f"""An ``immlib.plan`` object for a set of calculations.

            This documentation was generated automatically from the docstrings
            of the individual ``immlib.calc`` objects that make up this plan.

            This plan contains the following calculations:
             {calcstr}
            {inputstr}{outputstr}"""
        docstr = reindent(docstr, 0, final_endline=False)
        object.__setattr__(self, 'docstr', docstr)
        object.__setattr__(self, '__doc__', docstr)
        # We now have a calculation ordering that we can use to turn the plan's
        # filtered values into sequential values.  For example, if the variable
        # 'x' is filtered through calculations 'f', 'g', and 'h', in that
        # order, then we update the inputs/outputs of the functions to force an
        # ordering. First, the initial parameter will be renamed to 'x.', then
        # f is changed to take 'x.' as an input in place of 'x' and to produce
        # 'x.f'. Then g is changed so that it takes 'x.f' instead of x and
        # produces 'x.g'. Then h is changed so that it takes 'x.h' as instead
        # of 'x' and produces the output 'x'.  The actual internal names don't
        # use periods (they stay as valid variable names that are potentially
        # randomly chosen using the _find_trname staticmethod).
        names = tuple(calcorder)
        calcs = tuple(self[k] for k in calcorder)
        if len(filters) == 0:
            # We don't actually have any filters to put in order, so we have
            # the straightforward job of wiring things up as-is. We already
            # know that there aren't any cycles in the graph (they would have
            # appeared earlier).
            (srcs, args) = plan._make_srcs_args(names, calcs, params)
            calcdata = plan.CalcData(names, calcs, args, srcs, calcidx)
            valsources = srcs
            tr = {}
            itr = {}
        else:
            # We need to do two things: (1) make a plan out of the unfiltered
            # calcs (i.e., separate inputs and outputs of each filter calc into
            # different variables and link them up across calculations), and
            # (2) make a translation for the output values.
            tr = {}
            itr = {}
            valnames = set(val2calc.keys())
            new_calcs = []
            for (nm,c) in zip(names, calcs):
                c = to_calc(c)
                filts = c.inputs & set(c.outputs)
                calctr = {}
                for f in filts:
                    k = tr.get(f, f)
                    new_k = plan._find_trname(valnames, k, nm)
                    valnames.add(new_k)
                    tr[f] = new_k
                    itr[new_k] = f
                    calctr[f] = (k, new_k)
                new_calcs.append(c.rename_keys(tr, calctr))
            (srcs, args) = plan._make_srcs_args(names, new_calcs, params)
            new_calcs = tuple(new_calcs)
            calcdata = plan.CalcData(names, new_calcs, args, srcs, calcidx)
            valsrcs = tdict()
            for k in valnames:
                valsrcs[k] = srcs[tr.get(k, k)]
            valsources = valsrcs.persistent()
        # Go through the calc ordering and find the earliest default value for
        # each of the keys.
        defaults = rmerge(
            *(c.defaults for c in map(to_calc, calcs) if c.defaults))
        # Note the requirements.
        requirements = pset(reqs)
        # One final thing we need to do is to make the dependants graph; this
        # is basically the graph of calculations and outputs that need to be
        # updated / reset any time a parameter is changed.
        depset = set()
        for (cidx,c) in enumerate(calcdata.calcs):
            c = to_calc(c)
            for k in c.inputs:
                depset.add((k, cidx))
            for k in c.outputs:
                depset.add((cidx, k))
        depgraph = plan._transitive_closure(depset)
        deps = tdict()
        for k in inputs:
            odeps = []
            cdeps = []
            for d in depgraph[k]:
                if isinstance(d, str):
                    odeps.append(d)
                else:
                    cdeps.append(d)
            # Translate to original key name (not dep key)
            k = itr.get(k, k)
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
    # Methods -----------------------------------------------------------------
    def filtercall(self, *args, **kwargs):
        """Calls the plan object, but filters out args that aren't in the plan.

        ``plan_obj.filtercall(dict1, dict2, ..., k1=v1, k2=v2, ...)`` is
        equivalent to ``plan_obj(dict1, dict2, ..., k1=v1, k2=v2, ...)`` except
        that any keys in the argument list to ``filtercall`` that aren't in the
        parameter list of ``plan_obj`` are automatically filtered out.
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
        n = len(self.calcdata.calcs)
        m = len(self.inputs)
        return f"plan(<{n} calcs>, <{m} params>)"
    def __repr__(self):
        n = len(self.calcdata.calcs)
        m = len(self.inputs)
        return f"plan(<{n} calcs>, <{m} params>)"
@docwrap
def is_plan(arg):
    """Determines if an object is a ``plan`` instance.

    ``is_plan(x)`` returns ``True`` if ``x`` is a calculation ``plan`` and
    ``False`` otherwise.
    """
    return isinstance(arg, plan)


# plandict ####################################################################

class plandict(ldict):
    """A persistent dict type that manages the outputs of executing a plan.

    ``plandict(plan, params)`` instantiates a plan object with the given
    dict-like object of parameters, ``params``.

    ``plandict(plan, params, k1=v1, k2=v2, ...)`` additional merges all keyword
    arguments into parameters.

    ``plandict(plan, k1=v1, k2=v2, ...)`` uses only the keyword arguments as
    the plan parameters.

    Note that ``plandict(plan, args...)`` is equivalent to ``plan(args...)``.

    ``plandict`` is a subclass of ``lazydict``, but it has some unique
    behavior, primarily in that only the parameters of a ``plandict`` may be
    updated; the rest of the items are consequences of the plan and parameter.

    Parameters
    ----------
    plan : plan
        The ``plan`` object that is to be instantiated.
    *params : dict-like, optional
        The dict-like object of the parameters of the ``plan``. All and only
        ``plan`` parameters must be provided, after the ``params`` argument is
        merged with the ``kwargs`` options. This may be a ``lazydict``, and
        this dict's laziness is respected as much as possible.
    **kwargs : optional keywords
        Optional keywords that are merged into ``params`` to form the set of
        parameters for the plan.
    
    Attributes
    ----------
    plan : immlib.plan
        The plan object on which this plandict is based or alternatively a
        ``plandict`` or object to copy.
    inputs : pdict
        The parameters that fulfill the plan. Note that these are the only keys
        in the ``plandict`` that can be updated using methods like ``set`` and
        ``setdefault``.
    """
    __slots__ = ('plan', 'inputs', '_calcdata', '_inputdata')
    def __new__(cls, *args, **kwargs):
        # There are two valid ways to call plandict(): plandict(planobj,
        # params) and plandict(plandictobj, new_params). We call different
        # classmethods for each version.
        if len(args) == 0:
            raise TypeError(
                "plandict() requires 1 argument that is a plan or plandict")
        (obj, args) = (args[0], args[1:])
        if is_plan(obj):
            pd = cls._new_from_plan(obj, *args, **kwargs)
        elif isinstance(obj, (plandict, tplandict)):
            pd = cls._new_from_plandict(obj, *args, **kwargs)
        else:
            raise TypeError(
                "plandict(obj, ...) requires that obj be a plan or plandict")
        return pd
    @classmethod
    def _new_plandict(cls, items, plan, params, calctup, inputtup):
        self = ldict.__new__(cls, items)
        # params needs to be a pdict (not a tdict)
        if isinstance(params, tdict):
            params = params.persistent()
        elif not isinstance(params, pdict):
            raise ValueError("plandict received non-pdict inputs")
        # And set our special member-values.
        object.__setattr__(self, 'plan', plan)
        object.__setattr__(self, 'inputs', params)
        object.__setattr__(self, '_calcdata', calctup)
        object.__setattr__(self, '_inputdata', inputtup)
        # At this point, the object should be entirely initialized, so we can
        # go ahead and run its required calculations.
        calcdata = plan.calcdata
        for r in plan.requirements:
            cidx = calcdata.index[r]
            calctup[cidx]()
        # That's all; just return the object.
        return self
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
        pparams = holdlazy(params)
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
            src = plan.valsources[k]
            # If this input gets filtered, we need a lazy lookup:
            if isinstance(src, tuple):
                v = lazy(plan._source_lookup, inputtup, calctup, src)
            else:
                v = pparams[k]
            items[k] = v
        for k in plan.outputs:
            src = plan.valsources[k]
            items[k] = lazy(plan._source_lookup, inputtup, calctup, src)
        # We can now make and return the object (this also runs requirements).
        return cls._new_plandict(items, plan, params, calctup, inputtup)
    @classmethod
    def _new_from_plandict(cls, pd, *args, **kwargs):
        plan = pd.plan
        calcdata = plan.calcdata
        if len(args) == 0 and len(kwargs) == 0:
            if isinstance(pd, plandict):
                return pd
            else:
                return cls._new_plandict(
                    pd, plan, pd.inputs, pd._calcdata, pd._inputdata)
        # First, merge from left-to-right, respecting laziness.
        param_updates = merge(*args, **kwargs)
        # There must only be parameters here.
        planins = plan.inputs
        if any(k not in planins for k in param_updates.keys()):
            extras = set(params_updates.keys()) - plan.inputs
            raise ValueError(f"unrecognized inputs: {tuple(extras)}")
        # Make a new inputtup, calctup, and updates to the items dict.
        if len(param_updates) == 0:
            items = pd
            inputtup = pd._inputdata
            calctup = pd._calcdata
        else:
            (inputtup, calctup, items) = plan._update_dictdata(
                pd._inputdata,
                pd._calcdata,
                param_updates)
            items = merge(pd, items)
        inputs = merge(pd.inputs, param_updates)
        return cls._new_plandict(items, plan, inputs, calctup, inputtup)
    def set(self, k, v):
        return plandict(self, {k:v})
    def setdefault(self, k, v=None):
        # All possible keys to set are already set in a plandict, so just pass
        # this through to set.
        return self.set(k, v)
    def delete(self, k):
        raise TypeError("cannot delete from a plandict")
    def transient(self):
        return tplandict(self)
    def __hash__(self):
        return hash(self.inputs) + hash(self.plan)

class tplandict(tldict):
    """A transient dict type that follows an ``immlib`` plan.

    See ``immlib.plandict`` and ``immlib.plan`` for more information about
    ``immlib`` plans and workflows. The ``tplandict`` type is a transient
    counterpart to the persistent ``plandict`` type. A ``tplandict`` object can
    be created from a ``plandict`` object ``pd`` using either the syntax ``td =
    tplandict(pd)`` or ``td = pd.transient()``. The keys corresponding to the
    inputs of the plan can be changed in the resulting ``tplandict`` object,
    and the downstream outputs of the plan will be automatically updated as
    these are changed. A ``plandict`` can be recreated using either the syntax
    ``plandict(td)`` or ``td.transient()``.

    Parameters
    ----------
    plan : plan
        The ``plan`` object that is to be instantiated or alternatively a
        ``plandict`` or ``tplandict`` object to copy.
    *params : dict-like, optional
        The dict-like object of the parameters of the ``plan``. All and only
        ``plan`` parameters must be provided, after the ``params`` argument is
        merged with the ``kwargs`` options. This may be a ``lazydict``, and
         this dict's laziness is respected as much as possible.
    **kwargs : optional keywords
        Optional keywords that are merged into ``params`` to form the set of
        parameters for the plan.
    
    Attributes
    ----------
    plan : plan
        The plan object on which this tplandict is based.
    inputs : pdict
        The parameters that fulfill the plan. Note that these are the only keys
        in the ``tplandict`` that can be updated directly.

    """
    __slots__ = ('plan', 'inputs', '_calcdata', '_inputdata')
    def __new__(cls, *args, **kwargs):
        # There are two valid ways to call plandict(): plandict(planobj,
        # params) and plandict(plandictobj, new_params). We call different
        # classmethods for each version.
        if len(args) == 0:
            raise TypeError(
                "tplandict() requires 1 argument that is a plan or plandict")
        (obj, args) = (args[0], args[1:])
        if is_plan(obj):
            pd = plandict(obj, *args, **kwargs)
        elif isinstance(obj, tplandict):
            pd = obj.persistent()
        else:
            pd = obj
        if not isinstance(pd, plandict):
            raise TypeError(
                f"tplandict(obj, ...) requires that obj be a plan or plandict;"
                f" god {type(obj)}")
        plan = pd.plan
        calcdata = plan.calcdata
        # First, merge from left-to-right, respecting laziness.
        param_updates = merge(*args, **kwargs)
        # There must only be parameters here.
        planins = plan.inputs
        if any(k not in planins for k in param_updates.keys()):
            extras = set(params_updates.keys()) - plan.inputs
            raise ValueError(f"unrecognized inputs: {tuple(extras)}")
        # Make a new inputtup, calctup, and update the items dict.
        (inputtup, calctup, items) = plan._update_dictdata(
            pd._inputdata,
            pd._calcdata,
            param_updates)
        inputs = pd.inputs
        # Inputs stays a pdict because we don't want to allow the user to
        # update them directly.
        inputs = inputs.update(param_updates)
        items = merge(pd, items)
        return cls._new_tplandict(items, plan, inputs, calctup, inputtup)
    @classmethod
    def _new_tplandict(cls, items, plan, params, calctup, inputtup):
        # We make a new tplandict, but we don't initialize items here--we do
        # that below.
        self = tldict.__new__(cls)
        # params needs to be a tdict (not a pdict)
        if isinstance(params, tdict):
            params = params.persistent()
        elif not isinstance(params, pdict):
            raise ValueError("tplandict received non-pdict inputs")
        # And set our special member-values.
        object.__setattr__(self, 'plan', plan)
        object.__setattr__(self, 'inputs', params)
        object.__setattr__(self, '_calcdata', calctup)
        object.__setattr__(self, '_inputdata', inputtup)
        # Now we add items.
        for (k,v) in holdlazy(items).items():
            tldict.__setitem__(self, k, v)
        # At this point, the object should be entirely initialized, so we can
        # go ahead and run its required calculations.
        calcdata = plan.calcdata
        for r in plan.requirements:
            cidx = calcdata.index[r]
            calctup[cidx]()
        # That's all; just return the object.
        return self
    def __setitem__(self, k, v):
        vv = holdlazy(self.inputs).get(k, self)
        if vv is self:
            raise ValueError(f"cannot set non-input key in tplandict: {k}")
        elif vv is v:
            # No change; just return.
            return
        plan = self.plan
        # First, we reset any calculation that relies on this value and update
        # the calctup / inputtup and items.
        (inputtup, calctup, items) = plan._update_dictdata(
            self._inputdata,
            self._calcdata,
            {k:v})
        # Next, we next need to reset the downstream items.
        for kk in items.keys():
            tldict.__setitem__(self, kk, items.getlazy(kk))
        # Before we return, we change the object's values.
        object.__setattr__(self, 'inputs', self.inputs.set(k, v))
        object.__setattr__(self, '_inputdata', inputtup)
        object.__setattr__(self, '_calcdata', calctup)
        # Finally, we need to rerun any requirements that were changed.
        calcdataidx = plan.calcdata.index
        for r in plan.requirements:
            cidx = calcdataidx[r]
            calctup[cidx]()
    def __delitem__(self, k):
        raise TypeError("cannot delete items from tplandict objects")
    def setdefault(self, k, default=None, /):
        # All possible keys to set are already set in a plandict, so just
        # return the current value of key k
        return self[k]
    def persistent(self):
        return plandict(self)
@docwrap
def is_plandict(arg):
    """Determines if an object is a ``plandict`` instance.

    ``is_plandict(x)`` returns ``True`` if ``x`` is a ``plandict`` object and
    ``False`` otherwise.
    """
    return isinstance(arg, plandict)
@docwrap
def is_tplandict(arg):
    """Determines if an object is a ``tplandict`` instance.

    ``is_tplandict(x)`` returns ``True`` if ``x`` is a ``tplandict`` object and
    ``False`` otherwise.
    """
    return isinstance(arg, tplandict)
