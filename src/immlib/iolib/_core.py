# -*- coding: utf-8 -*-
################################################################################
# immlib/iolib/_core.py

"""Input and output operations via the load() and save() functions."""


# Dependencies #################################################################

import os, io, gzip, numbers
from pathlib import Path

import numpy as np
from pcollections import pdict, plist, pset, ldict, lazy

from ..doc        import docwrap
from ..util       import is_str, is_amap, is_aseq
from ..pathlib    import path, is_path, like_path


# Format and Formatter #########################################################

# The Format class, an inset class for managing the various stream formats.
class Format:
    """Manages the details of a stream format for use with `save` and `load`.

    `Format` objects are typically created with the `save.register` or
    `load.register` methods.  This class should be handled internally but should
    not be instantiated outside of the `Save` or `Load` classes.

    `Format` objects can be treated as save functions: the `__call__` method
    on a `Format` object accepts a stream or path-like object representing
    the destination to which the format should be written, an object to save
    in the format, and any optional arguments understood by the type.
    """
    def __init__(self, name, function, *suffixes, mode='b', gzip_suffix=None):
        self.name = name
        self.function = function
        self.suffixes = []
        for suff in suffixes:
            if is_str(suff):
                suff = (suff,)
            elif all(is_str, suff):
                suff = tuple(suff)
            else:
                raise ValueError(f"invalid path suffix: {suff}")
            self.suffixes.append(suff)
        if mode != 'b' and mode != 't':
            raise ValueError("format mode must be 't' or 'b'")
        self.mode = mode
        if gzip_suffix is None:
            gzip_suffix = ()
        elif is_str(gzip_suffix):
            gzip_suffix = ((gzip_suffix,),)
        elif is_aseq(gzip_suffix):
            if all(is_str(s) for s in gzip_suffix):
                gzip_suffix = tuple((s,) for s in gzip_suffix)
            else:
                g = []
                for suff in gzip_suffix:
                    if is_str(suff):
                        g.append((suff,))
                    elif is_aseq(suff) and all(is_str(s) for s in suff):
                        g.append(tuple(suff))
                    else:
                        raise ValueError(f"invalid gzip suffix: {suff}")
                gzip_suffix = tuple(g)
        else:
            raise ValueError(f"invalid gzip suffix: {gzip_suffix}")
        self.gzip_suffix = gzip_suffix
        # We also want to change the documentation.
        self.__doc__ = self.function.__doc__
    def __call__(self, stream, stream_mode, *args, **kwargs):
        # If dest isn't a stream, we need to open it as a path.
        if isinstance(stream, io.IOBase):
            return self.function(stream, *args, **kwargs)
        else:
            p = path(stream)
            with p.open(stream_mode + self.mode) as s:
                return self.function(s, *args, **kwargs)
class Formatter:
    """Base class for the Save and Load classes.

    Handles common operations between saving and loading systems.
    """
    # The mode use for opening streams from paths ('r' or 'w' typically).
    stream_mode = ''
    # Data/Details of the Save/Load classes.
    __slots__ = ("formats", "_format_by_suffix")
    # Constructor.
    def __init__(self, template=None):
        self.formats = {}
        self._format_by_suffix = {}
        if template is not None:
            cls = type(self)
            if isinstance(template, cls):
                formats = template.formats
            elif is_amap(template):
                formats = templates
            else:
                raise TypeError(f"template must be a {cls} object or a mapping")
            for (k,format) in formats.items():
                if not isinstance(format, Format):
                    raise TypeError(f"template contains non-format named {k}")
                self.register(format)
    def deduce_format(self, arg, ignore_gz=True):
        """Deduces the file format for a given path or suffix.

        `save.deduce_format(arg)` deduces the format implied by `arg`. The
        argument may be a path, a string representing a path, or a sequence of
        strings representing the suffixes of the path. If the final suffix is
        `.gz` then this suffix is ignored.
        """
        if is_str(arg):
            suffixes = path(arg).suffixes
        elif is_path(arg):
            suffixes = arg.suffixes
        elif is_aseq(arg) and all(is_str(s) for s in arg):
            suffixes = arg
        else:
            raise ValueError(
                f"deduce_format given invalid argument of type {type(arg)}")
        suff = tuple(suffixes)
        if ignore_gz and suff[-1] == '.gz':
            suff = suff[:-1]
        while suff:
            format = self._format_by_suffix.get(suff)
            if format:
                return format
            suff = suff[1:]
        # No format found.
        return None
    @staticmethod
    def _expandall(p):
        p = path(p)
        if isinstance(p, Path):
            p = Path(os.path.expanduser(os.path.expandvars(os.fspath(p))))
        return p
    def _call(self, target, format, gzip, /, *args, **kwargs):
        if isinstance(target, io.IOBase):
            target_path = None
            target_stream = target
            if gzip is Ellipsis:
                gzip = False
        else:
            target_path = path(target)
            if isinstance(target_path, Path):
                target_path = Formatter._expandall(target_path)
            target_stream = None
        if format is None:
            if target_path is None:
                raise ValueError("format can't be None when target is a stream")
            # We can walk through the formats and attempt to deduce the correct
            # format from the ending of the dest_str.
            format = self.deduce_format(target_path)
            if not format:
                raise ValueError(f"format can't be deduced from path: {target}")
        # At this point we have a format name or we've raised an error.
        if is_str(format):
            fmt = self.formats.get(format)
            if fmt is None:
                raise ValueError(f"format not recognized: {format}")
            else:
                format = fmt
        elif not isinstance(format, Format):
            raise TypeError("format arg must be a format name or Format object")
        # Now we know the format and thus have a format function; if we were
        # given a path instead of a stream, we need to open the path for the
        # formatting function. We also need to handle gzipping--if the format is
        # a gzip format then gzip will be True at this point.
        if gzip is Ellipsis:
            suff = tuple(target_path.suffixes)
            gzip = (target_path.suffix == '.gz') or suff in format.gzip_suffix
        if gzip is True:
            if target_stream:
                with self._gzip(target_stream, format.mode) as act_stream:
                    r = format.function(act_stream, *args, **kwargs)
                    return (r, target_stream)
            else:
                with target_path.open(self.stream_mode + 'b') as stream:
                    with self._gzip(stream, format.mode) as act_stream:
                        r = format.function(act_stream, *args, **kwargs)
                        return (r, target_path)
        elif target_stream:
            r = format.function(target_stream, *args, **kwargs)
            return (r, target_stream)
        else:
            with target_path.open(self.stream_mode + format.mode) as stream:
                r = format.function(stream, *args, **kwargs)
            return (r, target_path)
    def _gzip(self, stream, mode,
              compresslevel=9,
              encoding=None,
              errors=None,
              newline=None):
        return gzip.open(
            stream,
            mode=(self.stream_mode + mode),
            compresslevel=compresslevel,
            encoding=encoding,
            errors=errors,
            newline=newline)
    def register(self, name, /, *suffixes, mode='b', gzip_suffix=None):
        """Registers a format type with the `save`/`load` interface.

        The `immlib.save.register` method is intended to be used as a decorator:
        `@immlib.save.register(format_name, suffix1, suffix2...)` ensures that
        the function that follows it is registered as a format managed by the
        `immlib.save` system. The function that is decorated should be written
        to accept a stream object (though the decorator will ensure that it
        works when called with a path-like object or a stream). The function's
        signature must always match `f(output_stream, object_to_save, **opts)`
        where the `**opts` must be any set of named parameters. The return value
        of the function is ignored, but it should raise an error if the object
        cannot be saved.

        The `immlib.load.register` method is identical except that the function
        that follows is not passed an object to load and instead must return the
        object that is loaded from the given stream.

        Parameters
        ----------
        name : str
            The name of the format.
        *suffixes : strings or tuples of strings, optional
            Any number of suffixes that this format uses. Suffixes must be
            unique, and an error is raised if a requested suffix is already
            registered to another format. Each suffix may be a string such as
            `.tgz` or a tuple such as `('.tar', '.gz')`.
        mode : 't' or 'b', optional
            Whether any stream that is opened in order to save the file should
            be opened in text mode (`'t'`) or binary mode (`'b'`). The default
            is `'b'`.
        gzip_suffix : None or str or tuple of str
            A sequence of suffixes that indicate that the format is additionally
            encoded using gzip. Files that end with `.gz` are automatically
            interpreted as gzipped files by the `save` system, but if an
            additional file ending needs to be specified as a gzip format, such
            as the suffix `.npz` for gzipped-numpy files (equivalent to
            `.npy.gz`), then it should be specified in this option. If a single
            string is given (such as `'.npz'`), then it is interpreted as a
            single suffix (`[['.npz']]`); if a sequence of strings are given
            (`['.npz', '.nz']`), then they are interpreted as individual
            suffixes (`[['.npz'], ['.nz']]`). For compound suffixes, they must
            be elements of a sequence, e.g. `[['.json', '.gz']]` not `['.json',
            '.gz']`. Note, however, that `[['.json', '.gz']]` is automatically
            interpreted as a gzip file by virtue of its `.gz` suffix.
        """
        # There are actually two uses: (1) that described in the help above and
        # (2) save.register(format) where format is already a Format object. In
        # the latter case we aren't a decorator.
        if isinstance(name, Format):
            # We're in case 2, so we register this format specifically.
            format = name
            if format.name in self.formats:
                raise RuntimeError(f"format {format.name} already registered")
            suffs = tuple(format.suffixes) + format.gzip_suffix
            for suff in suffs:
                ex = self._format_by_suffix.get(suff)
                if ex is not None:
                    raise RuntimeError(
                        f"suffix {suff} already mapped to format {ex.name}")
            # We pass the criteria; go ahead and add this format.
            self.formats[format.name] = format
            for suff in suffs:
                self._format_by_suffix[suff] = format
            # Return the format itself.
            return format
        else:
            # We're in case 1, so we return a decorator for the function.
            def _formatter_register_dec(f):
                if isinstance(f, Format):
                    f = f.function
                format = Format(
                    name, f, *suffixes,
                    mode=mode,
                    gzip_suffix=gzip_suffix)
                return self.register(format)
            return _formatter_register_dec
        # That's it for the register function.
    def unregister(self, name, *, error_on_missing=False):
        """Unregisters the format with the given name from the save/load system.

        The format with the given name is unregistered from the `immlib.save`
        system, and the `Format` object that is removed is returned. If the
        format is not found, then `None` is returned, but no error is raised by
        default.

        The `immlib.load.unregister` method works identically to the
        `immlib.save` version.

        Parameters
        ----------
        name : str
            The name of the format to unregister.
        error_on_missing : boolean, optional
            Whether to throw a `RuntimeError` if the named format is not found
            in the save manager. By default this is `False`.

        Returns
        -------
        Format or None
            The format object that is unregistered or `None` if the given name
            was not a registered format.

        Raises
        ------
        RuntimeError
            If the given name does not map to a format in the save manager and
            the `error_on_missing` option is `True`.
        """
        format = self.formats.get(name)
        if format is None:
            if error_on_missing:
                fnnm = type(self).__name__.lower()
                raise RuntimeError(
                    f"format {name} not found in immlib {fnnm} system")
            else:
                return None
        # Actually remove things:
        for suff in (format.suffixes + format.gzip_suffix):
            del self._format_by_suffix[suff]
        del self.formats[name]
        return format
    def copy(self):
        """Returns a copy of the given save manager.

        `immlib.save.copy()` can be used to return a copy of the save manager,
        for instances where a single save manager is not ideal.
        """
        cls = type(self)
        return cls(self)


# Save #########################################################################

class Save(Formatter):
    """Saves a Python object to a stream or path then returns the stream/path.

    `immlib.save(path, object, format)` saves the given `object` to the given
    `path` using the named file `format` and returns the path on success or
    rasies an error on failure. If the format can be deduced from the path
    suffix, then it may be omitted.

    `immlib.save(stream, object, format)` writes the given `object` to the given
    stream object using the named `format` and returns the stream. 
    
    In fact, `immlib.save` is an object of the `immlib.pathlib.Save` type that
    primarily behaves like a function. The `immlib.save.register` and
    `immlib.save.unregister` methods can be used to add understood formats.

    Parameters
    ----------
    dest : path-like or stream
        The output destination to which the object is to be saved. This may be
        either a path-like object or a writeable `IOBase` stream.
    obj : object
        Any object that can be saved in the given format.
    format : str or None, optional
        If provided, `format` must be a string that names a format that has been
        previously registered with `save` using the `save.register` method. If
        `format` is not provided or is `None` (the default), then an attempt is
        made to deduce the format using the suffix of the `dest` argument,
        assuming that `dest` is a path and not a stream. If `dest` is a stream
        or if the suffix does not indicate a specific format, then a
        `ValueError` is raised.
    **kwargs
        Any additional parameters are passed to the registered export function.

    Returns
    -------
    path-like or stream
        The destination stream or path. If the `dest` argument is a path name,
        then a path object is returned instead of the path name.

    Raises
    ------
    TypeError
        If the destination is not a stream or path-like object or if the format
        is not a string.
    ValueError
        If the format is not recognized or if it cannot be deduced from the
        destination object.
    """
    stream_mode = 'w'
    def __call__(self, dest, obj, format=None, /, gzip=Ellipsis, **kwargs):
        (loadret, saveret) = self._call(dest, format, gzip, obj, **kwargs)
        return saveret

# We can go ahead and declare a single global save object for the immlib
# library.  We will later use this object to register various basic format
# types.
save = Save()

# Register a few basic file formats.
@save.register('str', mode='t')
def save_str(stream, obj, append_nl=False):
    """Saves `str(obj)` to the given stream."""
    stream.write(str(obj))
    if append_nl:
        stream.write('\n')
@save.register('bytes', mode='t')
def save_bytes(stream, obj):
    """Saves `bytes(obj)` to the given stream."""
    stream.write(bytes(obj))
@save.register('repr', mode='t')
def save_repr(stream, obj, append_nl=False):
    """Saves `repr(obj)` to the given stream."""
    stream.write(repr(obj))
    if append_nl:
        stream.write('\n')
@save.register('text', '.txt', '.text', mode='t')
def save_text(stream, lines, append_nls=False):
    """Saves a blob of text or a sequence of lines to a stream or path.

    Parameters
    ----------
    stream : stream or path-like
        The stream or path to which to write the lines.
    lines : string or sequence of strings
        The text that should be written to the file or stream.
    append_nls : boolean, optional
        Whether to append newlines to the end of each line. If `False` (the
        default), then the lines are written as they are; otherwise, a newline
        character is appended after each line in the `lines` argument.
    """
    if is_str(lines):
        lines = (lines,)
    if not is_aseq(lines):
        raise TypeError("lines must be text or a sequence of text objects")
    if append_nls:
        for ln in lines:
            stream.write(ln)
            stream.write('\n')
    else:
        for ln in lines:
            stream.write(ln)
@save.register('pickle', '.pickle', '.pkl', '.pcl', mode='b')
def save_pickle(stream, obj, protocol=None, **kwargs):
    """Saves a pickled object to a destination path or stream.

    All keyword options are forwarded to the `pickle.dump` function. The
    `protocol` keyword option is provided as the `protocol` positional argument
    to the `pickle.dump` function.
    """
    import pickle
    pickle.dump(obj, stream, protocol, **kwargs)
@save.register('numpy', '.npy', '.np', '.numpy', mode='b', gzip_suffix='.npz')
def save_numpy(stream, obj, **kwargs):
    """Saves a numpy object to a destination path or stream.

    All keyword options are forwarded to the `numpy.save` function.
    """
    import numpy as np
    np.save(stream, obj, **kwargs)
def json_default(obj):
    """Converts an object to a json-formattable object or raises TypeError.
    """
    if is_str(obj) or obj is None or obj is True or obj is False:
        return obj
    elif isinstance(obj, numbers.Integral):
        return int(obj)
    elif isinstance(obj, numbers.Real):
        return float(obj)
    elif is_amap(obj):
        return dict(obj)
    elif is_aseq(obj):
        return list(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    #elif is_planobject(obj):
    #    cls = type(obj)
    #    return {
    #        '__plantype__': f'{cls.__module__}.{cls.__name__}',
    #        '__params__': dict(obj.__plandict__.params)}
    else:
        raise TypeError(type(obj))
@save.register('json', '.json', mode='t')
def save_json(stream, obj, /, default=json_default, **kwargs):
    """Saves an object as a JSON string or raises a TypeError if not possible.

    All keywords are passed along to the `json.dump` function. The `default`
    option uses the `immlib.iolib.json_default` function, which is different
    than the default used by `json.dump`, but all other options are unaltered.
    """
    import json
    json.dump(obj, stream, default=default, **kwargs)
def yaml_prepare(obj):
    """Returns a version of the argument that can be JSON/YAML serialized."""
    if is_str(obj) or obj is None or obj is True or obj is False:
        return obj
    elif isinstance(obj, numbers.Integral):
        return int(obj)
    elif isinstance(obj, numbers.Real):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif is_amap(obj):
        r = {}
        for (k,v) in obj.items():
            if not is_str(k):
                raise TypeError("JSON/YAML dict keys must be strings")
            r[k] = yaml_prepare(v)
        return r
    elif is_aseq(obj):
        return [yaml_prepare(u) for u in obj]
    else:
        raise TypeError(type(obj))
@save.register('yaml', '.yaml', mode='t')
def save_yaml(stream, obj, /, **kwargs):
    """Saves an object as a YAML string or raises a TypeError if not possible.

    All keywords are passed along to the `yaml.Dumper` object that is used for
    dumping `obj`. Note that unlike with the `yaml.dump` function, object are
    only serialized if they are JSON/YAML serializable; classes that are not
    mappings, sequences, numbers, strings, or booleans cannot be serialized
    using this function and instead raise TypeErrors.
    """
    import yaml
    yaml.dump(yaml_prepare(obj), stream, **kwargs)
@save.register('csv', '.csv', mode='t')
def save_csv(stream, obj, /, index=False, **kwargs):
    """Saves a pandas DataFrame to a CSV file.

    All options are passed along to `pandas.DataFrame.to_csv()`. The option
    `index` has the default value of `False` in this function.
    """
    import pandas
    obj = pandas.DataFrame(obj)
    obj.to_csv(stream, index=index, **kwargs)
@save.register('tsv', '.tsv', mode='t')
def save_tsv(stream, obj, /, sep="\t", index=False, **kwargs):
    """Saves a pandas DataFrame to a TSV file.

    All options are passed along to `pandas.DataFrame.to_csv()`. The option
    `index` has the default value of `False` in this function, and the option
    `sep` has the default value `"\t"`.
    """
    import pandas
    obj = pandas.DataFrame(obj)
    obj.to_csv(stream, sep=sep, index=index, **kwargs)


# Load #########################################################################

class Load(Formatter):
    """Loads a Python object from a stream or path then returns the object.

    `immlib.load(path, format)` loads a Python object from the given `path`
    using the named file `format` and returns the loaded object or rasies an
    error on failure. If the format can be deduced from the path suffix, then it
    may be omitted.

    `immlib.load(stream, format)` reads a Python object from the given stream
    object using the named `format` and returns the object. The format argument
    cannot be omitted when the first argument is a stream.
    
    In fact, `immlib.load` is an object of the `immlib.pathlib.Load` type that
    primarily behaves like a function. The `immlib.load.register` and
    `immlib.load.unregister` methods can be used to add understood formats.

    Parameters
    ----------
    source : path-like or stream
        The input source from which the object is to be loaded. This may be
        either a path-like object or a readable `IOBase` stream.
    format : str or None, optional
        If provided, `format` must be a string that names a format that has been
        previously registered with `load` using the `load.register` method. If
        `format` is not provided or is `None` (the default), then an attempt is
        made to deduce the format using the suffix of the `source` argument,
        assuming that `source` is a path and not a stream. If `source` is a
        stream or if the suffix does not indicate a specific format, then a
        `ValueError` is raised.
    **kwargs
        Any additional parameters are passed to the registered export function.

    Returns
    -------
    object
        The object that was loaded from the given stream or path.

    Raises
    ------
    TypeError
        If the input source is not a stream or path-like object or if the format
        is not a string.
    ValueError
        If the format is not recognized or if it cannot be deduced from the
        source.
    """
    stream_mode = 'r'
    def __call__(self, src, format=None, /, gzip=Ellipsis, **kwargs):
        # There's a special case for loading: if we're given a path, and it
        # refers to a directory, we load it as a lazy dictionary.
        if format is None:
            if not isinstance(src, io.IOBase):
                p = Formatter._expandall(src)
                if p.is_dir():
                    format = 'dir'
        if format == 'dir':
            return Load.from_dir(src)
        (loadret, saveret) = self._call(src, format, gzip, **kwargs)
        return loadret
    @staticmethod
    def from_dir(src, filter=None):
        """Loads a nested dictionary structure of a directory.

        `Load.from_dir(path)` returns `path` if `path` refers to a file.
        Alternatively, if `path` refers to a directory, this function returns a
        lazy dictionary whose keys are the names of the entries in the directory
        and whose values are the result of calling `Load.from_dir` on their
        paths.
        """
        src = path(src)
        if src.is_file():
            raise NotADirectoryError(src)
        if filter:
            d = {
                p.name: (
                    lazy(Load.from_dir, p, filter=filter) if p.is_dir() else p)
                for p in src.iterdir()
                if filter(p)}
        else:
            d = {
                p.name: (lazy(Load.from_dir, p) if p.is_dir() else p)
                for p in src.iterdir()}
        return ldict(d)

# We can go ahead and declare a single global save object for the immlib
# library.  We will later use this object to register various basic format
# types.
load = Load()

# Register a few basic file formats.
@load.register('str', mode='t')
def load_str(stream, strip_nl=False, size=-1):
    """Loads a string from the given source."""
    s = stream.read(size)
    if strip_nl:
        s = s.rstrip('\n')
    return s
@load.register('bytes', mode='b')
def load_bytes(stream, size=-1):
    """Loads a string from the given source."""
    return stream.read(size)
@load.register('repr', mode='t')
def load_repr(stream, size=-1):
    """Loads an object using the `ast.literal_eval` function."""
    from ast import literal_eval
    return literal_eval(stream.read(size))
@load.register('text', '.txt', '.text', mode='t')
def load_text(stream, strip_nls=True, size=-1):
    """Loads a blob of text or a sequence of lines from a stream or path.

    Parameters
    ----------
    stream : stream or path-like
        The stream or path from which to read the lines.
    strip_nls : boolean, optional
        Whether to strip newlines from the end of each line. If `False`, then
        the lines are returned as they are; if `True` (the default), any newline
        character is stripped from each line in the `lines` argument.
    """
    if strip_nls:
        return stream.read(size).splitlines()
    else:
        return stream.readlines(size)
@load.register('pickle', '.pickle', '.pkl', '.pcl', mode='b')
def load_pickle(stream, **kwargs):
    """Loads a pickled object from a path or stream and returns the object.

    All keyword options are forwarded to the `pickle.load` function.
    """
    import pickle
    return pickle.load(stream, **kwargs)
@load.register('numpy', '.npy', '.np', '.numpy', mode='b', gzip_suffix='.npz')
def load_numpy(stream, **kwargs):
    """Loads a numpy object from a path or stream and returns the object.

    All keyword options are forwarded to the `numpy.load` function.
    """
    import numpy as np
    return np.load(stream, **kwargs)
@load.register('json', '.json', mode='t')
def load_json(stream, /, **kwargs):
    """Loads an object from a JSON stream or path and returns the object.

    All keywords are passed along to the `json.load` function.
    """
    import json
    return json.load(stream, **kwargs)
@load.register('yaml', '.yaml', '.yml', mode='t')
def load_yaml(stream, /, safe=True):
    """Loads an object from a YAML stream or path and returns the object.

    The optional argument `safe` may be set to `False` to use unsafe YAML
    loading via the `yaml.load()` function.
    """
    import yaml
    if safe:
        return yaml.safe_load(stream)
    else:
        return yaml.load(stream)
@load.register('csv', '.csv', mode='t')
def load_csv(stream, /, sep=',', **kwargs):
    """Loads a pandas DataFrame from a CSV file.

    All options are passed along to `pandas.read_csv()`.
    """
    import pandas
    return pandas.read_csv(stream, sep=sep, **kwargs)
@load.register('tsv', '.tsv', mode='t')
def load_tsv(stream, /, sep="\t", **kwargs):
    """Loads a pandas DataFrame from a TSV file.

    All options are passed along to `pandas.read_csv()`.
    """
    import pandas
    return pandas.read_csv(stream, sep=sep, **kwargs)
