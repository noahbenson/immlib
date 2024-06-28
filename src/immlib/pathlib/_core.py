# -*- coding: utf-8 -*-
################################################################################
# immlib/pathlib/_core.py

# Dependencies #################################################################

from platform     import platform
from urllib.parse import urlparse
from collections  import namedtuple
from pathlib      import (Path, PurePath)
from os           import (PathLike, fspath)

from pcollections import (ldict, lazy)
from cloudpathlib import (
    CloudPath,
    S3Path, AzureBlobPath, GSPath,
    S3Client, AzureBlobClient, GSClient)

from ..doc        import docwrap
from ..util       import (is_str, is_amap, is_aseq, strstarts, strends)

from ._osf        import (OSFClient, OSFPath)
from ._cache      import CloudCachePath


# Global Values ################################################################

# The set of drives, if we are in windows; this is used by the path logic to
# automatically determine if a path like 'c:/a/b/c' is a valid path.
windows_drives = None
if 'windows' in platform().lower():
    from pathlib import WindowsPath
    from ctypes  import cdll
    _ord_a = ord('a')
    def is_windows_drive(letter):
        try:
            logical_drives = cdll.kernel32.GetLogicalDrives()
            bitno_letter = ord(letter.lower()) - _ord_a
            return bool((logical_drives >> bitno_letter) & 0x01)
        except Exception:
            return False
else:
    def is_windows_drive(letter):
        return False
is_windows_drive.__doc__ = \
    """Determines whether a given string represents a valid Windows drive.

    On non-windows platforms, `is_windows_drive` always returns `False`. It is
    additionally, possible that `is_windows_drive` will always return false for
    certain Windows systems. The function requires the the
    `ctypes.cdll.kernel32` object contain the `GetLogicalDrives` function and
    that the substring 'windows' appear in the `platform` package's
    `platform.platform().lower()` string.

    If the above requirements are met, then `is_windows_drive(letter)` will
    return `True` is `letter` is the letter of a valid drive and returns `False`
    otherwise. Case is ignored.
    """


# General Utilities ############################################################

@docwrap
def pathstr(obj):
    """Returns a string or bytes representation of a path.

    `pathstr(obj)` returns `obj` itself if `obj` is either a `str` or `bytes`
    object. If `obj` is a `CloudPath` object, then `str(obj)` is returned. 
    Otherwise, If `obj` is a `PathLike` object, then `os.fspath(obj)` is
    returned.
    """
    return str(obj) if isinstance(obj, CloudPath) else fspath(obj)


# OSFPath Functions ############################################################

@docwrap
def is_osfpath(obj):
    """Detects whether the input is an `OSFPath` object.

    `is_osfpath(obj)` returns `True` if `obj` is an instance of the `OSFPath`
    class and `False` otherwise.

    See also: `like_osfpath`

    Parameters
    ----------
    obj : object
        The object whose membership in the `OSFPath` class is to be determined.
    
    Returns
    -------
    boolean
        `True` if `obj` is an instance of `OSFPath` and `False` otherwise.
    """
    return isinstance(obj, OSFPath)
@docwrap
def like_osfpath(obj):
    """Detects whether the input can be converted into an `OSFPath` object.

    `like_osfpath(obj)` returns `True` if `obj` is an instance of the `OSFPath`
    class or is a string that forms a valid OSF path, and `False` otherwise.

    See also: `is_osfpath`

    Parameters
    ----------
    obj : object
        The object whose ability to be converted into an `OSFPath` instance is
        to be determined.
    
    Returns
    -------
    boolean
        `True` if `obj` is an instance of `OSFPath` or is a string that could be
        converted into an `OSFPath` and `False` otherwise.
    """
    if isinstance(obj, OSFPath):
        return True
    url = urlparse(pathstr(obj))
    return bool(url.scheme == 'osf' and url.netloc)
@docwrap
def osfpath(obj, *args,
            client=None,
            cache_path=Ellipsis,
            file_cache_mode=Ellipsis,
            mkdir_mode=Ellipsis,
            pagesize=Ellipsis,
            local_cache_dir=None):
    """Creates and returns an `OSFPath` representing an OSF.io repository.

    `osfpath(p)` creates and returns an `OSFPath` object, which is a type of
    `cloudpathlib.CloudPath` object, from the path or path-string `p`. If `p` is
    an `OSFPath`, then it is returned as-is. Otherwise `str(p)` is converted
    into an `OSFPath`; `str(p)` may start with `'osf://'` (not case-sensitive)
    or, if it does not have a scheme specifier, `'osf://'` will be prepended to
    it.

    `osfpath(p, a1, a2...)` converts `p` into an `OSFPath` then joins the `a1`,
    `a2`, etc. values to the end of the path and returns the joined path.

    OSF paths take the format `'osf://<project-ID>:<storage>/<path>'` where the
    storage is optional (defaulting to `'osfstorage'`) and an empty path refers
    to the project storage's root. The project-ID is the code used to find the
    project online. For example, the webpage reached at the website
    `https://osf.io/tery8/` is the project page for the project whose ID is
    `tery8`.

    If any of the optional keyword arguments are given, then the returned path
    will always use the specified options; a new path is returned with updated
    options if necessary; this is done before joining paths if multiple
    positional arguments are given. If the `client` keyword is given, then it is
    modified by the keyword options before being used in the path. All optional
    keyword arguments have a default value of `Ellipsis`, which indicates that
    the `client`'s value for that option should be used.

    Parameters
    ----------
    obj : path-like
        The path or path-like object to convert into an OSFPath, typically a
        string.
    client : OSFClient or None, optional
        The `OSFClient` object to use. The `OSFClient` is responsible primarily
        for the caching of data locally. If `OSFClient` is `None`, then an
        `OSFClient` object is created for the project using a temporary cache
        directory.
    cache_path : path-like or None, optional
        The local directory in which cache files should be stored. This option
        is ignored if `client` is not `None`; otherwise it is passed to the
        created client object. The cache directory is the root cache directory
        for the entire OSF project.
    file_cache_mode : cloudpathlib.enums.FileCacheMode, optional
        How often to clear the file cache; see [cloudpathlib's caching
        docs](https://cloudpathlib.drivendata.org/stable/caching/) for more
        information about the options in `cloudpathlib.enums.FileCacheMode`.
    mkdir_mode : int, optional
        The mode to use when making directories in the cache. By default this is
        `0o775`. This option is ignored if the `client` option is not `None`.
    pagesize : int, optional
        The number of items to include in a single page when paging directory
        contents from the OSF server. The default is 100. This option is ignored
        if the `client` option is not `None`.
    """
    if isinstance(obj, OSFPath):
        # We may want to grab some options out of the argument in this case.
        if client is None:
            client = obj.client
    else:
        # If there's no '://' in the path, then we add 'osf://' to the front.
        obj = pathstr(obj)
        if scheme_sep not in obj:
            obj = 'osf://' + obj
    # If the cache_path is Ellipsis and there's no client provided, then we want
    # to use pimm's default cache path
    if local_cache_dir is None:
        if client is None:
            if cache_path is Ellipsis:
                cache_path = None #TODO
            if cache_path is not None:
                local_cache_dir = Path(cache_path).expanduser() / 'osf'
    # At this point we can go ahead and create the initial path.
    path = OSFPath(obj,
                   client=client,
                   local_cache_dir=local_cache_dir,
                   file_cache_mode=file_cache_mode,
                   mkdir_mode=mkdir_mode,
                   pagesize=pagesize)
    # If there were additional arguments, append them to the path now.
    for arg in args:
        path = path / arg
    # Return the path.
    return path


# S3Path Functions #############################################################

@docwrap
def is_s3path(obj):
    """Detects whether the input is an `S3Path` object.

    `is_s3path(obj)` returns `True` if `obj` is an instance of the `S3Path`
    class and `False` otherwise.

    See also: `like_s3path`

    Parameters
    ----------
    obj : object
        The object whose membership in the `S3Path` class is to be determined.
    
    Returns
    -------
    boolean
        `True` if `obj` is an instance of `S3Path` and `False` otherwise.
    """
    return isinstance(obj, S3Path)
@docwrap
def like_s3path(obj):
    """Detects whether the input can be converted into an `S3Path` object.

    `like_s3path(obj)` returns `True` if `obj` is an instance of the `S3Path`
    class or is a string that forms a valid S3 path, and `False` otherwise.

    See also: `is_s3path`

    Parameters
    ----------
    obj : object
        The object whose ability to be converted into an `S3Path` instance is
        to be determined.
    
    Returns
    -------
    boolean
        `True` if `obj` is an instance of `S3Path` or is a string that could be
        converted into an `S3Path` and `False` otherwise.
    """
    if isinstance(obj, S3Path):
        return True
    url = urlparse(pathstr(obj))
    return bool(url.scheme == 's3' and url.netloc)
@docwrap
def s3path(obj, *args, **kwargs):
    """Creates and returns an `S3Path` representing an AWS S3 repository.

    `s3path(p)` creates and returns an `S3Path` object, which is a type of
    `cloudpathlib.CloudPath` object, from the path or path-string `p`. If `p` is
    an `S3Path`, then it is returned as-is. Otherwise `pathstr(p)` is converted
    into an `S3Path`; `pathstr(p)` may start with `'s3://'` (not case-sensitive)
    or, if it does not have a scheme specifier, `'s3://'` will be prepended to
    it.

    `s3path(p, a1, a2...)` converts `p` into an `S3Path` then joins the `a1`,
    `a2`, etc. values to the end of the path and returns the joined path.

    The `s3path` function accepts all the optional arguments of the `S3Client`
    type from `cloudpathlib` as well as the `client` option. If the `client`
    option is given along with additional optional arguments, then the optional
    arguments are ignored.

    Additionally, `s3path` parses the option `cache_path`, which is not normally
    accepted by `S3Path`, which instead requires the option `local_cache_dir`.
    Any time that a `local_cache_dir` is given, it overrides the `cache_path`;
    however, if `local_cache_dir` is not given and `cache_path` is, then the
    directory `os.path.join(cache_path, "s3")` is given as the `local_cache_dir`
    option.
    """
    # Extract a few options.
    client = kwargs.pop('client', None)
    cache_path = kwargs.pop('cache_path', None)
    lcd = kwargs.get('local_cache_dir', None)
    # Do some initial configuration.
    if isinstance(obj, S3Path):
        # We may want to grab some options out of the argument in this case.
        if client is None:
            client = obj.client
    else:
        # If there's no '://' in the path, then we add 'osf://' to the front.
        obj = pathstr(obj)
        if scheme_sep not in obj:
            obj = 's3://' + obj
    if client is None:
        # If the cache_path is Ellipsis and there's no client provided, then we
        # want to use pimm's default cache path.
        if cache_path is Ellipsis and client is None:
            cache_path = None #TODO
        if cache_path is not None and 'local_cache_dir' not in kwargs:
            kwargs['local_cache_dir'] = Path(cache_path).expanduser() / "s3"
        # At this point we can go ahead and create the client object.
        client = S3Client(**kwargs)
    path = S3Path(obj, client=client)
    # If there were additional arguments, append them to the path now.
    for arg in args:
        path = path / arg
    # Return the path.
    return path


# GSPath Functions #############################################################

@docwrap
def is_gspath(obj):
    """Detects whether the input is an `GSPath` object.

    `is_gspath(obj)` returns `True` if `obj` is an instance of the `GSPath`
    class and `False` otherwise.

    See also: `like_gspath`

    Parameters
    ----------
    obj : object
        The object whose membership in the `GSPath` class is to be determined.
    
    Returns
    -------
    boolean
        `True` if `obj` is an instance of `GSPath` and `False` otherwise.
    """
    return isinstance(obj, GSPath)
@docwrap
def like_gspath(obj):
    """Detects whether the input can be converted into an `GSPath` object.

    `like_gspath(obj)` returns `True` if `obj` is an instance of the `GSPath`
    class or is a string that forms a valid GS path, and `False` otherwise.

    See also: `is_gspath`

    Parameters
    ----------
    obj : object
        The object whose ability to be converted into an `GSPath` instance is
        to be determined.
    
    Returns
    -------
    boolean
        `True` if `obj` is an instance of `GSPath` or is a string that could be
        converted into an `GSPath` and `False` otherwise.
    """
    if isinstance(obj, GSPath):
        return True
    url = urlparse(pathstr(obj))
    return bool(url.scheme == 'gs' and url.netloc)
@docwrap
def gspath(obj, *args, **kwargs):
    """Creates and returns an `GSPath` representing a Google Storage repository.

    `gspath(p)` creates and returns a `GSPath` object, which is a type of
    `cloudpathlib.CloudPath` object, from the path or path-string `p`. If `p` is
    a `GSPath`, then it is returned as-is. Otherwise `pathstr(p)` is converted
    into an `GSPath`; `pathstr(p)` may start with `'gs://'` (not case-sensitive)
    or, if it does not have a scheme specifier, `'gs://'` will be prepended to
    it.

    `gspath(p, a1, a2...)` converts `p` into an `GSPath` then joins the `a1`,
    `a2`, etc. values to the end of the path and returns the joined path.

    The `gspath` function accepts all the optional arguments of the `GSClient`
    type from `cloudpathlib` as well as the `client` option. If the `client`
    option is given along with additional optional arguments, then the optional
    arguments are ignored.

    Additionally, `gspath` parses the option `cache_path`, which is not normally
    accepted by `GSPath`, which instead requires the option `local_cache_dir`.
    Any time that a `local_cache_dir` is given, it overrides the `cache_path`;
    however, if `local_cache_dir` is not given and `cache_path` is, then the
    directory `os.path.join(cache_path, "gs")` is given as the `local_cache_dir`
    option.
    """
    # Extract a few options.
    client = kwargs.pop('client', None)
    cache_path = kwargs.pop('cache_path', None)
    lcd = kwargs.get('local_cache_dir', None)
    # Do some initial configuration.
    if isinstance(obj, GSPath):
        # We may want to grab some options out of the argument in this case.
        if client is None:
            client = obj.client
    else:
        # If there's no '://' in the path, then we add 'osf://' to the front.
        obj = pathstr(obj)
        if scheme_sep not in obj:
            obj = 'gs://' + obj
    # At this point we can go ahead and create the client object.
    if client is None:
        # If the cache_path is Ellipsis and there's no client provided, then we
        # want to use pimm's default cache path.
        if cache_path is Ellipsis and client is None:
            cache_path = None #TODO
        if cache_path is not None and 'local_cache_dir' not in kwargs:
            kwargs['local_cache_dir'] = Path(cache_path).expanduser() / "gs"
        client = GSClient(**kwargs)
    path = GSPath(obj, client=client)
    # If there were additional arguments, append them to the path now.
    for arg in args:
        path = path / arg
    # Return the path.
    return path


# AzureBlobPath Functions ######################################################

@docwrap
def is_azpath(obj):
    """Detects whether the input is an `AzureBlobPath` object.

    `is_azpath(obj)` returns `True` if `obj` is an instance of the
    `AzureBlobPath` class and `False` otherwise.

    See also: `like_azpath`

    Parameters
    ----------
    obj : object
        The object whose membership in the `AzureBlobPath` class is to be
        determined.
    
    Returns
    -------
    boolean
        `True` if `obj` is an instance of `AzureBlobPath` and `False` otherwise.
    """
    return isinstance(obj, AzureBlobPath)
@docwrap
def like_azpath(obj):
    """Detects whether an input can be converted into an `AzureBlobPath` object.

    `like_azpath(obj)` returns `True` if `obj` is an instance of the
    `AzureBlobPath` class or is a string that forms a valid Azure path, and
    `False` otherwise.

    See also: `is_azpath`

    Parameters
    ----------
    obj : object
        The object whose ability to be converted into an `AzureBlobPath`
        instance is to be determined.
    
    Returns
    -------
    boolean
        `True` if `obj` is an instance of `AzureBlobPath` or is a string that
        could be converted into an `AzureBlobPath` and `False` otherwise.
    """
    if isinstance(obj, AzureBlobPath):
        return True
    url = urlparse(pathstr(obj))
    return bool(url.scheme == 'az' and url.netloc)
@docwrap
def azpath(obj, *args, **kwargs):
    """Creates and returns an `AzureBlobPath` representing an Azure repository.

    `azpath(p)` creates and returns an `AzureBlobPath` object, which is a type
    of `cloudpathlib.CloudPath` object, from the path or path-string `p`. If `p`
    is an `AzureBlobPath`, then it is returned as-is. Otherwise `pathstr(p)` is
    converted into an `AzureBlobPath`; `pathstr(p)` may start with `'az://'`
    (not case-sensitive) or, if it does not have a scheme specifier, `'az://'`
    will be prepended to it.

    `azpath(p, a1, a2...)` converts `p` into an `AzureBlobPath` then joins the
    `a1`, `a2`, etc. values to the end of the path and returns the joined path.

    The `azpath` function accepts all the optional arguments of the
    `AzureBlobClient` type from `cloudpathlib` as well as the `client`
    option. If the `client` option is given along with additional optional
    arguments, then the optional arguments are ignored.

    Additionally, `azpath` parses the option `cache_path`, which is not normally
    accepted by `AzureBlobPath`, which instead requires the option
    `local_cache_dir`.  Any time that a `local_cache_dir` is given, it overrides
    the `cache_path`; however, if `local_cache_dir` is not given and
    `cache_path` is, then the directory `os.path.join(cache_path, "az")` is
    given as the `local_cache_dir` option.
    """
    # Extract a few options.
    client = kwargs.pop('client', None)
    cache_path = kwargs.pop('cache_path', None)
    lcd = kwargs.get('local_cache_dir', None)
    # Do some initial configuration.
    if isinstance(obj, AzureBlobPath):
        # We may want to grab some options out of the argument in this case.
        if client is None:
            client = obj.client
    else:
        # If there's no '://' in the path, then we add 'osf://' to the front.
        obj = pathstr(obj)
        if scheme_sep not in obj:
            obj = 'az://' + obj
    if client is None:
        # If the cache_path is Ellipsis and there's no client provided, then we
        # want to use pimm's default cache path.
        if cache_path is Ellipsis and client is None:
            cache_path = None #TODO
        if cache_path is not None and 'local_cache_dir' not in kwargs:
            kwargs['local_cache_dir'] = Path(cache_path).expanduser() / "az"
        # At this point we can go ahead and create the client object.
        client = AzureBlobClient(**kwargs)
    path = AzureBlobPath(obj, client=client)
    # If there were additional arguments, append them to the path now.
    for arg in args:
        path = path / arg
    # Return the path.
    return path


# Filesystem Paths #############################################################

@docwrap
def is_filepath(p):
    """Detects whether an object is a filesystem `Path` object.

    Any object that inherits from the `Path` type is considered a filesystem
    path. File-paths correspond to URLs that begin with `'file://'`.

    Parameters
    ----------
    p : path-like
        The object whose quality as a path is to be assessed.

    Returns
    -------
    boolean
        `True` if `p` is an instance of the `Path` type and `False` otherwise.
    """
    return isinstance(p, Path)
@docwrap
def like_filepath(obj):
    """Detects whether an input can be converted into a `Path` object.

    `like_filepath(obj)` returns `True` if `obj` is an instance of the `Path`
    class or is a string that forms a valid path, and `False` otherwise. Most
    strings are at least theoretically valid paths, but any string that starts
    with a sheme followed by `'://'` must have the `'file'` scheme.

    See also: `is_filepath`

    Parameters
    ----------
    obj : object
        The object whose ability to be converted into a `Path` instance is to be
        determined.
    
    Returns
    -------
    boolean
        `True` if `obj` is an instance of `AzureBlobPath` or is a string that
        could be converted into an `AzureBlobPath` and `False` otherwise.

    """
    if isinstance(obj, Path):
        return True
    url = urlparse(pathstr(obj))
    return bool(url.scheme == '' or url.scheme == 'file')
@docwrap
def filepath(p, *args):
    """Returns a local `Path` object for the given path if possible.

    The `filepath` function is intended to coerce remote paths (such as the
    S3 or OSF paths managed through the `cloudpathlib.CloudPath` class) into
    paths representing their local caches. If a local file is requested, then it
    is always downloaded before the `Path` is returned. For directories, the
    cache directory itself will always exist, but no such guarantee is made
    about its contents.

    If the argument to `filepath` is a string and not a path object, then
    it is converted into a path via the `immlib.path` function.

    Parameters
    ----------
    p : path-like
        The object whose local cache path is to be returned after it has been
        downloaded.

    Returns
    -------
    Path
        If the input path `p` is already a local path, then it is returned. If
        `p` is a remote path, then it is downloaded and its cache path is
        returned. If there is no local cache or if the file does not exist, an
        error is raised.
    """
    # We are converting the argument p into a Path of some kind.
    if isinstance(p, CloudPath):
        # If p is a CloudPath, we return a path that represents the cache.
        return CloudCachePath(p)
    if isinstance(p, (PathLike, PurePath)):
        # If p is a path, str, or PathLike, we can just return a copy.
        return Path(p)
    elif isinstance(p, str):
        # We need to make sure that it doesn't start with file:// before we pass
        # this string along.
        if p.startswith('file://'):
            p = p[7:]
        return Path(p)
    elif isinstance(p, bytes):
        # Same as above but with bytes.
        if p.startswith(b'file://'):
            p = p[7:]
        return Path(p)
    else:
        # Try to convert it to a pathstr then turn it into a path.
        return Path(pathstr(p))


# Other Utility Functions ######################################################

# The registry of all recognized path types uses a named tuple to store entries.
# Pure paths are not included as they are innate to the functions.
PathTypeRecord = namedtuple(
    'PathTypeRecord',
    ('construct', 'isfn', 'likefn', 'fspath'))
def _cloudtofspath(cp):
    return cp.fspath
pathtypes = dict(
    s3=PathTypeRecord(s3path, is_s3path, like_s3path, _cloudtofspath),
    gs=PathTypeRecord(gspath, is_gspath, like_gspath, _cloudtofspath),
    az=PathTypeRecord(azpath, is_azpath, like_azpath, _cloudtofspath),
    osf=PathTypeRecord(osfpath, is_osfpath, like_osfpath, _cloudtofspath),
    file=PathTypeRecord(filepath, is_filepath, like_filepath, lambda path:path))
scheme_sep = '://'
@docwrap
def pathtype(path, default='file', encoding='utf-8'):
    """Given a path, returns the pathtype record and remaining path.

    This is an internal function that looks up the pathtype for a path from the
    `pathtypes` dictionary and returning both the pathtype record for the path.

    Paths that do not have an explicit scheme marker are considered filesystem
    paths.

    Parameters
    ----------
    path : path-like or str
        The path whose pathtype record is being extracted.
    default : object, optional
        The scheme that is assumed if no scheme is specified (i.e., `str(path)`
        does not begin with `'<scheme>://'`. By default this is `'file'`.
    encoding : str, optional
        The encoding to use if the path argument is a `bytes` object; the
        default is `'utf-8'`.

    Returns
    -------
    PathTypeRecord
        A data record about the path type of the given path.

    Raises
    ------
    KeyError
        If the scheme of the the given path is not recognized.
    """
    if isinstance(path, bytes):
        path = path.decode('utf-8')
    spath = pathstr(path)
    if scheme_sep not in spath:
        scheme = default
    else:
        scheme = spath.split(scheme_sep)[0]
    return pathtypes[scheme]
@docwrap
def is_path(p):
    """Detects whether an object is either a `Path` or a `CloudPath` object.

    Both `Path` and `CloudPath` objects abstractly represent paths, but they do
    not share a subclass. `is_path` tests whether an object's type is a subclass
    of any of path types recognized by `immlib`. Additional path types can be
    registered by adding `immlib.paths.PathTypeRecord` instances to the
    `immlib.pathtypes` dictionary. The key for such a record should be the
    string prefix for the path type (such as `'s3'` for an `S3Path` type).

    Parameters
    ----------
    p : path-like
        The object whose quality as a path is to be assessed.

    Returns
    -------
    boolean
        `True` if `p` has a type that is recognized by `immlib` as a path type
        and `False` otherwise.

    """
    try:
        pt = pathtype(p)
    except KeyError:
        return False
    return pt.isfn(p)
@docwrap
def like_path(p):
    """Detects whether an object is either like a `Path` or `CloudPath` object.

    Both `Path` and `CloudPath` object abstractly represent paths, but they do
    not share a subclass. `like_path` tests whether an object's type is a
    subclass of any of path types recognized by `immlib` or is a string or bytes
    object that could be converted into a path. Additional path types can be
    registered by adding `immlib.pathlib.PathTypeRecord` instances to the
    `immlib.pathlib.pathtypes` dictionary. The key for such a record should be
    the string prefix for the path type (such as `'s3'` for an `S3Path` type).

    Parameters
    ----------
    p : object
        The object whose quality as a path-like object is to be assessed.

    Returns
    -------
    boolean
        `True` if `p` has a type that is recognized by `immlib` as a path type
        or is an object that can be converted into a path type and `False`
        otherwise.
    """
    try:
        pt = pathtype(p)
    except KeyError:
        return False
    return pt.likefn(p)


# path #########################################################################

@docwrap
def path(arg0, *args, **kwargs):
    """Convenience function for instantiating `Path` objects.

    `path(arg)` returns a `Path`-like object that references the path given by
    the argument `arg`. The `arg` is converted into a string prior to conversion
    into a path if it is not a path already.

    `path(arg, *args)` joins the list of arguments in `args` to the path created
    from the `arg`.

    Optional keyword arguments may be given as well, 
    """
    nargs = len(args)
    # What kind of pathtype is this?
    try:
        pt = pathtype(arg0)
    except KeyError:
        pt = None
    if pt is None:
        raise ValueError(f"unrecognized pathtype for path: {arg0}")
    # Start by converting the first argument into a path, if it isn't one.
    if not kwargs and pt.isfn(arg0):
        # We only need to update the path if new keyword args were given.
        p = arg0
    else:
        # We have a pathtype, so we can create a path from these arguments.
        p = pt.construct(arg0, **kwargs)
    # At this point, p is a pathtype, so we can append any extra args.
    # Join the rest of the arguments to the path.
    for arg in args:
        p = p / arg
    # That is all that's required.
    return p

@docwrap
def pathdict(arg, all=False, filter=None, ondir=None, onfile=None):
    """Returns a lazy dictionary of the nested paths beginning at the argument.

    Searches a directory and all its subdirectories, lazily, for all contents,
    uniting them in a single nested lazy dictionary. The dictionaries in the
    nest represent directories whose keys are the filenames of their
    contents. The value of a key is another lazy dictionaries or a path object
    if the file is not a directory.
    
    Parameters
    ----------
    arg : path-like
        The path from which to begin the search.
    all : boolean, optional
        Whether to include hidden files (`True`) or not (`False`) in the
        directory lists. The default is `False`.
    filter : None or function, optional
        A filter that must return either `True` (indicating that the path should
        be included in the pathdict) or `False` (indicating that the path should
        not be included in the pathdict) for each path that is scanned. The
        default is `None`, meaning that no filter is applied.
    ondir : function, optional
        A function to run on any path encountered during the `pathdict` search
        that is a directory. When `pathdict` is given a path that is a
        directory, it returns a lazy dictionary whose keys are the filenames of
        the contents of that directory. For subdirectories, their filenames are
        mapped to the return value of `ondir(path)` where `path` is the path
        object for the subdirectory. By default this is `pathdict` itself,
        resulting in a nested structure for subdirectories. The `all`, `filter`,
        `ondir`, and `onfile` parameters are all passed to this function.
    onfile : function, optional
        A function to run on any path encountered during the `pathdict` search
        that is a file. When `pathdict` is given a path that is a directory, it
        returns a lazy dictionary whose keys are the filenames of the contents
        of that directory. For files in the directory, their filenames are
        mapped to the return value of `onfile(path)` where `path` is the path
        object for the file. By default this is `None`, indicating that the path
        itself should be returned.

    Returns
    -------
    ldict
        A lazy dictionary of the contents of the argument.
    """
    if ondir is None:
        ondir = pathdict
    root = path(arg)
    if root.is_dir():
        contents = {}
        kw = {'all':all, 'filter':filter, 'ondir':ondir, 'onfile':onfile}
        for p in root.iterdir():
            if p.is_file():
                q = p if onfile is None else lazy(onfile, p)
            else:
                q = p if ondir is None else lazy(ondir, p, **kw)
            contents[p.name] = q
        return ldict(contents)
    elif onfile is None:
        return root
    else:
        return onfile(root)
