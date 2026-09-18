# -*- coding: utf-8 -*-
###############################################################################
# pimms/util/_url.py


# Dependencies ################################################################

import os, shutil, tempfile
import urllib.parse, urllib.request
from pathlib import Path
from contextlib import contextmanager


# Utilities ###################################################################

# tempfile.mkstemp creates its file readable and writable by its owner only,
# but a downloaded file should have the permissions an ordinary new file
# would have, so that (for example) a cache directory can be shared. The
# umask is read once, here, because reading it is not thread-safe.
def _default_file_mode():
    try:
        umask = os.umask(0)
        os.umask(umask)
    except OSError:  # pragma: no cover - not available everywhere
        return None
    return 0o666 & ~umask
_FILE_MODE = _default_file_mode()


@contextmanager
def _atomic_open(path, mode='wb'):
    """Context manager that opens a temporary file for writing, then moves it
    into place at `path` when the block exits without an error.

    The temporary file is created in the same directory as `path`, and it is
    moved into place with ``os.replace``, which is atomic, so readers (in
    other threads or processes) never see a partially written file at
    `path`. If the block raises an exception, the temporary file is removed
    and `path` is left unchanged.

    .. Note:: On Windows, replacing a file that another thread or process
        currently has open raises ``PermissionError`` (POSIX allows it).
        The error is raised, so that a write never fails silently; a caller
        that only wants the file to exist, such as a download into a cache
        directory, should treat that error as success when the file is
        there (see ``immlib.pathlib._osf._osf_cache_file``).
    """
    path = Path(path)
    (fd, tmp) = tempfile.mkstemp(
        dir=path.parent, prefix=f'.{path.name}.', suffix='.part')
    def rmtmp():
        try:
            os.remove(tmp)
        except OSError:
            pass
    try:
        with os.fdopen(fd, mode) as fl:
            yield fl
        if _FILE_MODE is not None:
            try:
                os.chmod(tmp, _FILE_MODE)
            except OSError:
                pass
    except BaseException:
        rmtmp()
        raise
    try:
        os.replace(tmp, path)
    except BaseException:
        rmtmp()
        raise


# URL Functions ###############################################################

def is_url(url, /):
    '''Returns ``True`` if given a valid URL string and ``False`` otherwise.
    
    ``is_url(url)`` returns ``True`` if and only if the given URL is a valid
    URL string that includes the URL scheme and the netloc unless the scheme is
    ``'file'``, in which case the netloc is optional. Whether the URL can be
    requested or not does not make a difference; ``is_url`` operates on the
    given URL string alone.
    
    See Also
    --------
    can_download_url
    '''
    try:
        p = urllib.parse.urlparse(url)
        return bool(p.scheme and (p.netloc or p.scheme == 'file'))
    except Exception:
        return False
def can_download_url(url):
    '''Returns ``True`` if given a requestable URL and ``False`` otherwise.
    
    ``can_download_url(url)`` returns ``True`` if and only if the given URL is
    both a valid URL string and can be retrieved. If a URL-request fails for
    the given URL then ``False`` is returned.
    
    See Also
    --------
    is_url
    '''
    try: 
        with urllib.request.urlopen(url) as response:
            return bool(response)
    except Exception:
        return False
def url_download(url, /, destpath=None, *,
                 mkdirs=True, mkdir_mode=0o775, expanduser=True):
    '''Returns the contents of the given URL as a byte-string.
    
    ``url_download(url)`` returns the contents of the given url as a
    byte-string.

    ``url_download(url, destpath)`` downloads the given url to the given
    destination path, ``destpath``, and returns that path on success.
    
    Parameters
    ----------
    url : str or URL
        The URL to be downloaded.
    destpath : PathLike or None, optional
        A string, ``pathlib.Path`` object, or any object that can be converted
        into a ``Path``, which details the local destination path to which the
        URL should be saved. The default, ``None``, indicates that the file
        should not be downloaded to a path but should instead just be returned
        as a byte string.
    mkdirs : boolean, optional
        Whether to make directories that do not exist in order to save the URL
        to the path `destpath`. The default is ``True``.
    mkdir_mode : int, optional
        The mode to give any directory created by this function. The default is
        ``0o775``. If `mkdirs` is set to ``False``, then this option is
        ignored.
    expanduser : bool, optional
        Whether to expand the ``~`` character into the user's directory in the
        destination path. The default is ``True``.

    Returns
    -------
    bytes or Path
        If `destpath` is ``None``, then a ``bytes`` object containing the URL
        contents is returned; otherwise, the ``pathlib.Path`` object
        representing the downloaded file is returned.
    '''
    # Make the URL request and download it.
    with urllib.request.urlopen(url) as response:
        # We need to handle things differently depending on whether we have
        # been given a destination path.
        if destpath is None:
            return response.read()
        destpath = Path(destpath)
        if expanduser:
            destpath = destpath.expanduser()
        if destpath.is_dir():
            raise ValueError(f"destpath is a directory but must be a"
                             f" filename: {destpath}")
        p = destpath.resolve()
        # Make the directory if it doesn't exist and we have been asked to.
        if mkdirs:
            p.parent.mkdir(mode=mkdir_mode, parents=True, exist_ok=True)
        # Make sure the directory exists regardless.
        if not p.parent.is_dir():
            raise ValueError(f"destpath parent does not exist: {p.parent}")
        # Now save the file. It is written to a temporary file first and
        # moved into place when complete, so that a partially downloaded
        # file is never visible at destpath.
        with _atomic_open(p, 'wb') as fl:
            shutil.copyfileobj(response, fl)
    return destpath
