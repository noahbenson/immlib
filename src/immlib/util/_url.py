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
def _atomic_open(path, mode='wb', *, overwrite=True):
    """Context manager that opens a temporary file for writing, then moves it
    into place at `path` when the block exits without an error.

    The temporary file is created in the same directory as `path`, and it is
    moved into place atomically, so readers (in other threads or processes)
    never see a partially written file at `path`. If the block raises an
    exception, the temporary file is removed and `path` is left unchanged.

    If `overwrite` is ``True`` (the default), a file already at `path` is
    replaced. If it is ``False``, then a file already at `path` is left
    alone and the newly written file is discarded; this is the right choice
    for a cache, whose files are written once and whose contents do not
    depend on which writer wrote them.

    .. Note:: Windows differs from POSIX in ways that make `overwrite` more
        than a matter of taste when a file may be written and read at the
        same time. Windows refuses to replace a file that another thread or
        process has open, so overwriting can fail with a
        ``PermissionError``; the error is raised, so that a write never
        fails silently, and a caller that only wants the file to exist
        should treat it as success when the file is there (see
        ``immlib.pathlib._osf._osf_cache_file``). Worse, a *successful*
        replacement leaves the replaced file in a pending-delete state until
        every handle to it is closed, and opening `path` by name raises
        ``PermissionError`` for as long as that lasts, so a reader can fail
        on a file that exists and is complete. Passing ``overwrite=False``
        avoids both problems, because nothing at `path` is ever deleted or
        replaced: the file is linked into place, or, where hard links are
        not supported, renamed into place, which on Windows fails rather
        than replacing an existing file.
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
    if not overwrite:
        # A hard link fails if something is already at path, and it never
        # deletes or replaces anything, so no reader of path is disturbed.
        try:
            os.link(tmp, path)
        except FileExistsError:
            # Another writer got there first; its file is complete.
            rmtmp()
            return
        except OSError:
            # Hard links are not supported everywhere (FAT filesystems and
            # some network shares); fall back to a rename below.
            pass
        else:
            rmtmp()
            return
        # os.rename, unlike os.replace, refuses to replace an existing file
        # on Windows, which is what is wanted here: another writer's file is
        # kept, and no file is deleted for a reader to trip over. On POSIX
        # it does replace, but replacing a file that a reader has open is
        # harmless there.
        try:
            os.rename(tmp, path)
        except FileExistsError:
            rmtmp()
        except BaseException:
            rmtmp()
            raise
        return
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
                 mkdirs=True, mkdir_mode=0o775, expanduser=True,
                 overwrite=True):
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
    overwrite : bool, optional
        Whether to replace a file that is already at `destpath` when the
        download finishes. The default is ``True``. If ``False``, then a
        file that is already at `destpath` is left alone and the downloaded
        data are discarded; this is appropriate for a cache, in which the
        file that is already in place is as good as the one just
        downloaded, and in which replacing it can disturb a concurrent
        reader on Windows.

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
        with _atomic_open(p, 'wb', overwrite=overwrite) as fl:
            shutil.copyfileobj(response, fl)
    return destpath
