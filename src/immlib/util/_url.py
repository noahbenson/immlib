# -*- coding: utf-8 -*-
################################################################################
# pimms/util/_url.py


# Dependencies #################################################################

from ..doc import docwrap
import urllib, shutil
from pathlib import Path


# URL Functions ################################################################

def is_url(url):
    '''Returns `True` if given a valid URL string and `False` otherwise.
    
    `is_url(url)` returns `True` if and only if the given URL is a valid
    URL string that includes the URL scheme and the netloc. Whether the URL
    can be requested or not does not make a difference; `is_url` operates on
    the given URL string alone.
    
    See also: `can_download_url`
    '''
    try:
        p = urllib.parse.urlparse(url)
        return p.scheme and p.netloc
    except Exception:
        return False
def can_download_url(url):
    '''Returns `True` if given a requestable URL and `False` otherwise.
    
    `can_download_url(url)` returns `True` if and only if the given URL is both
    a valid URL string and can be retrieved. If a URL-request fails for the
    given URL then `False` is returned.
    
    See also: `is_url`
    '''
    try: 
        with urllib.request.urlopen(url) as response:
            return bool(response)
    except Exception:
        return False
@docwrap
def url_download(url, destpath=None,
                 mkdirs=True, mkdir_mode=0o775, expanduser=True):
    '''Returns the contents of the given URL as a byte-string.
    
    `url_download(url)` returns the contents of the given url as a byte-string.

    `url_download(url, destpath)` downloads the given url to the given
    destination path, `destpath`, and returns that path on success.
    
    Parameters
    ----------
    url : str or URL
        The URL to be downloaded.
    destpath : PathLike or None, optional
        A string, `pathlib.Path` object, or any object that can be converted
        into a `Path`, which details the local destination path to which the URL
        should be saved. The default, `None`, indicates that the file should not
        be downloaded to a path but should instead just be returned as a byte
        string.
    mkdirs : boolean, optional
        Whether to make directories that do not exist in order to save the URL
        to the path `destpath`. The default is `True`.
    mkdir_mode : int, optional
        The mode to give any directory created by this function. The default is
        `0o775`. If `mkdirs` is set to `False`, then this option is ignored.
    expanduser : boolean, optional
        Whether to expand the `~` character into the user's directory in the
        destination path. The default is `True`.
    '''
    # Make the URL request and download it.
    with urllib.request.urlopen(url) as response:
        # We need to handle things differently depending on whether we have been
        # given a destination path.
        if destpath is None:
            destpath = response.read()
        else:
            destpath = Path(destpath)
            if expanduser:
                destpath.expanduser()
            if destpath.is_dir():
                raise ValueError(f"destpath is a directory but must be a"
                                 f" filename: {destpath}")
            p = destpath.resolve()
            # Make the directory if it doesn't exist and we have been asked to.
            if mkdirs:
                if not p.parent.exists():
                    p.mkdir(mode=mkdir_mode, parents=True)
            # Make sure the directory exists regardless.
            if not p.parent.exists():
                raise ValueError("destpath parent does not exist: {p.parent}")
            # Now open the path and save the file.
            with p.open('wb') as fl:
                shutil.copyfileobj(response, fl)
    return destpath
