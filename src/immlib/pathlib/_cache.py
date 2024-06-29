# -*- coding: utf-8 -*-
################################################################################
# immlib/pathlib/_cache.py


# Dependencies #################################################################

import os, sys
from pathlib      import (Path, PurePath)

from pcollections import (pdict, ldict, lazy)
from cloudpathlib import CloudPath

from   ..doc      import docwrap


# CloudCachePath ###############################################################

_pymajor = int(sys.version.split('.')[0])
_pyminor = int(sys.version.split('.')[1])
_pyv3_12 = (_pymajor > 3 or (_pymajor == 3 and _pyminor > 11))
_pathbase = Path if _pyv3_12 else type(Path())

class CloudCachePath(_pathbase):
    """A filesystem path wrapper for the CloudPath type.

    A `CloudCachePath` object is a simple wrapper around a `CloudPath` object.
    However, unlike a `CloudPath` object, a `CloudCachePath` object is also a
    `Path` object. `CloudCachePath` objects represent the cache directories of
    `CloudPath` objects, presenting the directories as fully fleshed out
    instances of their associated cloud paths. Interacting with files forces
    them to be downloaded.
    """
    def __new__(cls, cloud_path):
        if not isinstance(cloud_path, CloudPath):
            raise TypeError(
                f"CloudCachePath requiers a CloudPath object, not"
                f" {type(cloud_path)}")
        pre = cloud_path.cloud_prefix
        lcd = cloud_path.client._local_cache_dir
        fspath = Path(lcd).absolute() / str(cloud_path)[len(pre):]
        p = super().__new__(cls, fspath)
        p.cloud_path = cloud_path
        return p
    def __init__(self, cloud_path):
        if _pyv3_12:
            pre = cloud_path.cloud_prefix
            lcd = cloud_path.client._local_cache_dir
            fspath = Path(lcd).absolute() / str(cloud_path)[len(pre):]
            super().__init__(self, fspath)
    def __truediv__(self, other):
        return CloudCachePath(self.cloud_path / other)
    def iterdir(self):
        return map(CloudCachePath, self.cloud_path.iterdir())
    @property
    def parents(self):
        return tuple(map(CloudCachePath, self.cloud_path.parents))
    @property
    def parts(self):
        return self.cloud_path.parts[1:]
    def stat(self, *args, **kw):
        return Path(self.cloud_path.fspath).stat(*args, **kw)
    def chmod(self, *args, **kw):
        #return Path(self.cloud_path.fspath).chmod(*args, **kw)
        raise TypeError(f"{type(self)} filesystem access is read-only")
    def exists(self):
        return Path(self.cloud_path.fspath).exists()
    def expanduser(self):
        return Path(self.cloud_path.fspath).expanduser()
    def glob(self, patt):
        return map(CloudCachePath, self.cloud_path.glob(patt))
    def group(self):
        return Path(self.cloud_path.fspath).group()
    def is_dir(self):
        if os.path.exists(self):
            return os.path.is_dir(self)
        else:
            return self.cloud_path.is_dir()
    def is_file(self):
        if os.path.exists(self):
            return os.path.is_file(self)
        else:
            return self.cloud_path.is_file()
    def is_mount(self):
        return self.cloud_path.is_mount()
    def is_symlink(self):
        return self.cloud_path.is_symlink()
    def is_socket(self):
        return self.cloud_path.is_socket()
    def is_fifo(self):
        return self.cloud_path.is_fifo()
    def is_block_device(self):
        return self.cloud_path.is_block_device()
    def is_char_device(self):
        return self.cloud_path.is_char_device()
    def iterdir(self):
        return map(CloudCachePath, self.cloud_path.iterdir())
    def lchmod(self, mode):
        return Path(self.cloud_path.fspath).lchmod(mode)
    def lstat(self):
        return Path(self.cloud_path.fspath).lstat()
    def mkdir(self, *args, **kw):
        #return Path(self.cloud_path.fspath).mkdir(*args, **kw)
        raise TypeError(f"{type(self)} filesystem access is read-only")
    def open(self, *args, **kw):
        return Path(self.cloud_path.fspath).open(*args, **kw)
    def owner(self):
        return Path(self.cloud_path.fspath).owner()
    def read_bytes(self):
        return Path(self.cloud_path.fspath).read_bytes()
    def read_text(self, *args, **kw):
        return Path(self.cloud_path.fspath).read_text(*args, **kw)
    def readlink(self):
        return Path(self.cloud_path.fspath).readlink()
    def rename(self, target):
        #return Path(self.cloud_path.fspath).rename(target)
        raise TypeError(f"{type(self)} filesystem access is read-only")
    def replace(self, target):
        #return Path(self.cloud_path.fspath).replace(target)
        raise TypeError(f"{type(self)} filesystem access is read-only")
    def absolute(self):
        return CloudCachePath(self.cloud_path)
    def resolve(self, *args, **kw):
        return Path(self.cloud_path.fspath).resolve(*args, **kw)
    def rglob(self, patt):
        return Path(self.cloud_path.fspath).rglob(patt)
    def rmdir(self):
        #return Path(self.cloud_path.fspath).rmdir()
        raise TypeError(f"{type(self)} filesystem access is read-only")
    def samefile(self, other):
        return Path(self.cloud_path.fspath).samefile(other)
    def symlink_to(self, *args, **kw):
        #return Path(self.cloud_path.fspath).symlink_to(target, *args, **kw)
        raise TypeError(f"{type(self)} filesystem access is read-only")
    def hardlink_to(self, target):
        #return Path(self.cloud_path.fspath).hardlink_to(target)
        raise TypeError(f"{type(self)} filesystem access is read-only")
    def link_to(self, target):
        #return Path(self.cloud_path.fspath).link_to(target)
        raise TypeError(f"{type(self)} filesystem access is read-only")
    def touch(self, *args, **kw):
        #return Path(self.cloud_path.fspath).touch(*args, **kw)
        raise TypeError(f"{type(self)} filesystem access is read-only")
    def unlink(self, *args, **kw):
        #return Path(self.cloud_path.fspath).unlink(*args, **kw)
        raise TypeError(f"{type(self)} filesystem access is read-only")
    def write_bytes(self, data):
        #return Path(self.cloud_path.fspath).write_bytes(data)
        raise TypeError(f"{type(self)} filesystem access is read-only")
    def write_text(self, *args, **kw):
        #return Path(self.cloud_path.fspath).write_text(*args, **kw)
        raise TypeError(f"{type(self)} filesystem access is read-only")
