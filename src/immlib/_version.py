# -*- coding: utf-8 -*-
################################################################################
# immlib/_version.py
#
# Loads the version and splits it into meaningful pieces.


import os, re
from collections import namedtuple

try:
    from importlib.metadata import version as get_version
    from importlib.metadata import PackageNotFoundError
except ModuleNotFoundError:
    try:
        from importlib_metadata import version as get_version
        from importlib_metadata import PackageNotFoundError
    except ModuleNotFoundError:
        pass


# ImmLibVersion Type ###########################################################

_ImmLibVersionBase = namedtuple(
    '_ImmLibVersionBase',
    ['string', 'major', 'minor', 'micro', 'stage', 'tuple'])
class ImmLibVersion(_ImmLibVersionBase):
    """The type that manages the version information for the immlib package."""
    __slots__ = ()
    def __new__(cls, string=None):
        if string is None:
            # Try to deduce the version string.
            try:
                string = get_version(__package__)
            except PackageNotFoundError:
                # Probably immlib isn't installed via pip or poetry so it
                # doesn't show up as a registered module. As a backup, we can
                # grab it from the pyproject.toml file.
                base_path = os.path.join(os.path.split(__file__)[0], '..', '..')
                pyproject_toml_path = os.path.join(base_path, 'pyproject.toml')
                with open(pyproject_toml_path, 'r') as fl:
                    pyproject_toml_lines = fl.read().split('\n')
                for ln in pyproject_toml_lines:
                    ln = ln.strip()
                    if ln.startswith('version = '):
                        string = ln.split('"')[1]
                        break
        if string is None:
            from warnings import warn
            warn("immlib could not detect its version number")
            (major, minor, micro, stage) = (None, None, None, None)
        else:
            s = string
            ss = s.split('.')
            if len(ss) == 4:
                (major, minor, micro, stage) = ss
            else:
                if len(ss) == 3:
                    (major, minor, micro) = ss
                elif len(ss) == 2:
                    (major, minor) = ss
                    micro = '0'
                stage = None
                for tag in ('rc', 'a', 'b'):
                    if tag in micro:
                        (micro, stage) = micro.split(tag)
                        stage = tag + stage
                        break
            major = int(major)
            minor = int(minor)
            micro = int(micro)
        tup = tuple(u for u in (major, minor, micro, stage) if u is not None)
        return super(ImmLibVersion, cls).__new__(
            cls, 
            string=string,
            major=major, minor=minor, micro=micro,
            stage=stage,
            tuple=tup)
    def __str__(self):
        return self.string
    def __repr__(self):
        return f"ImmLibVersion({repr(self.string)})"
    def __iter__(self):
        return iter(self.tuple)
    def __reversed__(self):
        return reversed(self.tuple)
    def __contains__(self, k):
        if isinstance(k, str):
            return k in self.string
        else:
            return k in self.tuple

# Declare the version object; this will automatically detect the version.
version = ImmLibVersion()
