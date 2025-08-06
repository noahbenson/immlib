# -*- coding: utf-8 -*-
###############################################################################
# immlib/_version.py


# Dependencies ################################################################

from ast import literal_eval
from collections import namedtuple
from pathlib import Path
from warnings import warn


# Version Type ################################################################

VersionTuple = namedtuple(
    'VersionTuple',
    ['string', 'tuple', 'major', 'minor', 'micro', 'stage'])
class Version(VersionTuple):
    """A type that represents a Python package version.

    Python packages are represented simultaneously as version strings, version
    tuples, and by the version components ``major``, ``minor``, ``micro``, and
    ``stage``.

    Parameters
    ----------
    string : str or None, optional
        The version string to be represented. If this argument is not provided,
        then one or both of the ``package_name`` and ``pyproject_path`` options
        must be provided so that the version string can be obtained via the
        package version or the ``pyproject.toml`` file.
    package_name : str or None, optional
        If the first argument (``string``) is provided, then this argument is
        ignored; otherwise, the version string is first searched for by this
        package name using the ``importlib`` or ``importlib_metadata``
        packages. If found, then this version string is represented in the
        ``Version`` object.
    pyproject_path : path-like or None, optional
        If the first argument (``string``) is not given and the
        ``package_name`` is not given, then the version is searched for in the
        ``pyproject.toml`` file given by this path. In order for such a file to
        be valid, it must contain a line that, when stripped of whitespace,
        begins with the string ``'version='`` followed by a string
        representation (e.g., ``'version="1.12.5"'``). If such a line is found
        in the ``[project]`` section of the TOM: file pointed to by this
        argument, then it is represented as the version string in the
        ``Version`` object.
    on_error : {'warn' | 'ignore' | 'raise'}, optional
        How to handle failures to deduce or parse the version number. If
        ``'raise'`` is given, then the errors are allowed to be raised. If
        ``'warn'``, then a warning is raised and a null version is returned. If
        ``'ignore'``, then errors are ignored and a null version is
        returned. The default is ``'raise'``.
    
    tag_prefixes : tuple of str, optional
        An optional tuple of strings that can appear as the prefixes of stage
        tagss at the end of the version string. By default, this is ``('rc',
        'a', 'b')``, so version strings like ``'1.1.12a6'`` and ``'1.1.12rc6'``
        are valid but ``'1.1.12c6'`` is not.
    
    Attributes
    ----------
    string : str
        The string representing the package version. For example ``"1.2.15"``
        or ``"0.2.2.dev1"``.
    tuple : tuple of int and str
        The components of the version string, for example, ``(1, 2, 15)`` or
        ``(0, 2, 2, 'dev1')``. Any missing component is excluded.
    major : int
        The major version number, typically indicates major API version.
    minor : int
        The minor version number, typically indicates minor API version.
    micro : int
        The micro version number, typically indicates patch increment number.
    stage : str
        The development stage of the version. For example ``'dev1'`` or
        ``'rc2'``.
    """

    # Static Methods ----------------------------------------------------------
    def getstring(package_name=None, pyproject_path=None):
        """Returns the current version string for the given package name.
        
        ``Version.getstring(package_name)`` returns the version string of the
        package with the given package name.
        
        ``Version.getstring(pyproject_path=path)`` returns the version string
        found in the pyproject.toml file found at the given ``path``.
        
        ``Version.getstring(package_name, path)`` returns
        ``Version.getstring(package_name)`` if the given ``package_name`` is
        found, otherwise returns ``Version.getstring(pyproject_path=path)``.
        """
        if package_name is None and pyproject_path is None:
            raise ValueError("Version.getstring() requires 1 or 2 arguments")
        if package_name is not None:
            try:
                from importlib.metadata import version
                from importlib.metadata import PackageNotFoundError
            except ModuleNotFoundError:
                from importlib_metadata import version
                from importlib_metadata import PackageNotFoundError
            if pyproject_path is None:
                return version(package_name)
            # Try to deduce the version string but don't raise if this fails.
            try:
                return version(package_name)
            except PackageNotFoundError:
                pass
        # Either a package name wasn't given or the package wasn't found; check
        # the pyproject.toml if possible.
        path = Path(pyproject_path)
        with path.open('rt') as fl:
            toml_lines = fl.read().split('\n')
        in_project_section = False
        for ln in toml_lines:
            ln = ln.strip()
            if ln == '[project]':
                in_project_section = True
            elif ln[0] == '[' and ln[-1] == ']':
                in_project_section = False
            elif in_project_section and '=' in ln:
                parts = ln.split('=')
                if parts[0].strip() == 'version':
                    v = '='.join(parts[1:]).strip()
                    return literal_eval(v)
        # If we reach this point, we didn't fine a version line.
        if package_name is not None:
            raise RuntimeError(
                f"Version.getstring() found no package named '{package_name}'"
                f" and no 'version = ...' line in file {path}")
        else:
            raise RuntimeError(
                f"Version.getstring() found no 'version = ...' line in file"
                f" {path}")

    # Construction ------------------------------------------------------------
    __slots__ = ()
    null = None
    def __new__(cls, string=None, /, *,
                package_name=None,
                pyproject_path=None,
                on_error='warn',
                tag_prefixes=('rc', 'a', 'b')):
        if on_error not in ('raise', 'warn', 'ignore'):
            raise ValueError(
                "invalid value for on_error; must be one of 'raise', 'warn',"
                " or 'ignore'")
        if string is None:
            # Try to deduce the version string.
            if on_error == 'raise':
                string = Version.getstring(package_name, pyproject_path)
            else:
                try:
                    string = Version.getstring(package_name, pyproject_path)
                except Exception as e:
                    if on_error == 'warn':
                        warn(
                            f"Version: could not deduce version string for"
                            f" package_name={repr(package_name)} and"
                            f" pyproject_path={repr(pyproject_path)}")
                        return Version.null
                    else:
                        return Version.null
        ss = string.split('.')
        nss = len(ss)
        if nss == 4:
            (major, minor, micro, stage) = ss
        else:
            if nss == 3:
                (major, minor, micro) = ss
                last = micro
            elif nss == 2:
                (major, minor) = ss
                micro = '0'
                last = minor
            elif nss == 1:
                major = ss
                (minor, micro) = ('0', '0')
                last = major
            else:
                raise ValueError(
                    f"invalid version string: '{string}' contains {nss}"
                    f" components")
            stage = None
            for tag in tag_prefixes:
                if tag in last:
                    (num, stage) = last.split(tag)
                    stage = tag + stage
                    if last is micro:
                        micro = num
                    elif last is minor:
                        minor = num
                    else:
                        major = num
                    break
        major = int(major)
        minor = int(minor)
        micro = int(micro)
        tup = tuple(u for u in (major, minor, micro, stage) if u is not None)
        return super(Version, cls).__new__(
            cls,
            string=string,
            tuple=tup,
            major=major,
            minor=minor,
            micro=micro,
            stage=stage)
    def __str__(self):
        return self.string
    def __repr__(self):
        return f"Version({repr(self.string)})"
    def __iter__(self):
        return iter(self.tuple)
    def __reversed__(self):
        return reversed(self.tuple)
    def __contains__(self, k):
        if isinstance(k, str):
            return k in self.string
        else:
            return k in self.tuple
Version.null = VersionTuple.__new__(
    Version,
    string='',
    tuple=(),
    major=None,
    minor=None,
    micro=None,
    stage=None)


# Variables ###################################################################

# The path of immlib's pyproject.toml file.
pyproject_path = Path(__file__).parent.parent.parent / 'pyproject.toml'

# Declare the version object; this will automatically detect the version.
version = Version(package_name=__package__, pyproject_path=pyproject_path)
