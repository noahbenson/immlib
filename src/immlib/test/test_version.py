# -*- coding: utf-8 -*-
################################################################################
# immlib/test/test_version.py

"""Tests for the version submodule of immlib: i.e., tests for the code in the
`immlib._version` module.
"""


# Dependencies #################################################################

import os
from unittest import TestCase
from tempfile import TemporaryDirectory


# Tests ########################################################################

class TestVersion(TestCase):
    """Tests for the Version type and immlib's own version object."""

    def _toml(self, tmpdir, text, name='pyproject.toml'):
        """Writes `text` to a file in `tmpdir` and returns its path."""
        p = os.path.join(tmpdir, name)
        with open(p, 'wt') as fl:
            fl.write(text)
        return p

    def test_version_parsing(self):
        """Tests that a version string is split into its components."""
        from .._version import Version
        # The ordinary three-component form.
        v = Version('1.2.3')
        self.assertEqual(v.string, '1.2.3')
        self.assertEqual(v.tuple, (1, 2, 3))
        self.assertEqual((v.major, v.minor, v.micro), (1, 2, 3))
        self.assertIsNone(v.stage)
        # Missing components default to zero and are not in the tuple's
        # stage position.
        self.assertEqual(Version('1.2').tuple, (1, 2, 0))
        self.assertEqual(Version('1').tuple, (1, 0, 0))
        self.assertEqual(Version('1').major, 1)
        # A four-component string puts the last component in stage,
        # whatever it looks like.
        v = Version('0.2.2.dev1')
        self.assertEqual(v.tuple, (0, 2, 2, 'dev1'))
        self.assertEqual(v.stage, 'dev1')
        # A stage tag attached to the last numeric component is split off,
        # at any number of components.
        for (s, want) in (('1.2.3rc4', (1, 2, 3, 'rc4')),
                          ('1.2.3a4', (1, 2, 3, 'a4')),
                          ('1.2.3b4', (1, 2, 3, 'b4')),
                          ('1.2b4', (1, 2, 0, 'b4')),
                          ('1rc2', (1, 0, 0, 'rc2'))):
            self.assertEqual(Version(s).tuple, want, s)
            self.assertEqual(Version(s).string, s)
        # tag_prefixes chooses which tags are recognized.
        self.assertEqual(
            Version('1.2.3c4', tag_prefixes=('c',)).tuple, (1, 2, 3, 'c4'))
        with self.assertRaises(ValueError):
            Version('1.2.3c4')
        # Too many components is an error.
        with self.assertRaises(ValueError):
            Version('1.2.3.4.5')
    def test_version_dunders(self):
        """Tests Version's string, iteration, and membership behavior."""
        from .._version import Version
        v = Version('1.2.3rc4')
        self.assertEqual(str(v), '1.2.3rc4')
        self.assertEqual(repr(v), "Version('1.2.3rc4')")
        self.assertEqual(list(v), [1, 2, 3, 'rc4'])
        self.assertEqual(list(reversed(v)), ['rc4', 3, 2, 1])
        # A string is looked for in the version string, anything else in
        # the tuple.
        self.assertIn('rc4', v)
        self.assertIn('1.2', v)
        self.assertNotIn('9', v)
        self.assertIn(3, v)
        self.assertNotIn(9, v)
        # It is a namedtuple, so it compares and unpacks as one.
        self.assertEqual(Version('1.2.3'), Version('1.2.3'))
        self.assertEqual(Version('1.2.3').tuple, (1, 2, 3))
    def test_version_getstring_package(self):
        """Tests looking a version up by package name."""
        from .._version import Version
        import immlib
        self.assertEqual(Version.getstring('immlib'), immlib.__version__)
        # With neither argument there is nothing to look up.
        with self.assertRaises(ValueError):
            Version.getstring()
        # A package that is not installed raises.
        with self.assertRaises(Exception):
            Version.getstring('immlib_no_such_package_xyz')
    def test_version_getstring_pyproject(self):
        """Tests reading a version out of a pyproject.toml file.

        Every real pyproject.toml has blank lines and comments in it, so the
        parser has to skip them; it used to index the first character of
        each stripped line, which raised an IndexError for a blank one.
        """
        from .._version import Version
        with TemporaryDirectory() as tmpdir:
            # The simplest possible file.
            p = self._toml(tmpdir, '[project]\nname = "x"\nversion = "1.2.3"\n')
            self.assertEqual(Version.getstring(pyproject_path=p), '1.2.3')
            # Blank lines and comments are skipped.
            p = self._toml(
                tmpdir,
                '# a comment\n\n[project]\n\n# another\nversion = "1.2.3"\n\n',
                name='blanks.toml')
            self.assertEqual(Version.getstring(pyproject_path=p), '1.2.3')
            # Only the [project] section counts: a version in another
            # section is not the project's version.
            p = self._toml(
                tmpdir,
                '[tool.poetry]\nversion = "9.9.9"\n\n[project]\nname = "y"\n',
                name='wrong_section.toml')
            with self.assertRaises(RuntimeError):
                Version.getstring(pyproject_path=p)
            # A section before [project] does not confuse it.
            p = self._toml(
                tmpdir,
                '[build-system]\nrequires = ["setuptools"]\n\n'
                '[project]\nname = "y"\nversion = "1.2.3"\n',
                name='ordered.toml')
            self.assertEqual(Version.getstring(pyproject_path=p), '1.2.3')
            # The value is read as a Python literal, so either quote works,
            # and an = inside another value does not confuse the split.
            p = self._toml(
                tmpdir,
                "[project]\nurl = \"http://x?a=b\"\nversion = '1.2.3'\n",
                name='quotes.toml')
            self.assertEqual(Version.getstring(pyproject_path=p), '1.2.3')
            # No version line at all is an error, and the message differs
            # depending on whether a package name was also tried.
            p = self._toml(tmpdir, '[project]\nname = "x"\n\n',
                           name='noversion.toml')
            with self.assertRaises(RuntimeError):
                Version.getstring(pyproject_path=p)
            with self.assertRaises(RuntimeError):
                Version.getstring('immlib_no_such_package_xyz', p)
            # A package name that is not installed falls back to the file.
            p = self._toml(tmpdir, '[project]\nversion = "1.2.3"\n',
                           name='fallback.toml')
            self.assertEqual(
                Version.getstring('immlib_no_such_package_xyz', p), '1.2.3')
            # A package name that *is* installed wins over the file.
            self.assertEqual(
                Version.getstring('immlib', p), Version.getstring('immlib'))
    def test_version_on_error(self):
        """Tests the on_error option of the Version constructor."""
        import warnings
        from .._version import Version
        # An unknown setting is rejected before anything else happens.
        with self.assertRaises(ValueError):
            Version('1.2.3', on_error='bogus')
        # 'raise' lets the lookup failure out.
        with self.assertRaises(Exception):
            Version(package_name='immlib_no_such_package_xyz',
                    on_error='raise')
        # 'warn' warns and gives the null version.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            v = Version(package_name='immlib_no_such_package_xyz',
                        on_error='warn')
            self.assertEqual(len(caught), 1)
        self.assertIs(v, Version.null)
        # 'ignore' does the same without the warning.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            v = Version(package_name='immlib_no_such_package_xyz',
                        on_error='ignore')
            self.assertEqual(len(caught), 0)
        self.assertIs(v, Version.null)
        # The null version is empty in every component.
        self.assertEqual(Version.null.string, '')
        self.assertEqual(Version.null.tuple, ())
        self.assertIsNone(Version.null.major)
        self.assertIsNone(Version.null.stage)
    def test_immlib_version(self):
        """Tests that immlib's own version object is consistent."""
        import immlib
        from .._version import Version, version, pyproject_path
        self.assertIsInstance(version, Version)
        self.assertEqual(version.string, immlib.__version__)
        self.assertIsInstance(version.major, int)
        self.assertIsInstance(version.minor, int)
        self.assertIsInstance(version.micro, int)
        self.assertEqual(
            version.tuple[:3], (version.major, version.minor, version.micro))
        # The installed version and the one in the repository's
        # pyproject.toml agree, when that file is where it is expected to
        # be (it is not, when immlib is installed from a wheel).
        if os.path.isfile(pyproject_path):
            self.assertEqual(
                Version.getstring(pyproject_path=pyproject_path),
                version.string)
