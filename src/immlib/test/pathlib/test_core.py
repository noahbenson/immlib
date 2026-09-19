# -*- coding: utf-8 -*-
################################################################################
# immlib/test/pathlib/test_core.py

"""Tests for the core pathlib submodule in immlib: i.e., tests for the code in
the `immlib.pathlib._core` module.
"""


# Dependencies #################################################################

from unittest import TestCase
from tempfile import TemporaryDirectory
from pathlib  import (Path, WindowsPath)

from cloudpathlib import (CloudPath, AzureBlobPath, S3Path, GSPath)

from ...pathlib import *


# Tests ########################################################################

# Some example path strings to use.
# Unfortunately, AzureBlobPaths require credentials, so we do not test these
# paths in the GitHub Actions automated testing (for now). These are tested by
# the cloudpathlib library, but the wrapper functions are not completely tested.
pathstrs = [
    "/etc/ssh/",
    "file://immlib/pathlib/__init__.py",
    #"az://storageaccount.blob.core.windows.net/container/blob/path",
    "osf://bw9ec/analysis.m",
    "s3://openneuro.org/ds003787/derivatives/",
    "gs://gcp-public-data-landsat/LC08/01/044/034/"]
pathtypes = [
    Path,
    Path,
    #AzureBlobPath,
    OSFPath,
    S3Path,
    GSPath]
pathfns = {
    Path:          (filepath, is_filepath, like_filepath),
    AzureBlobPath: (None,     is_azpath,   like_azpath),
    OSFPath:       (osfpath,  is_osfpath,  like_osfpath),
    S3Path:        (s3path,   is_s3path,   like_s3path),
    GSPath:        (gspath,   is_gspath,   like_gspath)}

# The test class.
class TestPathlibCore(TestCase):
    """Tests the immlib.pathlib._core module."""

    def cleanup_cache(self, p):
        """Given a path with a temporary directory, calls its cleanup routine.
        """
        if isinstance(p, CloudPath):
            cl = p.client
            if cl and cl._cache_tmp_dir:
                cl._cache_tmp_dir.cleanup()
    def test_pathstr(self):
        """Tests the `path` and `pathstr` functions."""
        # Path strings for basic strings should remain unchanged.
        for p in pathstrs:
            self.assertIs(p, pathstr(p))
        # Path strings for path objects should equal the original strings.
        paths = [path(p) for p in pathstrs]
        for (p, tt) in zip(paths, pathtypes):
            self.assertIsInstance(p, tt)
        # Note that Path strips trailing slashes while CloudPath does not.
        for (p,ps,pt) in zip(paths, pathstrs, pathtypes):
            s = pathstr(p)
            if isinstance(p, Path):
                ps = ps.rstrip('/')
            # if ps starts with file:// that will get stripped on a normal path.
            if ps.startswith('file://'):
                ps = ps[7:]
            if isinstance(p, WindowsPath):
                ps = ps.replace('/', '\\')
            self.assertEqual(pathstr(p), ps)
        # Adding to the path preserves a reasonable string:
        correct_path = "/etc/ssh/sshd_config"
        if isinstance(paths[0], WindowsPath):
            correct_path = correct_path.replace('/', '\\')
        self.assertEqual(pathstr(paths[0] / "sshd_config"), correct_path)
        # The path and pathstr functions should fail for non-string inputs.
        with self.assertRaises(TypeError):
            path(10)
        with self.assertRaises(TypeError):
            pathstr(10)
        # Before we exit, it's good form to cleanup the temporary directories
        # that we created.
        for p in paths:
            self.cleanup_cache(p)
    def test_pathfns(self):
        """Tests the various path-related functions for each path type."""
        from tempfile import TemporaryDirectory
        # We make a single temporary directory for this whole test.
        with TemporaryDirectory() as tempdir:
            # We'll now go through each of the path types and check their
            # functions using the pathstrs and pathtypes.
            for (ps, pt) in zip(pathstrs, pathtypes):
                (make, test, like) = pathfns[pt]
                if make is None:
                    continue
                if pt is Path:
                    p = make(ps)
                    pp = path(ps)
                else:
                    p = make(ps, local_cache_dir=tempdir)
                    pp = path(ps, local_cache_dir=tempdir)
                # The paths made with either method should be equivalent.
                self.assertEqual(pathstr(p), pathstr(pp))
                self.assertIs(type(p), type(pp))
                self.assertTrue(test(p))
                self.assertFalse(test(ps))
                self.assertTrue(like(p))
                self.assertTrue(like(ps))
                # If we get the local cache paths, they should also be equal;
                # (if p is a filepath, then this just returns their pathstrs).
                self.assertEqual(pathstr(filepath(p)), pathstr(filepath(pp)))
                # This path should be an instance of its own path type.
                self.assertIsInstance(p, pt)
                if '://' in ps:
                    # If there's a sheme at the beginning of the path, then it
                    # should fail to construct via other path functions.
                    for (k,(mk,tt,lk)) in pathfns.items():
                        if k is pt or k is Path:
                            continue
                        if mk is not None:
                            with self.assertRaises(Exception):
                                mk(ps, local_cache_dir=tempdir)
                        self.assertFalse(tt(p))
                        self.assertFalse(tt(ps))
                        self.assertFalse(lk(p))
                        self.assertFalse(lk(ps))
    def test_misc(self):
        from immlib.pathlib import (is_path, like_path, pathdict, path)
        import immlib as il
        p = path('osf://tery8/')
        self.assertTrue(is_path(p))
        self.assertTrue(like_path(p))
        self.assertFalse(is_path(10))
        self.assertFalse(like_path('blarg://nothing'))
        d = pathdict(path(il.__file__).parent)
        for k in ('util', 'math', 'types', 'workflow'):
            self.assertIn(k, d)
        self.assertTrue(path(d['util']['__init__.py']).is_file())

    # A connection string for the Azure development storage emulator. It is
    # the well-known public one from Microsoft's own documentation, and no
    # test here contacts the endpoint: AzureBlobClient refuses to be built
    # without credentials, and building the client is all that is needed to
    # build a path.
    AZURE_DEV_CONNECTION_STRING = (
        "DefaultEndpointsProtocol=https;AccountName=devstoreaccount1;"
        "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UV"
        "ErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;"
        "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;")

    def test_cloud_path_constructors(self):
        """Tests the s3path, gspath, and azpath constructors.

        None of these contacts a server: building a client and a path is
        entirely local, which is what is tested here.
        """
        with TemporaryDirectory() as tmpdir:
            az_kw = {'connection_string': self.AZURE_DEV_CONNECTION_STRING}
            cases = (('s3', s3path, S3Path, is_s3path, like_s3path, {}),
                     ('gs', gspath, GSPath, is_gspath, like_gspath, {}),
                     ('az', azpath, AzureBlobPath, is_azpath, like_azpath,
                      az_kw))
            for (scheme, fn, cls, isfn, likefn, kw) in cases:
                # A full URL, and a bare path that gets the scheme added.
                p = fn(f'{scheme}://bucket/key', cache_path=tmpdir, **kw)
                self.assertIsInstance(p, cls)
                self.assertEqual(str(p), f'{scheme}://bucket/key')
                q = fn('bucket/key', cache_path=tmpdir, **kw)
                self.assertEqual(str(q), str(p))
                # The predicates agree about what these are.
                self.assertTrue(isfn(p))
                self.assertTrue(likefn(p))
                self.assertTrue(likefn(f'{scheme}://bucket/key'))
                self.assertFalse(isfn(f'{scheme}://bucket/key'))
                # Extra arguments are joined onto the path.
                p = fn(f'{scheme}://bucket', 'a', 'b', cache_path=tmpdir, **kw)
                self.assertEqual(str(p), f'{scheme}://bucket/a/b')
                # cache_path gets a subdirectory named for the scheme, so
                # that two backends do not share one cache directory.
                self.assertEqual(
                    Path(p.client._local_cache_dir),
                    Path(tmpdir) / scheme)
                # Passing a path of the right type back in reuses its
                # client rather than building another.
                p2 = fn(p)
                self.assertIs(p2.client, p.client)
                self.assertEqual(str(p2), str(p))
                self.assertEqual(str(fn(p, 'c')), f'{scheme}://bucket/a/b/c')
                # local_cache_dir overrides cache_path when both are given.
                other = str(Path(tmpdir) / 'explicit')
                p = fn(f'{scheme}://bucket/key', cache_path=tmpdir,
                       local_cache_dir=other, **kw)
                self.assertEqual(Path(p.client._local_cache_dir), Path(other))
                # With neither, immlib names no cache directory and
                # cloudpathlib picks a temporary one of its own.
                p = fn(f'{scheme}://bucket/key', **kw)
                self.assertNotEqual(
                    Path(p.client._local_cache_dir), Path(tmpdir) / scheme)
    def test_interp_cache(self):
        """Tests _interp_cache, which reconciles cache_path with
        local_cache_dir."""
        from ...pathlib._core import _interp_cache
        # An explicit local_cache_dir wins, whatever cache_path says.
        self.assertEqual(_interp_cache('/tmp/y', '/tmp/x', 's3'), '/tmp/y')
        self.assertIsNone(_interp_cache(None, '/tmp/x', 's3'))
        # With no cache_path there is nothing to build.
        self.assertIsNone(_interp_cache(Ellipsis, None, 's3'))
        # Otherwise the tag becomes a subdirectory of cache_path, and the
        # user directory is expanded.
        self.assertEqual(
            _interp_cache(Ellipsis, '/tmp/x', 's3'), Path('/tmp/x/s3'))
        self.assertEqual(
            _interp_cache(Ellipsis, '/tmp/x', None), Path('/tmp/x'))
        self.assertEqual(
            _interp_cache(Ellipsis, '~/x', 'gs'),
            Path.home() / 'x' / 'gs')
    def test_pathtype_and_path_dispatch(self):
        """Tests pathtype and the path function's dispatch by scheme."""
        from ...pathlib._core import pathtype, pathtypes
        with TemporaryDirectory() as tmpdir:
            # The scheme of a string picks the constructor.
            self.assertIs(pathtype('s3://b/k'), pathtypes['s3'])
            self.assertIs(pathtype('gs://b/k'), pathtypes['gs'])
            self.assertIs(pathtype('az://b/k'), pathtypes['az'])
            self.assertIs(pathtype('osf://b/k'), pathtypes['osf'])
            # Bytes are decoded first.
            self.assertIs(pathtype(b's3://b/k'), pathtypes['s3'])
            # With no scheme at all, the default applies.
            self.assertIs(pathtype('/tmp/x'), pathtypes['file'])
            self.assertIs(pathtype('/tmp/x', default='s3'), pathtypes['s3'])
            # An unknown scheme is a KeyError.
            with self.assertRaises(KeyError):
                pathtype('nosuchscheme://b/k')
            # path() dispatches on the same rule and joins extra arguments.
            p = path('s3://bucket/key', cache_path=tmpdir)
            self.assertIsInstance(p, S3Path)
            p = path('gs://bucket', 'a', 'b', cache_path=tmpdir)
            self.assertEqual(str(p), 'gs://bucket/a/b')
            # A path given back with no new options is returned as it is.
            self.assertIs(path(p), p)
            # And with extra arguments it is extended.
            self.assertEqual(str(path(p, 'c')), 'gs://bucket/a/b/c')
            # An unrecognized scheme becomes a ValueError from path(),
            # rather than the KeyError that pathtype raises.
            with self.assertRaises(ValueError):
                path('nosuchscheme://b/k')
    def test_filepath_arguments(self):
        """Tests filepath's handling of each kind of argument."""
        from ...pathlib._core import filepath
        with TemporaryDirectory() as tmpdir:
            # A str, with or without the file:// scheme.
            self.assertEqual(filepath('/tmp/x'), Path('/tmp/x'))
            self.assertEqual(filepath('file:///tmp/x'), Path('/tmp/x'))
            # Bytes, likewise; Path does not accept bytes, so these are
            # decoded rather than passed through, which used to raise.
            self.assertEqual(filepath(b'/tmp/x'), Path('/tmp/x'))
            self.assertEqual(filepath(b'file:///tmp/x'), Path('/tmp/x'))
            # An existing Path or PathLike object.
            self.assertEqual(filepath(Path('/tmp/x')), Path('/tmp/x'))
            # A CloudPath becomes the CloudCachePath that stands for its
            # place in the cache.
            from ...pathlib._cache import CloudCachePath
            cp = s3path('s3://bucket/key', cache_path=tmpdir)
            fp = filepath(cp)
            self.assertIsInstance(fp, CloudCachePath)
            self.assertIs(fp.cloud_path, cp)
    def test_pathdict_callbacks(self):
        """Tests pathdict's onfile and ondir callbacks."""
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / 'sub').mkdir()
            (root / 'a.txt').write_text('x')
            (root / 'sub' / 'b.txt').write_text('y')
            # Without callbacks the leaves are paths and the branches are
            # lazy dictionaries.
            d = pathdict(root)
            self.assertEqual(set(d), {'a.txt', 'sub'})
            self.assertEqual(set(d['sub']), {'b.txt'})
            # onfile transforms each file.
            d = pathdict(root, onfile=lambda p: p.name.upper())
            self.assertEqual(d['a.txt'], 'A.TXT')
            self.assertEqual(d['sub']['b.txt'], 'B.TXT')
            # A file rather than a directory is returned as it is, or
            # passed through onfile when one is given.
            self.assertEqual(pathdict(root / 'a.txt'), root / 'a.txt')
            self.assertEqual(
                pathdict(root / 'a.txt', onfile=lambda p: p.name), 'a.txt')
            # ondir replaces the recursion.
            d = pathdict(root, ondir=lambda p, **kw: p.name)
            self.assertEqual(d['sub'], 'sub')
    def test_is_windows_drive(self):
        """Tests is_windows_drive, which is always False off Windows."""
        from ...pathlib._core import is_windows_drive
        import platform as _platform
        result = is_windows_drive('c')
        self.assertIsInstance(result, bool)
        if 'windows' not in _platform.platform().lower():
            self.assertFalse(result)
            self.assertFalse(is_windows_drive('z'))
    def test_osfpath_constructor(self):
        """Tests the osfpath constructor.

        Building an OSFPath contacts nothing: the project's contents are
        fetched lazily, when the path is used.
        """
        with TemporaryDirectory() as tmpdir, TemporaryDirectory() as other:
            # A full URL and a bare project id both work.
            p = osfpath('osf://bw9ec', cache_path=tmpdir)
            self.assertTrue(is_osfpath(p))
            self.assertEqual(str(p), 'osf://bw9ec')
            self.assertEqual(str(osfpath('bw9ec', cache_path=tmpdir)),
                             'osf://bw9ec')
            # cache_path gets an 'osf' subdirectory, as the other schemes
            # get one named for themselves.
            self.assertEqual(
                Path(p.client._local_cache_dir), Path(tmpdir) / 'osf')
            # Extra arguments are joined on.
            self.assertEqual(str(osfpath('osf://bw9ec', 'a', 'b',
                                         cache_path=tmpdir)),
                             'osf://bw9ec/a/b')
            # Passing a path back in reuses its client, and so keeps its
            # cache directory: it used to build a new client with no cache
            # directory, silently losing the one that had been set up.
            q = osfpath(p)
            self.assertIs(q.client, p.client)
            self.assertEqual(
                Path(q.client._local_cache_dir), Path(tmpdir) / 'osf')
            q = osfpath(p, 'sub')
            self.assertEqual(str(q), 'osf://bw9ec/sub')
            self.assertEqual(
                Path(q.client._local_cache_dir), Path(tmpdir) / 'osf')
            # Asking for a different cache does build a new client.
            q = osfpath(p, cache_path=other)
            self.assertIsNot(q.client, p.client)
            self.assertEqual(
                Path(q.client._local_cache_dir), Path(other) / 'osf')
            # An explicit client is used as it is.
            q = osfpath('osf://bw9ec', client=p.client)
            self.assertIs(q.client, p.client)
