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
