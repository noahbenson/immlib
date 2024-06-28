# -*- coding: utf-8 -*-
################################################################################
# immlib/test/pathlib/test_osf.py

"""Tests for the OSF components of the pathlib submodule in immlib: i.e.,
tests for the code in the `immlib.pathlib._osf` module.
"""


# Dependencies #################################################################

from unittest import TestCase
from tempfile import TemporaryDirectory

from ...pathlib import *


# Tests ########################################################################

class TestPathlibOSF(TestCase):
    """Tests the immlib.pathlib._osf module."""

    def test_osf(self):
        """Tests the OSFPath type."""
        # We'll do everything in a single temporary directory.
        with TemporaryDirectory() as tmpdir:
            # Create an OSFPath.
            p0 = osfpath("osf://bw9ec/", local_cache_dir=tmpdir)
            # Test out a few subpaths...
            p = p0 / "analysis.m"
            # This one is a file.
            self.assertTrue(p.is_file())
            # We should be able to read it.
            with p.open('rt') as f:
                lns = f.readlines()
            # This particular file has 94 lines in it.
            self.assertEqual(len(lns), 94)
            # This is the first line:
            self.assertEqual(
                lns[0],
                ('% This script shows how analyzePRF_HCP7TRET was used'
                 ' to analyze the\n'))
            # Here's a directory.
            p = p0 / "rotatingsurfaces"
            self.assertTrue(p.is_dir())
            # We should be able to list the contents.
            ls = {fl.name: fl for fl in p.iterdir()}
            # There are 12 files in this particular directory; here are a few
            # of them:
            self.assertEqual(len(ls), 12)
            # Here's one example:
            f = p / "rh-angle.mov"
            self.assertEqual(f, ls["rh-angle.mov"])
            self.assertTrue(f.is_file())

            
            
            
    
