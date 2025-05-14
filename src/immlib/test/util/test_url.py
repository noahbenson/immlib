# -*- coding: utf-8 -*-
################################################################################
# immlib/test/util/test_url.py

"""Tests of the URL utilities module in immlib: i.e., tests for the code in the
immlib.util._url module.
"""


# Dependencies #################################################################

from unittest import TestCase


# Tests ########################################################################

lic_url = (
    'https://raw.githubusercontent.com/noahbenson/immlib'
    '/refs/heads/main/LICENSE')

class TestUtilURL(TestCase):
    """Tests the immlib.util._url module."""
    def test_is_url(self):
        "Tests the is_url function."
        from immlib.util import is_url
        self.assertTrue(is_url(lic_url))
        self.assertTrue(is_url('https://github.com/noahbenson/immlib.git'))
        self.assertTrue(is_url('s3://natural-scenes-dataset/'))
        self.assertTrue(is_url('file:///etc/groups'))
        self.assertFalse(is_url(None))
        self.assertFalse(is_url(10))
        self.assertFalse(is_url('test'))
    def test_can_download_url(self):
        "Tests the can_download_url function."
        from immlib.util import can_download_url
        self.assertTrue(can_download_url(lic_url))
        self.assertTrue(can_download_url('https://github.com/noahbenson'))
        badurl = 'https://github.com/noahbenson/no-repo-here'
        self.assertFalse(can_download_url(badurl))
    def test_url_download(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from immlib.util import url_download
        b = url_download(lic_url)
        self.assertIn(b'MIT License', b)
        with TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            url_download(lic_url, p / "LICENSE")
            with (p / "LICENSE").open('rt') as fl:
                dat = fl.readlines()
            self.assertEqual(dat[0].strip(), "MIT License")
