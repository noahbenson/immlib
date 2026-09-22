# -*- coding: utf-8 -*-
################################################################################
# immlib/test/pathlib/test_osf.py

"""Tests for the OSF components of the pathlib submodule: the code in the
`immlib.pathlib._osf` module.

The OSF client talks to the network only through `url_download`, so these
tests serve a canned project by replacing that function (see
`_osf_fixture.osf_mock`) rather than contacting osf.io. The live API returns
503s often enough that a network-dependent OSF test is not a reliable one.
"""


# Dependencies #################################################################

import os
from unittest import TestCase
from tempfile import TemporaryDirectory
from pathlib import Path
from datetime import datetime

from pcollections import ldict
from cloudpathlib.cloudpath import NoStatError

from ...pathlib import (osfpath, OSFPath, OSFClient)
from ...pathlib._osf import (osf_contents, _osf_timestamp)

from ._osf_fixture import (
    osf_mock, PROJECT, STORAGE, BASE, ROOT_URL, SUB_URL,
    page, file_entry, dir_entry, DEFAULT_PAGES, DEFAULT_DOWNLOADS)


# Tests ########################################################################

class TestPathlibOSF(TestCase):
    """Tests the immlib.pathlib._osf module against a canned OSF project."""

    # Helpers ------------------------------------------------------------------
    def _path(self, tmpdir, path='a.txt'):
        "Builds an OSFPath for a fixture path in the mocked project."
        return osfpath(f'osf://{PROJECT}/{path}', local_cache_dir=tmpdir)

    # Timestamp parsing --------------------------------------------------------
    def test_timestamp(self):
        """`_osf_timestamp` parses OSF's ISO-8601 timestamps, tolerantly."""
        want = datetime(2020, 1, 2, 3, 4, 5).timestamp()
        self.assertEqual(_osf_timestamp('2020-01-02T03:04:05.000000Z'), want)
        self.assertEqual(_osf_timestamp('2020-01-02T03:04:05Z'), want)
        self.assertEqual(_osf_timestamp('2020-01-02T03:04:05'), want)
        # A missing or unparseable timestamp is zero, not an error.
        self.assertEqual(_osf_timestamp(None), 0)
        self.assertEqual(_osf_timestamp(''), 0)
        self.assertEqual(_osf_timestamp('not a date'), 0)

    # osf_contents -------------------------------------------------------------
    def test_contents_structure(self):
        """`osf_contents` builds the nested contents of a project."""
        with osf_mock(), TemporaryDirectory() as tmp:
            c = osf_contents(PROJECT, lazy=False, cache_path=tmp)
            self.assertEqual(c['kind'], 'directory')
            self.assertEqual(set(c['contents'].keys()), {'a.txt', 'sub'})
            a = c['contents']['a.txt']
            self.assertEqual(a['kind'], 'file')
            self.assertEqual(a['download_url'], 'osfdl://a.txt')
            self.assertEqual(a['size'], 11)
            sub = c['contents']['sub']
            self.assertEqual(sub['kind'], 'directory')
            self.assertEqual(set(sub['contents'].keys()), {'b.txt'})
            self.assertEqual(sub['contents']['b.txt']['size'], 5)

    def test_contents_lazy(self):
        """`osf_contents` does not consult the network until it is asked to."""
        with osf_mock() as calls, TemporaryDirectory() as tmp:
            c = osf_contents(PROJECT, cache_path=tmp)
            self.assertIsInstance(c, ldict)
            self.assertEqual(c['kind'], 'directory')
            # Nothing has been fetched yet...
            self.assertEqual(calls, [])
            # ...until the contents are actually requested.
            self.assertEqual(set(c['contents'].keys()), {'a.txt', 'sub'})
            self.assertTrue(calls)

    def test_contents_paging(self):
        """`osf_contents` follows the OSF paging link across pages."""
        p2 = BASE + '?page[size]=100&page=2'
        pages = {
            ROOT_URL: page([file_entry('a.txt', 'osfdl://a.txt', 11)],
                           next_url=p2),
            p2: page([file_entry('c.txt', 'osfdl://c.txt', 3)])}
        with osf_mock(pages=pages, downloads={'osfdl://c.txt': b'abc'}):
            c = osf_contents(PROJECT, lazy=False)
            self.assertEqual(set(c['contents'].keys()), {'a.txt', 'c.txt'})
            self.assertEqual(c['contents']['c.txt']['size'], 3)

    def test_contents_page_cache(self):
        """`osf_contents` writes a page cache that a later call reuses."""
        with osf_mock() as calls, TemporaryDirectory() as tmp:
            osf_contents(PROJECT, lazy=False, cache_path=tmp)
            cache_file = (
                Path(tmp) / PROJECT / STORAGE
                / f'.p0_100.{_osf_treecache_name()}')
            self.assertTrue(cache_file.is_file())
            # The second call reads the page from the cache, not the network.
            calls.clear()
            osf_contents(PROJECT, lazy=False, cache_path=tmp)
            self.assertEqual(calls, [])

    def test_contents_file_root_error(self):
        """A project root that the API reports as a file is an error."""
        pages = {ROOT_URL: {'data': {}, 'links': {}}}
        with osf_mock(pages=pages):
            with self.assertRaises(RuntimeError):
                osf_contents(PROJECT, lazy=False)

    def test_contents_missing_data_error(self):
        """A page with no `data` is an error."""
        pages = {ROOT_URL: {'links': {}}}
        with osf_mock(pages=pages):
            with self.assertRaises(ValueError):
                osf_contents(PROJECT, lazy=False)

    # OSFPath ------------------------------------------------------------------
    def test_path_file(self):
        """An OSFPath to a file reports the file's properties."""
        with osf_mock(), TemporaryDirectory() as tmp:
            p = self._path(tmp, 'a.txt')
            self.assertEqual(p.bucket, PROJECT)
            self.assertEqual(p.project_id, PROJECT)
            self.assertTrue(p.is_file())
            self.assertFalse(p.is_dir())
            self.assertTrue(p.exists())
            self.assertTrue(p.key.endswith('a.txt'))
            st = p.stat()
            self.assertEqual(st.st_size, 11)
            self.assertGreater(st.st_mtime, 0)
            self.assertEqual(p.read_bytes(), b'hello world')
            with p.open('rt') as fl:
                self.assertEqual(fl.read(), 'hello world')

    def test_path_directory(self):
        """An OSFPath to a directory lists and rejects stats."""
        with osf_mock(), TemporaryDirectory() as tmp:
            p = self._path(tmp, 'sub')
            self.assertTrue(p.is_dir())
            self.assertFalse(p.is_file())
            self.assertEqual(p.bucket, PROJECT)
            names = {fl.name for fl in p.iterdir()}
            self.assertEqual(names, {'b.txt'})
            # Directories have no stats.
            with self.assertRaises(NoStatError):
                p.stat()

    def test_path_missing(self):
        """A path that does not exist reports so, and does not stat."""
        with osf_mock(), TemporaryDirectory() as tmp:
            p = self._path(tmp, 'nope.txt')
            self.assertFalse(p.exists())
            with self.assertRaises(FileNotFoundError):
                p.is_file()

    def test_path_invalid_url(self):
        """An OSFPath rejects a non-OSF URL."""
        with self.assertRaises(ValueError):
            OSFPath('http://example.com/not-osf')

    def test_path_drive(self):
        """The drive of an OSF path is its project (or project:storage)."""
        with osf_mock(), TemporaryDirectory() as tmp:
            p = self._path(tmp, 'a.txt')
            self.assertEqual(p.drive.split(':')[0], PROJECT)

    # Read-only behavior -------------------------------------------------------
    def test_client_read_only(self):
        """All mutating client operations raise."""
        with osf_mock(), TemporaryDirectory() as tmp:
            p = self._path(tmp, 'a.txt')
            client = p.client
            with self.assertRaises(RuntimeError):
                client._remove(p)
            with self.assertRaises(RuntimeError):
                client._upload_file(Path(tmp), p)
            with self.assertRaises(RuntimeError):
                client._move_file(p, p)
            with self.assertRaises(NotImplementedError):
                client._list_dir(p, recursive=True)
            with self.assertRaises(TypeError):
                client._get_public_url(p)
            with self.assertRaises(TypeError):
                client._generate_presigned_url(p)

    def test_path_read_only(self):
        """All mutating CloudPath operations raise."""
        with osf_mock(), TemporaryDirectory() as tmp:
            p = self._path(tmp, 'a.txt')
            with self.assertRaises(TypeError):
                p.mkdir()
            with self.assertRaises(TypeError):
                p.touch()

    def test_client_wrong_type(self):
        """Client methods reject non-OSF paths."""
        with osf_mock(), TemporaryDirectory() as tmp:
            client = self._path(tmp, 'a.txt').client
            with self.assertRaises(TypeError):
                client._path_kind(Path('/not/osf'))
            with self.assertRaises(TypeError):
                client._list_dir(Path('/not/osf'))

    # Path extraction ----------------------------------------------------------
    def test_extract_path_errors(self):
        """`_extract_path` raises the right error for the right mistake."""
        contents = {
            'kind': 'directory',
            'contents': {'a.txt': {'kind': 'file'}}}
        with self.assertRaises(FileNotFoundError):
            OSFClient._extract_path(contents, f'osf://{PROJECT}/nope')
        with self.assertRaises(NotADirectoryError):
            OSFClient._extract_path(contents, f'osf://{PROJECT}/a.txt/x')


def _osf_treecache_name():
    "The page-cache filename stem used by immlib.pathlib._osf."
    from ...pathlib._osf import osf_pagecache_filename
    return osf_pagecache_filename
