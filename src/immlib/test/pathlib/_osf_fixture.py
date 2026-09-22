# -*- coding: utf-8 -*-
################################################################################
# immlib/test/pathlib/_osf_fixture.py
#
# A canned OSF project and a context manager that serves it, so that the OSF
# code paths can be tested without a network connection.
#
# The OSF client obtains everything it knows through `url_download`, either as
# the JSON of an API page (when no destination path is given) or as the bytes
# of a file (when one is). `osf_mock` replaces `immlib.pathlib._osf.url_download`
# with a function that answers both forms from a fixture, which is what makes
# the tests deterministic (the live OSF API returns 503s often enough that a
# network-dependent test is not a reliable one).


# Dependencies #################################################################

import json
from contextlib import contextmanager
from pathlib import Path

import immlib.pathlib._osf as _osf


# Fixture ######################################################################

PROJECT = 'projx'
STORAGE = 'osfstorage'
BASE = f'https://api.osf.io/v2/nodes/{PROJECT}/files/{STORAGE}/'
ROOT_URL = BASE + '?page[size]=100'
SUB_URL = BASE + 'sub?page[size]=100'


def file_entry(name, download, size,
               modified='2020-01-02T03:04:05.000000Z',
               created='2020-01-01T00:00:00.000000Z'):
    """Builds the OSF JSON for one file entry."""
    return {
        'attributes': {
            'name': name, 'kind': 'file', 'size': size,
            'date_modified': modified, 'date_created': created},
        'links': {'download': download}}


def dir_entry(name, path=None):
    """Builds the OSF JSON for one directory entry."""
    return {
        'attributes': {
            'name': name, 'kind': 'directory',
            'path': path if path is not None else '/' + name},
        'links': {}}


def page(entries, next_url=None):
    """Builds one page of the OSF contents API."""
    links = {} if next_url is None else {'next': next_url}
    return {'data': list(entries), 'links': links}


#: One file (`a.txt`) and one directory (`sub/`, containing `b.txt`).
DEFAULT_PAGES = {
    ROOT_URL: page([file_entry('a.txt', 'osfdl://a.txt', 11),
                    dir_entry('sub')]),
    SUB_URL: page([file_entry('b.txt', 'osfdl://b.txt', 5)]),
}
#: The bytes each download URL yields; the sizes above match these.
DEFAULT_DOWNLOADS = {'osfdl://a.txt': b'hello world', 'osfdl://b.txt': b'12345'}


@contextmanager
def osf_mock(pages=None, downloads=None):
    """Serves a canned OSF project in place of the network.

    `pages` maps an API URL to the JSON object the API returns for it, and
    `downloads` maps a download URL to the bytes it yields. Each defaults to
    the fixture above. The context manager yields a list to which every
    requested URL is appended, so that a test can tell whether (and when) the
    network was consulted.
    """
    pages = DEFAULT_PAGES if pages is None else pages
    downloads = DEFAULT_DOWNLOADS if downloads is None else downloads
    real = _osf.url_download
    calls = []
    def fake(url, destpath=None, **kwargs):
        calls.append(url)
        if destpath is None:
            if url not in pages:
                raise AssertionError(f'unmocked OSF page URL: {url}')
            return json.dumps(pages[url]).encode('utf-8')
        destpath = Path(destpath)
        destpath.parent.mkdir(parents=True, exist_ok=True)
        destpath.write_bytes(downloads.get(url, b''))
        return destpath
    _osf.url_download = fake
    try:
        yield calls
    finally:
        _osf.url_download = real
