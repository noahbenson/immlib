# -*- coding: utf-8 -*-
################################################################################
# immlib/test/pathlib/test_cache.py

"""Tests for the cache submodule of immlib.pathlib: i.e., tests for the code in
the `immlib.pathlib._cache` module.

These tests use `cloudpathlib`'s own local test backend (`LocalS3Client`),
which implements the `CloudPath` interface against a directory on the
filesystem. That keeps the tests offline while still exercising
`CloudCachePath` against a real `CloudPath` object with a real cache
directory, which is the only thing it wraps.
"""


# Dependencies #################################################################

import os
from pathlib  import Path
from unittest import TestCase
from tempfile import TemporaryDirectory

from cloudpathlib.local import LocalS3Client


# Tests ########################################################################

class TestPathlibCache(TestCase):
    """Tests for the CloudCachePath type of the immlib.pathlib subpackage."""

    def _client(self, stack):
        """Builds a local S3 "bucket" with a small tree in it and returns
        `(client, storage_dir, cache_dir)`.

        The two temporary directories are cleaned up by `stack`, which must
        be a `contextlib.ExitStack`.
        """
        storage = stack.enter_context(TemporaryDirectory())
        cache = stack.enter_context(TemporaryDirectory())
        bucket = os.path.join(storage, 'bucket')
        os.makedirs(os.path.join(bucket, 'sub'), exist_ok=True)
        with open(os.path.join(bucket, 'a.txt'), 'w') as fl:
            fl.write('hello')
        with open(os.path.join(bucket, 'sub', 'b.txt'), 'w') as fl:
            fl.write('deep')
        client = LocalS3Client(local_storage_dir=storage, local_cache_dir=cache)
        return (client, storage, cache)

    def _delegates(self, p, name, *args):
        """Asserts that ``p.<name>(*args)`` does the same thing as
        ``Path(p.cloud_path.fspath).<name>(*args)``.

        Several of CloudCachePath's methods exist only to hand the call to
        the cached file, and several of those are not supported on every
        platform: `group` and `owner` are implemented with the pwd and grp
        modules, so a Path raises for them on Windows, and `lchmod` raises
        wherever chmod cannot follow symlinks. Asserting a return value
        would make this test a test of the platform. Asserting that the two
        agree--including that they raise the same kind of error--is what
        the delegation actually promises.
        """
        ref = Path(p.cloud_path.fspath)
        try:
            want = getattr(ref, name)(*args)
        except Exception as e:
            with self.assertRaises(type(e), msg=name):
                getattr(p, name)(*args)
        else:
            self.assertEqual(getattr(p, name)(*args), want, name)

    def test_construction(self):
        """Tests the CloudCachePath constructor."""
        from contextlib import ExitStack
        from ...pathlib._cache import CloudCachePath
        with ExitStack() as stack:
            (client, storage, cache) = self._client(stack)
            cp = client.CloudPath('s3://bucket/a.txt')
            p = CloudCachePath(cp)
            # The result is a Path as well as a wrapper, which is the point
            # of the type: a CloudPath is not a Path.
            self.assertIsInstance(p, Path)
            self.assertIsInstance(p, CloudCachePath)
            self.assertIs(p.cloud_path, cp)
            # The filesystem path it represents is the cloud path's place in
            # the cache directory, made absolute.
            self.assertEqual(
                Path(p), Path(cache).absolute() / 'bucket' / 'a.txt')
            # It must be given a CloudPath; nothing else will do.
            with self.assertRaises(TypeError):
                CloudCachePath('s3://bucket/a.txt')
            with self.assertRaises(TypeError):
                CloudCachePath(Path('a.txt'))
    def test_path_arithmetic(self):
        """Tests the parts of CloudCachePath that build other paths."""
        from contextlib import ExitStack
        from ...pathlib._cache import CloudCachePath
        with ExitStack() as stack:
            (client, storage, cache) = self._client(stack)
            root = CloudCachePath(client.CloudPath('s3://bucket'))
            # Division follows the cloud path, and gives another
            # CloudCachePath rather than a plain Path.
            p = root / 'a.txt'
            self.assertIsInstance(p, CloudCachePath)
            self.assertEqual(str(p.cloud_path), 's3://bucket/a.txt')
            p = root / 'sub' / 'b.txt'
            self.assertEqual(str(p.cloud_path), 's3://bucket/sub/b.txt')
            # parts are the cloud path's parts without its scheme, so they
            # describe the object rather than where it happens to be cached.
            self.assertEqual(root.parts, ('bucket',))
            self.assertEqual(p.parts, ('bucket', 'sub', 'b.txt'))
            # parents likewise walk the cloud path, and each is a
            # CloudCachePath.
            parents = p.parents
            self.assertTrue(
                all(isinstance(q, CloudCachePath) for q in parents))
            self.assertEqual(
                [str(q.cloud_path) for q in parents],
                ['s3://bucket/sub', 's3://bucket'])
            # absolute() returns an equivalent path (the constructor has
            # already made it absolute).
            self.assertEqual(Path(p.absolute()), Path(p))
            self.assertIsInstance(p.absolute(), CloudCachePath)
    def test_listing(self):
        """Tests iterdir, glob, and rglob."""
        from contextlib import ExitStack
        from ...pathlib._cache import CloudCachePath
        with ExitStack() as stack:
            (client, storage, cache) = self._client(stack)
            root = CloudCachePath(client.CloudPath('s3://bucket'))
            entries = list(root.iterdir())
            self.assertTrue(
                all(isinstance(q, CloudCachePath) for q in entries))
            self.assertEqual(
                sorted(str(q.cloud_path) for q in entries),
                ['s3://bucket/a.txt', 's3://bucket/sub'])
            # glob also walks the cloud path.
            self.assertEqual(
                sorted(str(q.cloud_path) for q in root.glob('*.txt')),
                ['s3://bucket/a.txt'])
            # rglob works on the cache directory instead, so it sees what has
            # been downloaded.
            (root / 'a.txt').read_text()
            self.assertEqual(
                [Path(q).name for q in root.rglob('*.txt')], ['a.txt'])
    def test_reading(self):
        """Tests the methods that read the cached file."""
        from contextlib import ExitStack
        from ...pathlib._cache import CloudCachePath
        with ExitStack() as stack:
            (client, storage, cache) = self._client(stack)
            p = CloudCachePath(client.CloudPath('s3://bucket/a.txt'))
            # Reading downloads the object into the cache.
            self.assertFalse(os.path.exists(p))
            self.assertEqual(p.read_text(), 'hello')
            self.assertTrue(os.path.exists(p))
            self.assertEqual(p.read_bytes(), b'hello')
            with p.open() as fl:
                self.assertEqual(fl.read(), 'hello')
            self.assertTrue(p.exists())
            # stat and lstat describe the cached file.
            self.assertEqual(p.stat().st_size, 5)
            self.assertEqual(p.lstat().st_size, 5)
            # As do these. group, owner, and lchmod are asked of the cached
            # file, and what they do is whatever a Path does with it: on
            # POSIX they answer, and on Windows they raise, because CPython
            # implements them with the pwd and grp modules. The test is
            # that the CloudCachePath and the Path agree, which is the
            # whole promise of the delegation.
            self._delegates(p, 'group')
            self._delegates(p, 'owner')
            self._delegates(p, 'lchmod', 0o644)
            self.assertEqual(Path(p.expanduser()), Path(p))
            self.assertEqual(Path(p.resolve()), Path(p).resolve())
            self.assertTrue(p.samefile(p.cloud_path.fspath))
            # readlink raises, as it does for any ordinary file. Which
            # error it is differs by platform, so this too is checked
            # against what a Path does.
            self._delegates(p, 'readlink')
    def test_is_dir_is_file(self):
        """Tests is_dir and is_file, which answer from the cache when the
        object has been downloaded and from the cloud when it has not."""
        from contextlib import ExitStack
        from ...pathlib._cache import CloudCachePath
        with ExitStack() as stack:
            (client, storage, cache) = self._client(stack)
            f = CloudCachePath(client.CloudPath('s3://bucket/a.txt'))
            d = CloudCachePath(client.CloudPath('s3://bucket'))
            # Nothing has been downloaded yet, so these answers come from the
            # cloud path.
            self.assertFalse(os.path.exists(f))
            self.assertTrue(f.is_file())
            self.assertFalse(f.is_dir())
            self.assertTrue(d.is_dir())
            self.assertFalse(d.is_file())
            # After a read, the cache entry exists and the answers come from
            # the filesystem instead. They must agree.
            f.read_text()
            self.assertTrue(os.path.exists(f))
            self.assertTrue(f.is_file())
            self.assertFalse(f.is_dir())
    def test_special_file_predicates(self):
        """Tests the predicates that ask whether this is a special file.

        A cloud object is never a symlink, socket, fifo, device, or mount
        point, so all of these are False. They are checked because code that
        treats a CloudCachePath as the Path it is will call them, and because
        they used to delegate to the CloudPath, which does not define them.
        """
        from contextlib import ExitStack
        from ...pathlib._cache import CloudCachePath
        with ExitStack() as stack:
            (client, storage, cache) = self._client(stack)
            p = CloudCachePath(client.CloudPath('s3://bucket/a.txt'))
            for name in ('is_mount', 'is_symlink', 'is_socket', 'is_fifo',
                         'is_block_device', 'is_char_device'):
                self.assertFalse(getattr(p, name)(), name)
            # And they do not need the object to be downloaded first.
            self.assertFalse(os.path.exists(p))
    def test_read_only(self):
        """Tests that every method that would modify the filesystem raises.

        A CloudCachePath presents a read-only view of a cloud object: writing
        to the cache would not write to the cloud, so it is refused rather
        than silently doing something local and temporary.
        """
        from contextlib import ExitStack
        from ...pathlib._cache import CloudCachePath
        with ExitStack() as stack:
            (client, storage, cache) = self._client(stack)
            p = CloudCachePath(client.CloudPath('s3://bucket/a.txt'))
            calls = {
                'chmod': (0o644,), 'mkdir': (), 'rename': ('x',),
                'replace': ('x',), 'rmdir': (), 'symlink_to': ('x',),
                'hardlink_to': ('x',), 'link_to': ('x',), 'touch': (),
                'unlink': (), 'write_bytes': (b'x',), 'write_text': ('x',)}
            for (name, args) in calls.items():
                with self.assertRaises(TypeError, msg=name):
                    getattr(p, name)(*args)
            # The cached file is untouched by all of that.
            self.assertEqual(p.read_text(), 'hello')
            # lchmod is the one metadata call that is not refused, since it
            # acts on the cache entry only; it is checked in test_reading,
            # against what a Path does with the same call, because it is not
            # supported on every platform.
