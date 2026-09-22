# -*- coding: utf-8 -*-
################################################################################
# immlib/test/iolib/test_core.py

"""Tests for the core iolib submodule in immlib: i.e., tests for the code in
the `immlib.iolib._core` module.
"""


# Dependencies #################################################################

import gzip
from io       import StringIO, BytesIO
from unittest import TestCase, skipIf
from tempfile import TemporaryDirectory

import numpy as np

from ...pathlib import *
from ...iolib   import *
from ...util    import is_str
from ..pathlib._osf_fixture import osf_mock, PROJECT


# Whether pandas, which the csv and tsv formats need, is importable. It is not
# a dependency of immlib, so the formats that use it are skipped without it.
try:
    import pandas as _pandas
    have_pandas = True
except ImportError:
    have_pandas = False


# Tests ########################################################################

text_example = """
   Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis molestie, nisl
   eget volutpat interdum, ex tortor suscipit ipsum, at aliquam justo neque in
   enim. In eu sollicitudin nunc. Integer egestas consequat nibh, ut pretium ex
   accumsan ornare. Fusce et nisl laoreet, aliquet felis ac, fermentum
   ipsum. Pellentesque at velit condimentum, auctor enim quis, ultrices
   libero. Praesent efficitur posuere orci, id pretium nunc iaculis
   id. Vestibulum varius massa in magna iaculis scelerisque. Etiam fermentum
   condimentum orci, eget volutpat elit faucibus vestibulum. Mauris nec
   imperdiet urna.
"""
json_example = {
    'a': [1, 'test_2', {'b': 3, 'c': 4}, 5],
    'x': {'y': [10, 20.01, 30], 'z': [40, 50.01, 60]},
    'q': 100.5,
    'n': None,
    'bools': [True, False]
}

class TestIOLibCore(TestCase):
    """Tests for the core module of the immlib.iolib subpackage.
    """

    def test_save(self):
        """Tests the immlib.save interface."""
        # We'll start by saving to StringIO and BytesIO objects.
        s = save(StringIO(), text_example, "str")
        self.assertEqual(s.getvalue(), text_example)
        s = save(StringIO(), text_example, "text")
        self.assertEqual(s.getvalue(), text_example)
        # For JSON and YAML, easiest to check for equality after deserializing.
        s = save(StringIO(), json_example, "json")
        d = load(StringIO(s.getvalue()), "json")
        self.assertEqual(json_example, d)
        s = save(StringIO(), json_example, "yaml")
        d = load(StringIO(s.getvalue()), "yaml")
        self.assertEqual(json_example, d)
        # We should also be able to save to a path.
        with TemporaryDirectory() as tmpdir:
            p = path(tmpdir) / "test.json"
            # Should auto-detect the format from extension.
            save(p, json_example)
            d = load(p)
            self.assertEqual(json_example, d)
            # Gzip objects should also work fine.
            p_gz = path(tmpdir) / "test.json.gz"
            save(p_gz, json_example)
            d = load(p_gz)
            self.assertEqual(json_example, d)
            # We should be able to use the gzip module to load this also.
            with gzip.open(p_gz, 'rt') as fl:
                d = load(fl, "json")
            self.assertEqual(json_example, d)
    def test_save_cloudpath(self):
        """Tests that load works with an OSF CloudPath.

        The OSF project is served from a fixture, so this does not require
        network access.
        """
        with osf_mock(), TemporaryDirectory() as tmpdir:
            p = osfpath(f"osf://{PROJECT}", local_cache_dir=tmpdir)
            lns = load(p / "a.txt", "text")
            self.assertEqual(lns, ["hello world"])
            self.assertTrue(all(is_str(ln) for ln in lns))

    # Format ###################################################################
    def test_format_suffixes(self):
        """Tests how the Format constructor interprets its suffix arguments."""
        from ...iolib._core import Format
        def fn(stream, obj):
            "A docstring."
            stream.write(str(obj))
        # A string suffix becomes a 1-tuple; a sequence of strings becomes a
        # tuple, which is how a compound suffix like ('.tar', '.gz') is given.
        f = Format('test', fn, '.a')
        self.assertEqual(f.suffixes, [('.a',)])
        f = Format('test', fn, ('.a', '.b'))
        self.assertEqual(f.suffixes, [('.a', '.b')])
        f = Format('test', fn, '.a', ['.b', '.c'])
        self.assertEqual(f.suffixes, [('.a',), ('.b', '.c')])
        # Anything else is an error, whether it is a sequence of non-strings
        # or not a sequence at all.
        with self.assertRaises(ValueError):
            Format('test', fn, ['.a', 10])
        with self.assertRaises(ValueError):
            Format('test', fn, 10)
        # The format takes its documentation from its function.
        self.assertEqual(f.__doc__, fn.__doc__)
        self.assertIs(f.function, fn)
        self.assertEqual(f.name, 'test')
    def test_format_mode(self):
        """Tests the Format constructor's mode argument."""
        from ...iolib._core import Format
        def fn(stream, obj):
            pass
        self.assertEqual(Format('test', fn).mode, 'b')
        self.assertEqual(Format('test', fn, mode='t').mode, 't')
        with self.assertRaises(ValueError):
            Format('test', fn, mode='x')
    def test_format_gzip_suffix(self):
        """Tests the Format constructor's gzip_suffix argument."""
        from ...iolib._core import Format
        def fn(stream, obj):
            pass
        # None means no gzip suffixes at all.
        self.assertEqual(Format('test', fn).gzip_suffix, ())
        # A single string is one suffix.
        self.assertEqual(
            Format('test', fn, gzip_suffix='.npz').gzip_suffix, (('.npz',),))
        # A sequence of strings is a sequence of one-element suffixes.
        self.assertEqual(
            Format('test', fn, gzip_suffix=['.npz', '.nz']).gzip_suffix,
            (('.npz',), ('.nz',)))
        # A sequence that mixes strings and sequences is read elementwise; this
        # is how a compound gzip suffix is given.
        self.assertEqual(
            Format('test', fn, gzip_suffix=['.npz', ('.json', '.gz')]
                   ).gzip_suffix,
            (('.npz',), ('.json', '.gz')))
        # Anything else is an error.
        with self.assertRaises(ValueError):
            Format('test', fn, gzip_suffix=10)
        with self.assertRaises(ValueError):
            Format('test', fn, gzip_suffix=[10])
    def test_format_call(self):
        """Tests calling a Format object directly, with a stream and a path."""
        from ...iolib._core import Format
        def fn(stream, obj, prefix=''):
            stream.write(prefix + str(obj))
            return 'returned'
        f = Format('test', fn, mode='t')
        # With a stream, the stream is used as it is.
        s = StringIO()
        self.assertEqual(f(s, 'w', 'abc', prefix='> '), 'returned')
        self.assertEqual(s.getvalue(), '> abc')
        # With anything else, the argument is opened as a path in the format's
        # own mode, combined with the stream mode given here.
        with TemporaryDirectory() as tmpdir:
            p = path(tmpdir) / 'test_format_call.txt'
            self.assertEqual(f(p, 'w', 'abc'), 'returned')
            self.assertEqual(p.read_text(), 'abc')

    # Formatter ################################################################
    def test_formatter_template(self):
        """Tests building a Save or Load object from a template."""
        from ...iolib._core import Save, Load, Format
        # A Save object can be copied from another Save object...
        s = Save(save)
        self.assertEqual(set(s.formats), set(save.formats))
        for (k, fmt) in save.formats.items():
            self.assertIs(s.formats[k], fmt)
        # ...or from a mapping of names to Format objects.
        s = Save({'json': save.formats['json'], 'text': save.formats['text']})
        self.assertEqual(set(s.formats), {'json', 'text'})
        self.assertIs(s.formats['json'], save.formats['json'])
        # The same works for Load.
        ld = Load(load)
        self.assertEqual(set(ld.formats), set(load.formats))
        # A template must be a Formatter of the same type or a mapping.
        with self.assertRaises(TypeError):
            Save(10)
        # A mapping whose values are not Formats is an error.
        with self.assertRaises(TypeError):
            Save({'nope': 10})
        # A Load is not a valid template for a Save.
        with self.assertRaises(TypeError):
            Save(load)
        # copy() is the same thing.
        s = save.copy()
        self.assertIsInstance(s, Save)
        self.assertEqual(set(s.formats), set(save.formats))
        self.assertIsNot(s.formats, save.formats)
    def test_expandall(self):
        """Tests Formatter._expandall, which normalizes a local path."""
        import os
        from pathlib import Path
        from ...iolib._core import Formatter
        # A local path has ~ and environment variables expanded.
        p = Formatter._expandall('~/some_file.json')
        self.assertIsInstance(p, Path)
        self.assertNotIn('~', str(p))
        self.assertTrue(str(p).startswith(os.path.expanduser('~')))
        os.environ['IMMLIB_TEST_EXPANDALL'] = 'expanded_dir'
        try:
            p = Formatter._expandall('$IMMLIB_TEST_EXPANDALL/f.json')
            self.assertEqual(p.parts[0], 'expanded_dir')
        finally:
            del os.environ['IMMLIB_TEST_EXPANDALL']
        # A cloud path is returned as it is: it has no home directory or
        # environment variables to expand, and converting it to a Path would
        # lose the fact that it is remote.
        with TemporaryDirectory() as tmpdir:
            cp = osfpath("osf://bw9ec", local_cache_dir=tmpdir)
            self.assertIs(Formatter._expandall(cp), cp)
    def test_deduce_format(self):
        """Tests Formatter.deduce_format."""
        # A string, a path, or a sequence of suffixes are all accepted.
        self.assertEqual(save.deduce_format('a.json').name, 'json')
        self.assertEqual(save.deduce_format(path('a.txt')).name, 'text')
        self.assertEqual(save.deduce_format(['.json']).name, 'json')
        self.assertEqual(save.deduce_format(('.txt',)).name, 'text')
        # A trailing .gz is ignored by default but not when ignore_gz is False.
        self.assertEqual(save.deduce_format('a.json.gz').name, 'json')
        self.assertIsNone(save.deduce_format('a.json.gz', ignore_gz=False))
        # Suffixes are dropped from the left until one matches, so a name with
        # extra dots in it still works.
        self.assertEqual(save.deduce_format('a.b.c.json').name, 'json')
        # An unknown suffix, or no suffix at all, deduces nothing.
        self.assertIsNone(save.deduce_format('a.unknown_suffix'))
        self.assertIsNone(save.deduce_format('a'))
        # Anything that is not a path, string, or sequence of strings is an
        # error.
        with self.assertRaises(ValueError):
            save.deduce_format(10)
        with self.assertRaises(ValueError):
            save.deduce_format([10])
    def test_register(self):
        """Tests Formatter.register, in both its decorator and direct forms."""
        from ...iolib._core import Format
        # Work on a copy so that the global save object is left alone.
        s = save.copy()
        # The decorator form builds the Format and returns it.
        @s.register('test_register', '.trg', mode='t')
        def save_trg(stream, obj):
            "A test format."
            stream.write('!' + str(obj))
        self.assertIsInstance(save_trg, Format)
        self.assertIs(s.formats['test_register'], save_trg)
        self.assertIs(s.deduce_format('x.trg'), save_trg)
        self.assertEqual(save_trg.mode, 't')
        # And the format works.
        st = s(StringIO(), 'abc', 'test_register')
        self.assertEqual(st.getvalue(), '!abc')
        st = s(StringIO(), 'abc', save_trg)
        self.assertEqual(st.getvalue(), '!abc')
        # The direct form registers an existing Format object, in another
        # Formatter for instance.
        s2 = save.copy()
        self.assertIs(s2.register(save_trg), save_trg)
        self.assertIs(s2.formats['test_register'], save_trg)
        # A duplicate name is an error, and so is a duplicate suffix.
        with self.assertRaises(RuntimeError):
            s.register(save_trg)
        dup = Format('test_register_2', save_trg.function, '.trg', mode='t')
        with self.assertRaises(RuntimeError):
            s.register(dup)
        # Decorating a Format re-wraps its function rather than nesting it.
        s3 = save.copy()
        f = s3.register('test_register_3', '.trg3', mode='t')(save_trg)
        self.assertIsInstance(f, Format)
        self.assertIs(f.function, save_trg.function)
        # A gzip suffix is registered along with the ordinary ones.
        s4 = save.copy()
        @s4.register('test_register_4', '.tr4', mode='b', gzip_suffix='.tr4z')
        def save_tr4(stream, obj):
            stream.write(bytes(obj))
        self.assertIs(s4.deduce_format('x.tr4'), save_tr4)
        self.assertIs(s4.deduce_format('x.tr4z'), save_tr4)
    def test_unregister(self):
        """Tests Formatter.unregister."""
        s = save.copy()
        fmt = s.formats['json']
        self.assertIs(s.unregister('json'), fmt)
        self.assertNotIn('json', s.formats)
        # The suffixes go with it.
        self.assertIsNone(s.deduce_format('x.json'))
        # Unregistering something absent returns None by default...
        self.assertIsNone(s.unregister('json'))
        self.assertIsNone(s.unregister('never_registered'))
        # ...and raises when asked to.
        with self.assertRaises(RuntimeError):
            s.unregister('json', error_on_missing=True)
        # The global save object is unaffected by any of this.
        self.assertIn('json', save.formats)
        # A format with a gzip suffix loses that suffix too.
        s2 = save.copy()
        s2.unregister('numpy')
        self.assertIsNone(s2.deduce_format('x.npy'))
        self.assertIsNone(s2.deduce_format('x.npz'))
    def test_call_argument_errors(self):
        """Tests the errors raised for bad target/format combinations."""
        # A stream with no format cannot be deduced.
        with self.assertRaises(ValueError):
            save(StringIO(), 'abc')
        with self.assertRaises(ValueError):
            load(StringIO())
        # Neither can a path whose suffix means nothing.
        with self.assertRaises(ValueError):
            save('test.unknown_suffix', 'abc')
        # An unrecognized format name is an error.
        with self.assertRaises(ValueError):
            save(StringIO(), 'abc', 'no_such_format')
        # So is a format argument that is neither a name nor a Format.
        with self.assertRaises(TypeError):
            save(StringIO(), 'abc', 10)

    # The individual save/load formats #########################################
    def test_str_bytes_repr(self):
        """Tests the str, bytes, and repr formats in both directions."""
        # str.
        s = save(StringIO(), 10.5, 'str')
        self.assertEqual(s.getvalue(), '10.5')
        self.assertEqual(load(StringIO('10.5'), 'str'), '10.5')
        # The append_nl and strip_nl options are mirror images.
        s = save(StringIO(), 10.5, 'str', append_nl=True)
        self.assertEqual(s.getvalue(), '10.5\n')
        self.assertEqual(load(StringIO('10.5\n'), 'str', strip_nl=True), '10.5')
        self.assertEqual(load(StringIO('10.5\n'), 'str'), '10.5\n')
        # A size can be given to the loader.
        self.assertEqual(load(StringIO('abcdef'), 'str', size=3), 'abc')
        # bytes.
        s = save(BytesIO(), b'abc', 'bytes')
        self.assertEqual(s.getvalue(), b'abc')
        self.assertEqual(load(BytesIO(b'abc'), 'bytes'), b'abc')
        self.assertEqual(load(BytesIO(b'abcdef'), 'bytes', size=3), b'abc')
        # repr, which round-trips through ast.literal_eval.
        obj = {'a': [1, 2.5, 'three'], 'b': (4, None, True)}
        s = save(StringIO(), obj, 'repr')
        self.assertEqual(load(StringIO(s.getvalue()), 'repr'), obj)
        s = save(StringIO(), obj, 'repr', append_nl=True)
        self.assertTrue(s.getvalue().endswith('\n'))
        self.assertEqual(load(StringIO(s.getvalue()), 'repr'), obj)
    def test_text(self):
        """Tests the text format in both directions."""
        # A single string is written as it is.
        s = save(StringIO(), 'one line', 'text')
        self.assertEqual(s.getvalue(), 'one line')
        # So is a sequence of strings.
        s = save(StringIO(), ['a', 'b', 'c'], 'text')
        self.assertEqual(s.getvalue(), 'abc')
        # With append_nls, each line gets a newline, which is what makes the
        # result load back as the same sequence.
        s = save(StringIO(), ['a', 'b', 'c'], 'text', append_nls=True)
        self.assertEqual(s.getvalue(), 'a\nb\nc\n')
        self.assertEqual(load(StringIO(s.getvalue()), 'text'), ['a', 'b', 'c'])
        # Without strip_nls the newlines are kept.
        self.assertEqual(
            load(StringIO('a\nb\n'), 'text', strip_nls=False), ['a\n', 'b\n'])
        # Anything that is not text or a sequence of text is an error.
        with self.assertRaises(TypeError):
            save(StringIO(), 10, 'text')
    def test_json_default(self):
        """Tests the json_default function, which save_json uses."""
        from ...iolib._core import json_default
        from pcollections import pdict, plist
        # The types that JSON already understands pass through.
        for obj in ('abc', None, True, False):
            self.assertIs(json_default(obj), obj)
        # NumPy scalars become Python numbers of the right kind.
        self.assertIsInstance(json_default(np.int64(3)), int)
        self.assertEqual(json_default(np.int64(3)), 3)
        self.assertIsInstance(json_default(np.float64(2.5)), float)
        self.assertEqual(json_default(np.float64(2.5)), 2.5)
        # Persistent collections become their mutable equivalents.
        self.assertEqual(json_default(pdict(a=1)), {'a': 1})
        self.assertIsInstance(json_default(pdict(a=1)), dict)
        self.assertEqual(json_default(plist([1, 2])), [1, 2])
        self.assertIsInstance(json_default(plist([1, 2])), list)
        # Arrays become nested lists, at any rank.
        self.assertEqual(json_default(np.arange(3)), [0, 1, 2])
        self.assertEqual(
            json_default(np.arange(4).reshape(2, 2)), [[0, 1], [2, 3]])
        # Anything else is a TypeError, which is what json.dump expects of a
        # default function that cannot help.
        with self.assertRaises(TypeError):
            json_default(object())
        # And the format as a whole handles all of the above.
        obj = {'i': np.int64(1), 'f': np.float64(2.5), 'a': np.arange(3),
               'd': pdict(x=1), 'l': plist([1, 2])}
        s = save(StringIO(), obj, 'json')
        self.assertEqual(
            load(StringIO(s.getvalue()), 'json'),
            {'i': 1, 'f': 2.5, 'a': [0, 1, 2], 'd': {'x': 1}, 'l': [1, 2]})
        with self.assertRaises(TypeError):
            save(StringIO(), {'bad': object()}, 'json')
    def test_yaml_prepare(self):
        """Tests the yaml_prepare function, which save_yaml uses."""
        from ...iolib._core import yaml_prepare
        for obj in ('abc', None, True, False):
            self.assertIs(yaml_prepare(obj), obj)
        self.assertIsInstance(yaml_prepare(np.int64(3)), int)
        self.assertIsInstance(yaml_prepare(np.float64(2.5)), float)
        self.assertEqual(yaml_prepare(np.arange(3)), [0, 1, 2])
        # Unlike json_default, this one recurses, so a nested structure is
        # converted all the way down.
        self.assertEqual(
            yaml_prepare({'a': [np.int64(1), {'b': np.arange(2)}]}),
            {'a': [1, {'b': [0, 1]}]})
        # Mapping keys must be strings, since YAML's are.
        with self.assertRaises(TypeError):
            yaml_prepare({1: 2})
        with self.assertRaises(TypeError):
            yaml_prepare(object())
        # The format as a whole refuses an object it cannot serialize rather
        # than writing out a Python-specific representation of it.
        with self.assertRaises(TypeError):
            save(StringIO(), {'bad': object()}, 'yaml')
        # yaml can be loaded unsafely on request.
        s = save(StringIO(), {'a': [1, 2]}, 'yaml')
        self.assertEqual(
            load(StringIO(s.getvalue()), 'yaml', safe=False), {'a': [1, 2]})
    def test_pickle(self):
        """Tests the pickle format in both directions."""
        obj = {'a': [1, 2, 3], 'b': 'text'}
        s = save(BytesIO(), obj, 'pickle')
        self.assertEqual(load(BytesIO(s.getvalue()), 'pickle'), obj)
        # The protocol option is forwarded to pickle.dump.
        s = save(BytesIO(), obj, 'pickle', protocol=2)
        self.assertEqual(load(BytesIO(s.getvalue()), 'pickle'), obj)
        with TemporaryDirectory() as tmpdir:
            for suffix in ('.pickle', '.pkl', '.pcl'):
                p = path(tmpdir) / ('test' + suffix)
                save(p, obj)
                self.assertEqual(load(p), obj)
    def test_numpy(self):
        """Tests the numpy format, including its .npz gzip suffix."""
        arr = np.arange(12).reshape(3, 4)
        s = save(BytesIO(), arr, 'numpy')
        self.assertTrue(np.array_equal(load(BytesIO(s.getvalue()), 'numpy'),
                                       arr))
        with TemporaryDirectory() as tmpdir:
            for suffix in ('.npy', '.np', '.numpy'):
                p = path(tmpdir) / ('test' + suffix)
                save(p, arr)
                self.assertTrue(np.array_equal(load(p), arr))
            # A .npz file is the same format, gzipped; the suffix is enough to
            # say so, without a .gz on the end.
            p = path(tmpdir) / 'test.npz'
            save(p, arr)
            self.assertTrue(np.array_equal(load(p), arr))
            # And it really is gzipped.
            with gzip.open(p, 'rb') as fl:
                self.assertTrue(np.array_equal(np.load(fl), arr))
    @skipIf(not have_pandas, "pandas is not installed")
    def test_csv_tsv(self):
        """Tests the csv and tsv formats in both directions."""
        import pandas
        df = pandas.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
        with TemporaryDirectory() as tmpdir:
            p = path(tmpdir) / 'test.csv'
            save(p, df)
            self.assertTrue(load(p).equals(df))
            p = path(tmpdir) / 'test.tsv'
            save(p, df)
            self.assertTrue(load(p).equals(df))
            # The index is not written by default, but can be.
            p = path(tmpdir) / 'test_index.csv'
            save(p, df, index=True)
            self.assertIn('Unnamed: 0', load(p).columns)
    def test_gzip(self):
        """Tests the gzip handling of save and load."""
        with TemporaryDirectory() as tmpdir:
            # A .gz suffix is detected from the path.
            p = path(tmpdir) / 'test.json.gz'
            save(p, json_example)
            with gzip.open(p, 'rt') as fl:
                self.assertEqual(load(fl, 'json'), json_example)
            self.assertEqual(load(p), json_example)
            # gzip=False writes the file uncompressed despite the suffix.
            p = path(tmpdir) / 'plain.json.gz'
            save(p, json_example, gzip=False)
            self.assertEqual(p.read_text()[0], '{')
            self.assertEqual(load(p, gzip=False), json_example)
            # gzip=True compresses a path whose suffix does not say to.
            p = path(tmpdir) / 'compressed.json'
            save(p, json_example, gzip=True)
            with gzip.open(p, 'rt') as fl:
                self.assertEqual(load(fl, 'json'), json_example)
            self.assertEqual(load(p, 'json', gzip=True), json_example)
        # gzip=True also works when the target is a stream rather than a path.
        s = save(BytesIO(), json_example, 'json', gzip=True)
        self.assertEqual(
            load(BytesIO(s.getvalue()), 'json', gzip=True), json_example)
    def test_load_from_dir(self):
        """Tests loading a directory as a lazy nested dictionary."""
        from pcollections import ldict
        from ...iolib._core import Load
        with TemporaryDirectory() as tmpdir:
            root = path(tmpdir)
            (root / 'a.json').write_text('{"x": 1}')
            (root / 'sub').mkdir()
            (root / 'sub' / 'b.txt').write_text('hello')
            (root / 'sub' / 'deeper').mkdir()
            (root / 'sub' / 'deeper' / 'c.txt').write_text('deep')
            # Loading a directory with no format gives a lazy dict.
            d = load(root)
            self.assertIsInstance(d, ldict)
            self.assertEqual(set(d), {'a.json', 'sub'})
            # A file entry is the path itself; loading it is left to the
            # caller, who then knows the format.
            self.assertEqual(load(d['a.json']), {'x': 1})
            # A directory entry is another lazy dict, one level at a time.
            self.assertIsInstance(d['sub'], ldict)
            self.assertEqual(set(d['sub']), {'b.txt', 'deeper'})
            self.assertEqual(load(d['sub']['b.txt']), ['hello'])
            self.assertEqual(set(d['sub']['deeper']), {'c.txt'})
            # The format can also be named explicitly.
            self.assertEqual(set(load(root, 'dir')), {'a.json', 'sub'})
            # A filter selects which entries appear, at every level.
            d = Load.from_dir(root, filter=lambda p: p.name != 'deeper')
            self.assertEqual(set(d['sub']), {'b.txt'})
            # A file is not a directory.
            with self.assertRaises(NotADirectoryError):
                Load.from_dir(root / 'a.json')
