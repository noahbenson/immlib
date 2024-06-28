# -*- coding: utf-8 -*-
################################################################################
# immlib/test/iolib/test_core.py

"""Tests for the core iolib submodule in immlib: i.e., tests for the code in
the `immlib.iolib._core` module.
"""


# Dependencies #################################################################

import gzip
from io       import StringIO, BytesIO
from unittest import TestCase
from tempfile import TemporaryDirectory

from ...pathlib import *
from ...iolib   import *
from ...util    import is_str


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
            # Also test the ability to load from CloudPaths.
            p = osfpath("osf://bw9ec", local_cache_dir=tmpdir)
            lns = load(p / "analysis.m", "text")
            self.assertEqual(len(lns), 94)
            self.assertTrue(all(is_str(ln) for ln in lns))
            # Gzip objects should also work fine.
            p_gz = path(tmpdir) / "test.json.gz"
            save(p_gz, json_example)
            d = load(p_gz)
            self.assertEqual(json_example, d)
            # We should be able to use the gzip module to load this also.
            with gzip.open(p_gz, 'rt') as fl:
                d = load(fl, "json")
            self.assertEqual(json_example, d)
