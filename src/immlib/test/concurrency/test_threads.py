# -*- coding: utf-8 -*-
################################################################################
# immlib/test/concurrency/test_threads.py
#
# Tests of immlib's behavior when used from several threads at once. These
# tests are meaningful with or without the GIL, but they are most useful on a
# free-threaded build of Python.


# Dependencies #################################################################

import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest import TestCase


NTHREADS = 8

def run_threads(fn, nthreads=NTHREADS):
    """Runs ``fn(i)`` for ``i in range(nthreads)`` in separate threads that
    start at the same time, and returns the results (re-raising the first
    exception, if any)."""
    barrier = threading.Barrier(nthreads)
    def task(i):
        barrier.wait()
        return fn(i)
    with ThreadPoolExecutor(nthreads) as ex:
        futs = [ex.submit(task, i) for i in range(nthreads)]
        return [f.result() for f in futs]

class TestThreads(TestCase):
    """Tests of immlib when used from several threads at once."""
    def test_default_ureg_threads(self):
        import immlib
        from immlib import UnitRegistry, default_ureg, quant
        global_ureg = immlib.units
        uregs = [UnitRegistry() for _ in range(NTHREADS)]
        def use_ureg(i):
            for _ in range(20):
                with default_ureg(uregs[i]):
                    time.sleep(0)
                    self.assertIs(immlib.units, uregs[i])
                    self.assertIs(quant(1.0, 'mm')._REGISTRY, uregs[i])
                    with default_ureg(global_ureg):
                        self.assertIs(immlib.units, global_ureg)
                    self.assertIs(immlib.units, uregs[i])
            return immlib.units
        results = run_threads(use_ureg)
        # Outside of the blocks, each thread sees the global registry.
        for r in results:
            self.assertIs(r, global_ureg)
        self.assertIs(immlib.units, global_ureg)
        # Assigning immlib.units changes the global default.
        other = UnitRegistry()
        try:
            immlib.units = other
            self.assertIs(immlib.units, other)
            self.assertIs(run_threads(lambda i: immlib.units)[0], other)
        finally:
            immlib.units = global_ureg
        self.assertIs(immlib.units, global_ureg)
    def test_url_download_atomic(self):
        from pathlib import Path
        from immlib.util import url_download
        from immlib.pathlib._osf import _osf_cache_file
        data = os.urandom(4_000_000)
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / 'source.bin'
            src.write_bytes(data)
            url = src.as_uri()
            self.assertEqual(url_download(url), data)
            for rep in range(3):
                dest = Path(tmpdir) / f'cache{rep}' / 'sub' / 'file.bin'
                def fetch(i):
                    p = _osf_cache_file(url, dest)
                    return len(Path(p).read_bytes())
                self.assertEqual(run_threads(fetch), [len(data)] * NTHREADS)
                # No temporary files are left behind.
                self.assertEqual(os.listdir(dest.parent), ['file.bin'])
            # A failed download leaves nothing at the destination.
            dest = Path(tmpdir) / 'missing.bin'
            with self.assertRaises(Exception):
                url_download((Path(tmpdir) / 'nosuchfile').as_uri(), dest)
            self.assertFalse(dest.exists())
            self.assertEqual(
                [f for f in os.listdir(tmpdir) if f.endswith('.part')], [])
    def test_plandict_threads(self):
        from immlib.workflow import calc, plan
        calls = []
        lock = threading.Lock()
        @calc('y')
        def slow_add_one(x):
            with lock:
                calls.append(x)
            time.sleep(0.01)
            return x + 1
        @calc('z')
        def double(y):
            return 2 * y
        p = plan(add=slow_add_one, double=double)
        for rep in range(5):
            pd = p(x=rep)
            results = run_threads(lambda i: (pd['z'], pd['y']))
            self.assertEqual(results, [(2*(rep + 1), rep + 1)] * NTHREADS)
        # Each plandict ran its calculation exactly once.
        self.assertEqual(calls, list(range(5)))
    def test_lambdadict_threads(self):
        from immlib.util import lambdadict
        calls = []
        lock = threading.Lock()
        def make(k):
            def f(a):
                with lock:
                    calls.append(k)
                time.sleep(0.001)
                return a + k
            return f
        d = lambdadict(a=1, **{f'b{k}': make(k) for k in range(10)})
        results = run_threads(lambda i: [d[f'b{k}'] for k in range(10)])
        self.assertEqual(results, [[1 + k for k in range(10)]] * NTHREADS)
        self.assertEqual(sorted(calls), list(range(10)))
    def test_quantity_threads(self):
        import numpy as np
        from immlib import quant
        import immlib.math as im
        a = quant(np.arange(100.0), 'm')
        b = quant(np.arange(100.0), 'cm')
        n = quant(np.arange(100.0))
        def compute(i):
            for _ in range(50):
                r = a + b
                self.assertEqual(r.units, a.units)
                self.assertTrue(np.allclose(r.m, 1.01 * np.arange(100.0)))
                self.assertTrue(np.array_equal(a == b, np.arange(100) == 0))
                self.assertIsNone((n * n).units)
                self.assertEqual(im.sum(a).units, a.units)
                quant(float(i), 'mm/s').to('m/s')
            return True
        self.assertTrue(all(run_threads(compute)))
