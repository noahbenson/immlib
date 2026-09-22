# Stability and Compatibility

`immlib` 1.0 is the library's first stable release. This page describes what
that means for the code that depends on it and which environments it supports.

## Versioning

From 1.0, `immlib` follows [semantic versioning](https://semver.org): within a
major version, names that the library exports from `immlib` and its public
submodules keep their meaning, and a change that would break them waits for the
next major version. Additions (new functions, new keyword options with sensible
defaults) may appear in a minor release. Bug fixes may appear in a patch
release.

## Supported Python versions

`immlib` requires Python 3.10 or later. It is tested on CPython 3.10 through
3.14, on Linux, Windows, and macOS, and additionally on a free-threaded (no-GIL)
build of 3.14.

## Dependencies

`immlib` depends on the following packages. The ranges below are the ones the
release is tested against; see `pyproject.toml` for the authoritative
declaration.

| Package | Supported | Notes |
| ------- | --------- | ----- |
| `pcollections` | `>= 1.0.0rc1` | The persistent and lazy collections that underlie every immutable data structure in `immlib`. |
| `numpy` | `>= 1.24.0` | Array magnitudes, and the default numerical backend. |
| `scipy` | `>= 1.8.0` | Sparse arrays. The sparse *array* classes (`csr_array`, etc.) immlib uses appeared in 1.8. |
| `pint` | `>= 0.24.0, < 0.27` | Units. See "Compatibility with Pint" below. The suite passes against 0.24.4, 0.25.3, and 0.26.1; note that Pint 0.25+ needs Python 3.11 and 0.26+ needs 3.12, so a Python 3.10 install resolves to 0.24.x. |
| `docshare` | `>= 0.2.0` | Docstring parsing and inheritance, which `immlib` uses to read a `calc`'s inputs and outputs. |
| `joblib` | `>= 1.3.0` | Filesystem caching (`pathcache`). |
| `cloudpathlib[s3,gs,azure]` | `>= 0.18.0, < 0.26` | Remote paths (S3, Google Storage, Azure) and the local cache behind them. The upper bound is kept because immlib reads a few of cloudpathlib's private members (notably `Client._local_cache_dir`), and 0.25.0 is the version it is tested against. |
| `pyyaml` | `>= 6.0` | The `yaml` save/load format. |

## Optional dependencies

Two dependencies are optional. `immlib` imports neither at import time, and
neither is needed to import the library or to use its NumPy functionality.

* **`torch`** (`>= 2.2.0`). PyTorch tensors are fully supported as magnitudes
  and as the second numerical backend, but `torch` is imported the first time
  an operation actually needs it, never when `immlib` is imported. `immlib`
  works without PyTorch installed, with the caveat that an operation that
  needs a tensor (for example `immlib.to_tensor`) raises a descriptive
  `immlib.util._numeric.TorchNotFound` error. Predicates such as
  `immlib.is_tensor` and `immlib.math.quant` do not need PyTorch and answer
  correctly without it. (This also matters on free-threaded builds: importing
  PyTorch re-enables the GIL, so importing `immlib` never does.)
* **`pandas`** (`>= 1.5.0`). Needed only by the `csv` and `tsv` formats of
  `immlib.save` and `immlib.load`. It is imported only when one of those
  formats is actually used; installing `immlib[pandas]` provides it.

## Free-threaded (no-GIL) Python

`immlib` is written so that its own code is safe under a free-threaded
interpreter: its persistent data structures come from `pcollections`, its
shared registries are immutable snapshots, file caches are written atomically,
and thread-local state uses `contextvars`. See the "Thread Safety" section of
[Design Principles](design.md) for the details. Importing `immlib` does not
import PyTorch, which would re-enable the GIL, so a free-threaded program that
does not use PyTorch keeps the GIL disabled through the import. (A dependency
other than `immlib` may still re-enable the GIL; `immlib` cannot prevent that.)

## Compatibility with Pint

`immlib.Quantity` is a subclass of `pint.Quantity`, and it departs from Pint in
one deliberate way: a quantity may have `units=None`, meaning it has no units
at all. Pint has no such value (its `dimensionless` is a real, measurable
unit), so supporting `None` requires `immlib.Quantity` to override some of
Pint's private methods and read some of its private attributes--for example the
`_add_sub` and `_mul_div` methods that Pint's `+` and `*` funnel through, and
the `_magnitude`, `_units`, and `_REGISTRY` attributes. None of those is part
of Pint's public API, so a new Pint release may change them.

Two things guard against that. The dependency declares an upper bound
(`pint < 0.27`), so a new minor series cannot arrive by accident; and
`immlib`'s test suite includes a set of *Pint contract* tests
(`immlib.test.util.test_pint`) that fail, with a message naming the broken
assumption, if a Pint release changes any internal that `immlib` depends on.
Widening the upper bound is therefore a deliberate act: it requires running
those tests and confirming they pass against the new Pint.
