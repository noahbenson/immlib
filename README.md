![immlib](https://noahbenson.github.io/immlib/_static/logo.svg "immlib")

![Build Status](https://github.com/noahbenson/immlib/actions/workflows/tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/noahbenson/immlib/graph/badge.svg?token=8KO3K6DUX4)](https://codecov.io/gh/noahbenson/immlib)
[![PyPI version](https://badge.fury.io/py/immlib.svg)](https://badge.fury.io/py/immlib)

---

**Author**: Noah C. Benson &lt;[nben@uw.edu](mailto:nben@uw.edu)&gt;  
**License**: MIT  
**[Documentation](https://noahbenson.github.io/immlib)**

---

`immlib` is a lightweight Python library that simplifies the design of
application programming interfaces (APIs) for scientific libraries. The name
immlib comes from the library’s philosophy of using immutable data to simplify
scientific workflows.

## What's in it

- **`immlib.workflow`** — immutable workflows built from `calc` units that are
  assembled into `plan` DAGs. Inputs and outputs are tracked and wired up
  automatically, results are computed lazily, and calculations can be cached in
  memory or on disk.
- **`immlib.pathlib` / `immlib.iolib`** — a single `path` function that resolves
  local and remote paths (S3, Google Storage, Azure, and OSF), downloads remote
  data into a local cache, and `save`/`load` functions that read and write a
  range of file formats.
- **quantities and units** — `immlib.Quantity` (a `pint.Quantity` that also
  supports having no units at all) and `immlib.math`, a small common namespace
  that operates on quantities, NumPy arrays, and PyTorch tensors alike, so the
  two backends are interchangeable and units are tracked through the
  arithmetic.

Alongside these are a set of utilities for testing and coercing types, handling
persistent and lazy collections, and working with numerical arguments.

## Requirements

`immlib` supports Python 3.10 through 3.14, including the free-threaded (no-GIL)
builds, and depends on NumPy, SciPy, Pint, `pcollections`, `docshare`, `joblib`,
`cloudpathlib`, and PyYAML.

PyTorch and pandas are optional. `immlib` never imports PyTorch at import time,
so it can be used without it (only operations that actually need a tensor
raise); pandas is needed only by the `csv`/`tsv` save/load formats. See the
"Stability and Compatibility" page of the
[documentation](https://noahbenson.github.io/immlib) for the supported version
ranges of each dependency and what the library promises across releases.

`immlib` is heavily based on the library
[`pimms`](https://github.com/noahbenson/pimms), which effectively served as a
prototype for `immlib`. Both libraries were motivated by a number of observations
about the design of scientific software and are an attempt to make some of these
problems easier to manage.

For more information, see the [documentation](https://noahbenson.github.io/immlib).
