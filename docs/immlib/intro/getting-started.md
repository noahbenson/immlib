---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# Getting Started


## Installation

`immlib` is available on [PyPI](https://pypi.org/project/immlib/) and can be
installed using `pip`:

```bash
$ pip install immlib
```

## Using `immlib`

The `immlib` library is intended to be used in writing APIs (i.e., in your
scientific libraries and tooling) and in a REPL or notebook interface to
perform occasional utility work. Once you have installed `immlib`, you should
be able to import and use it freely:

```{code-cell}
import immlib as il

# Make a path object for the Natural Scenes Dataset (uses CloudPathLib):
nsd_path = il.path('s3://natural-scenes-dataset/', no_sign_request=True)

# Load the lines of a text file and print the first one.
lines = il.load(nsd_path / 'nsddata' / 'information' / 'knowndataproblems.txt')
print(lines[0])
```
