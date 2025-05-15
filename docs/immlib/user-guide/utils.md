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
# General Utilities

The `immlib` library includes a large number of utility functions. Although
many of these functions are tangential to `immlib`'s core workflow components,
they are nonetheless intended to assist scientists in writing scientific
libraries and APIs. For example, the `immlib.numapi` decorator allows one to
easily write functions that accept either PyTorch tensors or NumPy
arrays. While such a function isn't necessary for supporting `immlib`'s
workflow tools, it is nonetheless a useful function for numerical APIs that
wish to work seamlessly with either library.

The utilities provided by `immlib` fall into several categories:
 * [Numerical Utilities](/user-guide/utils/numerical.md). These utilities
   include tools for transforming between NumPy arrays and PyTorch tensors and
   tools for querying them.
 * [Utilities for Physical Units](/user-guide/utils/units.md). `immlib` uses
   the `pint` library to represent physical units. Quantities with units are
   generally handled seamlessly my `immlib`'s functions, including the
   numerical utilities, but the library provides various helper functions as
   well.
 * [Functional Programming Utilities](/user-guide/utils/functional.md).
   Although `immlib` generally uses a functional and immutable style, there are
   a number of helper functions included that make certain functional
   operations that are not terribly common in Python much easier.
 * [String Utilities](/user-guide/utils/strings.md). A handful of string
   comparison functions that respect string encodings are included in the
   library.
 * [Other Utilities](/user-guide/utils/other.md). A few miscellaneous utilities
   that don't cleanly fall into any of the other categories are included in
   this section.
