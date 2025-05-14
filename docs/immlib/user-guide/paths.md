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
# Local and Remote Path Utilities

`immlib` includes a number of functions that integrate with the
[`cloudpathlib`](https://cloudpathlib.drivendata.org/stable/) and the
[`pathlib`](https://docs.python.org/3/library/pathlib.html) libraries in order
to provide a single interface for managing paths. This is simplified to a large
extent by the similarity between the libraries to begin with.

To obtain a path object&mdash;either a `CloudPath` or `Path`&mdash;the
`immlib.path` function can be used.

```{code-cell}
import immlib as il

p = il.path(il.doc.__file__)
print(type(p), ': ', il.pathstr(p), sep='')

cp = il.path(
    's3://natural-scenes-dataset/nsddata/',
    no_sign_request=True)
print(type(cp), ': ', il.pathstr(cp), sep='')
```

`immlib` also provides an `OSFPath` type for accessing the Open Science
Framework ([`osf.io`](https://osf.io/)). An OSF repository can be accessed using
its 5-digit ID:

```{code-cell}
import immlib as il

p = il.path('osf://bw9ec/')
list(p.iterdir())
```
