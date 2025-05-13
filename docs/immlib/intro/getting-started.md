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
be able to import and use it freely.


## Simple Example

```{code-cell}
import immlib as il
import torch

# Create a calculation that computes a normalized vector `u` and a length
# `xlen` given an unnormalized vector `x`.
@il.tensor_args(keep_arrays=True)
@il.calc('u', 'xlen')
def normalize_vector(x):
    print("Normalizing vector...")
    xlen = torch.sqrt(torch.sum(x**2))
    u = x / xlen
    return (u, xlen)

# Create another calculation that finds the signed distance between a point
# `y` and the vector `x`, as well as the point of intersection.
@il.tensor_args(keep_arrays=True)
@il.calc('distance', 'nearest_point')
def point_vec_intersection(u, y):
    print('Calculating distance...')
    d = torch.dot(u, y)
    return (d, u*d)

# Declare the plan by putting together all the calculations:
distance_plan = il.plan(
    normalize_step=normalize_vector,
    calculate_step=point_vec_intersection)
                
# Make a plandict of the results:
pd = distance_plan(x=[0.0, 1.0], y=[2.0, 1.0])

print('xlen:',     pd['xlen'])
print('distance:', pd['distance'])
print('u:',        pd['u'])
print('nearest:',  pd['nearest_point'])
```
