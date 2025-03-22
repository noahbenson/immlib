# Design Principles

`immlib` is designed to be useful and light-weight both to a data scientist who
is using it in the interactive examination of data and to a scientific
programmer who is writing interfaces for tools. It tries to follow a few
principles that are intended to simplify the library's design; these principles
are described below.

For those who are looking to jump into using the library and evaluating its
interface, this section is helpful but can safely be skipped.


## `immlib`'s interface should be mostly functional.

[Functional](https://en.wikipedia.org/wiki/Functional_programming)
[programming](https://docs.python.org/3/howto/functional.html) is a common
programming paradigm that itself involves many prescriptions. The advantages
and details of functional programming are described in detail elsewhere, but
the important aspects in brief are:
 1. Code should consist of simple functions.
 2. Those functions should be pure&mdash;that is, they should not have internal
    state and should not edit their inputs.


## Functions should expose their parameters but be smart about default values.

A common tradeoff in the design of scientific software orients around the extent to which a function exposes its parameters to the user. To explain this tradeoff, consider the following example of a function for calculating the values of a Gaussian function:

```python
def gaussian(x):
    """Returns the Gaussian weights for the given input value or values.
    
    The Gaussian weights of the input `x` are calculated for the standard
    Gaussian function `exp( -x**2 / 2 )`.
    """
    from numpy import exp
    return exp(-x**2 / 2)
```

The main shortcoming of the above function is that it is extremely situational;
it is useful only when you need a standard Gaussian function because it does
not expose any parameters. On the other hand, this function is very easy to
maintain and understand, and the more we expose parameters, the more
complicated the function can become and the more work maintenance tends to
be. Over-parameterized functions can be just as frustrating as
underparameterized functions for different reasons, however. For example:

```python
def gaussian(x, mu, sigma, expt, normalize):
    """Returns the Gaussian weights for the given input value or values.
    
    `gaussian(x, mu, sigma, expt, False)` returns the Gaussian weights of the
    input `x` are calculated using the generalized Gaussian function:
    `exp( -(1/2) * abs((x - mu)/sigma) ** expt )`
    
    `gaussian(x, mu, sigma, q, True)` normalizes the return value such that
    the integral of the Gaussian is 1.
    
    Note that other formulations of the generalized Gaussian function often use
    the form `exp(-(abs((x - mu)/sigma) ** exp)` instead of the form used by
    this function; however such a form does not result in a standard normal
    Gaussian `exp(-(1/2) * ((x - mu)/sigma)**2)` when `expt` is equal to 2, so
    this function uses a modified version.
    """
    from numpy import exp
    if normalize:
        from scipy.special import gamma
        u = 1 + 1/expt
        const = 1 / ((2**u) * sigma * gamma(u))
    else:
        const = 1
    return const * exp(-(1/2) * (abs(x - mu)/sigma) ** expt)
```

The above version of the Gaussian function is very general, but using it will
almost certainly require reading the documentation and processing some math,
unless one is very familiar with Gaussian distributions to begin with. A
programmer who comes along a function call like `gaussian(x, mu, sig, 2, True)`
will likely wonder what the `2` and `True` in the parameter list are about. A
compromise between making functions overly complex and making them overly
situational is the common strategy of giving some of the parameters default values that are reasonable in most situations.


```python
def gaussian(x, mu=0, sigma=1, expt=2, normalize=False):
    """Returns the Gaussian weights for the given input value or values.
 
    `gaussian(x)` returns the standard Gaussian function `exp(-(1/2) * x**2)`.
    
    `gaussian(x, mu, sigma)` returns a Gaussian of unit height whose mean is
    `mu` and whose standard deviation is `sigma`.
    
    The optional argument `normalize` (default: `False`) may be set to `True`
    to indicate that the return value should be normalized such that the
    integral of the Gaussian is equal to 1. For a probability density function,
    this should be set to `True`. Otherwise, the Gaussian will have unit
    height instead of unit area.
    
    The shape of the returned Gaussian can further be modified with the
    optional parameters `mu`, `sigma`, and `expt`, which use the following form
    for a generalized Gaussian function:
    `exp( -(1/2) * abs((x - mu)/sigma) ** expt )`.
    
    Note that other formulations of the generalized Gaussian function often use
    the form `exp(-(abs((x - mu)/sigma) ** exp)` instead of the form used by
    this function; however such a form does not result in a standard normal
    Gaussian `exp(-(1/2) * ((x - mu)/sigma)**2)` when `expt` is equal to 2, so
    this function uses a modified version.
    """
    from numpy import exp
    if normalize:
        from scipy.special import gamma
        u = 1 + 1/expt
        const = 1 / ((2**u) * sigma * gamma(u))
    else:
        const = 1
    return const * exp(-(1/2) * (abs(x - mu)/sigma) ** expt)
```

This function is highly general, but it's also easy to use. Maintaining it
takes more work than the simple version, but this comes with the advantage of
the code being much more general and thus more likely to be reused. Where to
draw the line in an API between simplicity and generality is not a decision
that `immlib` can prescribe a solution to, but `immlib` endeavors to make the
more general code&mdash;code that exposes many parameters but manages them
smartly&mdash;easier to write and maintian when it comes to scientific
computing.


## Data should be loaded and computed lazily.

`immlib` uses the [`pcollections`](https://github.com/noahbenson/pcollections/)
library as a backend for immutable data structures and lazy data
structures. The `ldict` type, inparticular, allows one to define dictionaries
whose values are calculated and cached when they are first requested, and
`immlib` builds a toolkit for declaring lazy data structures and workflows
based largely on this type.

The benefits of laziness in an API that deals with data are clear when one
considers how one typically interacts with data in an analysis environment like
Jupyter. Ideally, one can load a record or subset of data from a dataset
without waiting for the entire dataset to load, especially when a dataset is
large. Simultaneously, it is often most intuitive and desirable to give users
access to data structures that represent entire datasets, despite their
size. If the records of a dataset are loaded lazily, then this is not a
problem, as only the data that the user interacts with will be loaded.

`immlib` contains many tools for constructing lazily-computed data structures
that behave superficially like normal data structures, including support for
data structures based on computational workflows and for the caching of outputs
and intermediate data of computations.
