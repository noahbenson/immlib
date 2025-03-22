# About `immlib`

`immlib` is heavily based on the library
[`pimms`](https://github.com/noahbenson/pimms), which effectively served as a
prototype for `immlib`. Both libraries were motivated by a number of
observations about the design of scientific software and are an attempt to make
some of these problems easier to manage. These observations, and a brief
explanation of `immlib`'s approach to them, are described below.

Both `immlib` and `pimms` were originally written by
[Noah C. Benson](https://github.com/noahbenson).


## Scientific data is best treated as immutable whenever possible.

Raw measurements in particular should never be edited in scientific software,
but even the intermediate pieces of data in steps of analyses are best treated
as immutable products of the analysis workflow. The design of a library can be
substantially simplified by assuming that inputs will be immutable and by
keeping functions [pure](https://en.wikipedia.org/wiki/Pure_function) (or as
pure as possible).

However, there are few Python utilities that support immutable data or take
advantage of immutable data paradigms. `immlib` uses the
[`pcollections`](https://github.com/noahbenson/pcollections) (persistent
collections) library as a backend to support a number of such tools, in
particular a system for lazily computed data that can be organized as
scientific workflows.


## APIs that are friendly to computational scientists are difficult to write.

The experience of computational scientists who use a given library should be
central to the design of most scientific libraries, but writing good APIs for
computational scientists is hard. It is generally fine&mdash;good,
even&mdash;for low-level tools designed for specific tasks to be fussy about
their arguments. A computational scientits who is interacting with data in real
time, for example using plotting libraries like
[`matplotlib`](https://matplotlib.org/) and interfaces like
[Jupyter](https://jupyter.org/), can be very inconvenienced by a library that
requires an unflexible argument schema. Similarly, a library that performs a
computation, but does so in a way that makes the computation difficult to query
or interact with can be a serious inconvenience for someone trying to
understand an analysis that involves it.

`immlib` tries to tackle a few of the major headaches in designing APIs
friendly to computational scientists. The primary of these headaches is how to
design workflows of scientific computations that are clear, modular, and
convenient, but it also includes tools for various patterns involving physical
units, numerical data, data access, and code documentation.


## Lazily-loaded data makes a good API.

Whenever possible, an API that provides an interface to the user in the form of
lazily-loaded or lazily-computed scientific data is preferable to one that
loads and/or computes everything up front unless the wait time is minimal or
the laziness incurs a hidden cost. In general, the user would rather not spend
the time waiting for all that extra data to load unless and until they are
planning to use it. Although Python provides some ways to create lazy
computations, they are fairly limited and can be very brittle. `immlib`
attempts to provide more robust and useful interfaces for lazy interfaces.

