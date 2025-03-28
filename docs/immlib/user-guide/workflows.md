# Workflows

One of `immlib`'s most powerful features is its tools for creating modular
scientific workflows. Designing workflows using `immlib` is
straightforward&mdash;managed mostly through decorators&mdash;and comes with
features like automatic caching and laziness.

To explain workflows, it's easiest to consider a concrete example. Suppose we
need to write a utility that takes, as input, an array of values and that
produces a variety of statistics describing them. We might write such code
using `immlib` as follows:

```python
import immlib as il, numpy as np

# This is a calculation unit; it produces the outputs 'mean' and 'variance'
# from the input 'data'.
@il.calc(

```
