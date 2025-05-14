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
# Workflows

One of `immlib`'s most powerful features is its tools for creating modular
scientific workflows. Designing workflows using `immlib` is
straightforward&mdash;managed mostly through decorators&mdash;and comes with
features like automatic caching and laziness.

## Introductory Example

To explain workflows, it's easiest to consider a concrete example. Suppose we
need to write a utility that takes, as input, an array of values and that
produces a variety of statistics describing them. We might write such code
using `immlib` as follows:

```{code-cell}
import immlib as il, numpy as np

# The input data should be converted into an array:
@il.calc('data', 'n', lazy=False)
def check_data(data):
   """Ensures that the argument `data` is an array of numbers.
   
   Parameters
   ----------
   data : array-like
       The input data for which statistics should be calculated. The data must
       be a 1-dimensional vector of numbers, and it is converted into a NumPy
       array by this calculation.
   
   Outputs
   -------
   data : numpy.ndarray
       A 1-dimensional NumPy array of numbers.
   n : int
       The number of data-points in `data`.
   
   Raises
   ------
   TypeError
       If `data` is not a 1-dimensional vector of numbers.
   """
   print('Checking data...')
   data = il.to_array(data)
   if not il.is_array(data, shape=(-1,), dtype=np.number):
       raise TypeError("data must be a vector-like sequence of numbers")
   return (data, len(data))

@il.calc('sum_of_squares', 'sum')
def calc_sums(data):
    """Calculates the sum and sum of squares of the data.
    
    Outputs
    -------
    sum_of_squares : number
        The sum of squares of the absolute values of the data.
    sum : number
        The sum of the values in data.
    """
    print('Calculating sums...')
    sum_of_sq = np.sum(np.abs(data)**2)
    sum = np.sum(data)
    return (sum_of_sq, sum)

@il.calc('mean', 'var', 'std')
def calc_mean_etc(sum, sum_of_squares, n):
    """Calculates the mean, variance, and standard deviation of the data.
    
    Outputs
    -------
    mean : number
        The mean of the data.
    var : number
        The variance of the data.
    std : number
        The standard deviation of the data.
    """
    print('Calculating mean, etc...')
    mean = sum / n
    var = (sum_of_squares / n) - mean**2
    std = np.sqrt(var)
    return (mean, var, std)

# Make a plan for calculating these statistics:
stats_plan = il.plan(
    check_step=check_data,
    sums_step=calc_sums,
    mean_step=calc_mean_etc)
```

Once we have set up the example code above, we can create an instance of our
stats plan called a `plandict`. When we create the `plandict`, the `check_data`
function will automatically run because it is marked as not lazy
(`lazy=False`):

```{code-cell}
# Make a plandict for a set of data:
stats = stats_plan(data=np.random.randn(100))
print("The type of stats is", type(stats))
```

The `plandict` has a key for each of the inputs and outputs in the entire
calculation plan:

```{code-cell}
sorted(stats.keys())
```

The values associated with these keys are calculated according to the
calculation plan, with the inputs for each calculation being derived from the
outputs of other calculations. In the plan `stats_plan` above, there is no
output named `data`, so it is considered an input of the plan.

```{code-cell}
stats_plan.inputs
```

When the values associated with keys are requested, they get calculated and
cached. So in the example above, in which messages are printed as the values
are calculated, the messages will be printed only the first time a key is
requested.

```{code-cell}
# The message about the sums being calculated will appear immediately when this
# key is requested:
sum_sq = stats['sum_of_squares']
# But it won't appear again here:
print("The sum of squares is", stats['sum_of_squares'])

# Similarly, it won't appear again when we request the sum, because the sum was
# calculated in the same function as the sum of squares.
print("The sum is", stats['sum'])
```

Downstream values such as `mean` and `var` aren't calculated until they are
required:

```{code-cell}
print("The mean is", stats['mean'])
print("The standard deviation is", stats['std'])
```


## Calculation Metadata

The `@calc` decorator attaches metadata to the calculation functions by
creating a field `calc`. The `calc` object that manages the calculation's
integration with any plans is found here.

```{code-cell}
calc_sums.calc
```


## Tracking Documentation

Calculations and plans also track the documentation for their inputs and
outputs. These can be accessed via the `input_docs` and `output_docs` fields.

```{code-cell}
print(calc_sums.calc.output_docs['sum'])
```

```{code-cell}
print(check_data.calc.input_docs['data'])
```

Plans have a docstring that is automatically generated from their constituent
calculations.

```{code-cell}
print(stats_plan.__doc__)
```





