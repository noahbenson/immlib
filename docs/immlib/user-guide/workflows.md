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

# The input data should be converted into an array:
@il.calc('data', 'n', lazy=False)
def check_data(data):
   """Ensures that the argument `data` is an array of numbers.
   
   Inputs
   ------
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
```
