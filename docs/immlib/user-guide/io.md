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
# Input and Output Utilities

`immlib` provides a simple pair of utilities for defining input and output
formats. Any format that has been declared can be used to either save data to
files/streams (using `immlib.save()`) or to load data from files/streams (using
`immlib.load()`).


## `immlib.load`: Importing Data

To declare an input format, one can use the `@immlib.load.register` decorator on
a function that loads the associated data. For example, the following code
declares an importer for the `JSON` file format:

```python
import immlib as il

@il.load.register('json', '.json', mode='t')
def load_json(stream, /, **kwargs):
    """Loads an object from a JSON stream or path and returns the object.

    All keywords are passed along to the `json.load` function.
    """
    import json
    return json.load(stream, **kwargs)
```

The `@il.load.register` call above includes three arguments. The first is the
name of the format; this name must be unique across all formats registered to
`immlib`'s `load` utility. If one calls `il.load(filename, 'json')` then the
`load_json()` function would always be called no matter what ending the filename
has. The second argument is a filename ending (or list of endings) that
typically indicate this format. The final option (`mode='t'`) indicates that the
stream should be opened in text mode.

Once this load function has been registered, the `il.load()` function can be
used to load `json` data automatically. `il.load` accepts paths, including
non-local paths such as S3 paths.

For example:

```{code-cell}
import immlib as il

filename = il.util.__file__
print("Catting file", filename, "...")

lines = il.load(filename, 'text')
for ln in lines:
    print(ln)
```


## `immlib.save`: Exporting Data

Similar to `immlib.load`, the `immlib.save` function can be accessed using the
`immlib.save.register` function.

```python
import immlib as il

@il.save.register('json', '.json', mode='t')
def save_json(stream, obj, /, default=json_default, **kwargs):
    """Saves an object as a JSON string or raises a TypeError if not possible.

    All keywords are passed along to the `json.dump` function. The `default`
    option uses the `immlib.iolib.json_default` function, which is different
    than the default used by `json.dump`, but all other options are unaltered.
    """
    import json
    json.dump(obj, stream, default=default, **kwargs)
```

Note that in this example and the previous example, additional keyword arguments
are accepted after the `stream` argument (and the `obj` object that is being
saved). Any arguments given to the `save` or `load` functions are passed along.

Here is an example of using save and load together to first save a YAML file and
then to load and print it out:

```{code-cell}
import immlib as il
from tempfile import TemporaryDirectory
from pathlib import Path

yaml_data = [{'header1': [1, 2, 3], 'header2': [{'a': 10, 'b': 20}, 'end']}]

# Do these operations in a temporary directory:
with TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)

    # Save the YAML data out to a file:
    il.save(tmpdir / 'file.yaml', yaml_data)
    
    # Now load it as text and print it line by line:
    lines = il.load(tmpdir / 'file.yaml')
    for ln in lines:
        print(ln)
```

## Predefined Formats

The following formats are predefined in `immlib`:
 * `'str'`. Reads or writes a plain string as text. If an object that is not a
   string is saved, then it is `str(obj)` is written.
 * `'bytes'`. Reads or writes a plain byte string as binary.
 * `'repr'`. Writes `repr(obj)` as a text string.
 * `'text'` (`*.txt`). Reads or writes a list of strings.
 * `'pickle'` (`*.pickle`, `*.pcl`, `*.pkl`). Reads or writes a pickle file.
 * `'numpy'` (`*.npy`, `*.np`, `*.numpy`, `*.npz`). Reads or writes a numpy
   file.
 * `'json'` (`*.json`). Reads or writes a JSON file.
 * `'yaml'` (`*.ylm`, `*.yaml`). Reads or writes a YAML file.
 * `'csv'` (`*.csv`). Reads or writes a comma-separated-value file using
   `pandas`.
 * `'tsv'` (`*.tsv`). Reads or writes a tab-separated-value file using
   `pandas`.

## Formats and Threads

`immlib.save` and `immlib.load` are single objects shared by the whole
program, so registering a format is a change that every thread sees. Both are
safe to use from several threads at once, including in a free-threaded
interpreter, and no locking is needed by the caller.

The registry each of them keeps is two maps that have to agree: one from a
format's name to the format, and one from each of its suffixes to the same
format. It is held as a single immutable snapshot of both maps, which is
replaced wholesale whenever it changes. A call that reads the registry ---
`deduce_format`, or any `save`/`load` call that deduces a format from a path
--- takes the snapshot it is going to use in one step, so it never sees a
format that has been registered under its name but not yet under its
suffixes, or one whose suffixes have been removed but whose name has not.

```{code-cell}
il.save.formats['json']
```

Registration itself is serialized, which is what makes its duplicate checks
mean anything: `register` refuses a name or a suffix that is already taken,
and without a lock two threads could both find the same name free and both
claim it.

```{code-cell}
fmt = il.save.copy()

try:
    fmt.register(fmt.formats['json'])
except RuntimeError as e:
    print(e)
```

Two things are left to the caller. A `Format` object must not be modified
after it has been registered --- build a new one and register it under a new
name instead. And saving to or loading from the same path in two threads at
once is an ordinary file-level race, which the format system does not and
cannot arbitrate; `immlib.save` writes wherever it is told to write.

```{note}
The usual advice applies here as it does for unit registries: register the
formats a program needs while it is starting up, before the threads that use
them exist. The guarantees above are about not corrupting the registry, not
about making a format appear at a well-defined moment in another thread's
work.
```
