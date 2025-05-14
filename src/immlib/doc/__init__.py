# -*- coding: utf-8 -*-
###############################################################################
# immlib/doc/__init__.py

"""Documentation tools that operate via decorators.

The module ``immlib.doc`` primarily contains a decorator, ``docwrap``, which
can be used to parse the inputs, outputs, parameters, and return values in a
function's docstring and to save them in a global cache of all such docstring
components. These components can then be referenced in the docstring of another
function decorated with ``@docwrap`` such that the text needn't be repeated in
every related function.

Attributes
----------
docproc: docrep.DocstringProcessor
    This object is used to process all of the doc-strings in the ``immlib``
    library; it should be used only with the ``immlib.docwrap`` decorator,
    which can safely be applied anywhere in a sequence of decorators and which
    correctly applies the ``wraps`` decorator to its argument. Function
    documentation is always processed using the ``sections=('Parameters',
    'Returns', 'Raises', 'Examples', 'Inputs', 'Outputs')`` parameter and the
    ``with_indent(4)`` decorator. The base-name for the function ``f`` is
    ``f.__module__ + '.' + f.__name__``.
"""

from ._core import (
    docwrap, make_docproc, default_docproc,
    detect_indentation, reindent)

# make_docproc lives in this subpackage (the others live in the immlib package
# and will be reclaimed there).
make_docproc.__module__ = __name__

__all__ = ("docwrap", "default_docproc")

from .._init import reclaim
reclaim(__name__, del_reclaim=True)
