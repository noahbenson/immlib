# -*- coding: utf-8 -*-
################################################################################
# immlib/doc/__init__.py

"""Documentation tools that operate via decorators.

The module `immlib.doc` primarily contains a decorator, `docwrap`, which can be
used to parse the inputs, outputs, parameters, and return values in a function's
docstring and to save them in a global cache of all such docstring
components. These components can then be referenced in the docstring of another
function decorated with `@docwrap` such that the text needn't be repeated in
every related function.
"""

from ._core import (docwrap, docproc, make_docproc)

# For each of the above we transfer it to this (non-private) subpackage.
docwrap.__module__ = __name__
docproc.__module__ = __name__
make_docproc.__module__ = __name__

__all__ = ("docwrap", "docproc")
