# -*- coding: utf-8 -*-
###############################################################################
# immlib/doc/_core.py


# Dependencies ################################################################

from re import compile as re_compile
from functools import wraps
from docrep import DocstringProcessor


# The Document Processor ######################################################
def make_docproc():
    """Creates and returns a document preprocessor."""
    docproc = DocstringProcessor()
    # We need to add a few features to the docproc's members so that we can
    # process the Inputs and Outputs sections when present.
    docproc.param_like_sections = \
        docproc.param_like_sections + ['Inputs','Outputs']
    docproc.patterns['Inputs'] = re_compile(
        docproc.patterns['Parameters'].pattern
            .replace('Parameters', 'Inputs')
            .replace('----------', '------'))
    docproc.patterns['Outputs'] = re_compile(
        docproc.patterns['Parameters'].pattern
            .replace('Parameters', 'Outputs')
            .replace('----------', '-------'))
    return docproc
# This gets imported into `immlib` as `immlib.docproc`, which is the name it
# should be known by, but we create it here in this submodule.
_initial_global_docproc = make_docproc()


# The docwrap Decorator #######################################################
def _docwrap_helper(f, fnname, indent=None, proc=Ellipsis):
    # If no ident number was provided, deduce it.
    if indent is None:
        if not hasattr(f, '__doc__') or f.__doc__ is None:
            # Doesn't matter, no documentation.
            indent = 0
        else:
            lines = f.__doc__.split('\n')
            # We always skip the first line (the one that starts with """).
            lines = lines[1:]
            # Strip the lines.
            striplines = [s.lstrip() for s in lines]
            # Pick out the ones with text in them and calculate the
            # indentation.
            indents = [
                len(ws_s) - len(s)
                for (ws_s,s) in zip(lines, striplines)
                if len(s) > 0]
            # The minimum is the one we want.
            indent = min(indents) if len(indents) > 0 else 0
    if proc is Ellipsis:
        # We need to be able to obtain the default docproc object, even during
        # the process of importing the library, when we use the
        # _initial_global_docproc. Once immlib has been loaded, we use
        # `immlib.docproc` instead.
        try:
            from immlib import docproc as proc
        except ImportError:
            proc = _initial_global_docproc
    ff = f
    ff = proc.with_indent(indent)(ff)
    fd = proc.get_sections(base=fnname, sections=proc.param_like_sections)
    ff = fd(ff)
    ff = wraps(f)(ff) if f is not ff else f
    # Post-process the documentation sections.
    for section in ('parameters', 'other_parameters', 'inputs', 'outputs'):
        k = fnname + '.' + section
        v = proc.params.get(k, '')
        if len(v) == 0: continue
        for ln in v.split('\n'):
            # Skip lines that start with whitespace.
            if len(ln) == 0 or ln[0].strip() == '':
                continue
            pname = ln.split(':')[0].strip()
            proc.keep_params(k, pname)
    return ff
def docwrap(f=None, /, *, indent=None, proc=Ellipsis):
    """Applies standard doc-string processing to the decorated function.

    The ``immlib.docwrap`` decorator applies a standard set of pre-processing
    to the docstring of the function that follows it. This processing amounts
    to using the ``docrep`` module's ``DocstringProcessor`` as a filter on the
    documentation of the function. The function's documentation is always
    placed in the base-name equal to its fully-qualified namespace name.

    When called as ``@docwrap(name)`` for a string ``name``, the documentation
    for the decorated function is instead placed under the base-name ``name``.

    Parameters
    ----------
    f : function or str or None, optional
        The function to be decorated, when ``@docwrap`` is used alone as a
        decorator, or when used as a decorator with only the other options
        given, such as ``@docwrap(indent=8)``. If a string is given, as in
        ``@docwrap('immlib.dictmap')`` then the given string is used as the
        function's name instead of its ``__module__`` plus its
        ``__name__``. This is mostly useful when using ``@docwrap`` with a
        function defined in a private submodule; for example ``immlib.dictmap``
        is defined in ``immlib.util._core`` but is imported into a reclaimed by
        the ``immlib`` core namespace, so it is typically considered to belong
        to that namespace.

        Typically this argument does not need to be provided as it is given
        after the decorator line; the exception to this is when a string is
        given.
    indent : None or int, optional
        The number of spaces that are used as indentation before lines in the
        docstring. This is mostly useful when decorated functions appear in
        indented contexts and thus the default indentation of 4 is
        inappropriate. If ``None`` is given, then the decorator finds the
        non-empty line, not including the first line, with the smallest
        indentation and uses that.
    proc : docrep.DocstringProcessor or Ellipsis, optional
        The `proc` option provides the document processor object from the
        ``docrep`` library that should be used to process the decorated
        object. Because these objects can be specifically configured to enable
        different docstring formats, this option is provided to the user. The
        default value is ``Ellipsis``, in which case the ``immlib.docproc``
        object is used. The ``docproc`` object has been configured to work with
        the ``Input`` and ``Output`` sections that are used with calculations
        and plans. The ``immlib.with_docproc`` function can be used to change
        the ``immlib.docproc`` object that is used in a local code-block.

    Returns
    -------
    object
        The decorated function or object, after its docstring has been parsed.

    See Also
    --------
    default_docproc :
        Run a code-block with a specific default docstring processor.

    """
    # If we've been given a string, then we've been called as @docwrap(name)
    # instead of @docwrap.
    if f is None:
        return lambda fn: _docwrap_helper(
            fn, fn.__module__ + '.' + fn.__name__,
            indent=indent,
            proc=proc)
    if isinstance(f, str):
        return lambda fn: _docwrap_helper(fn, f, indent=indent, proc=proc)
    else:
        return _docwrap_helper(
            f, f.__module__ + '.' + f.__name__,
            indent=indent,
            proc=proc)
class default_docproc:
    """Context manager for setting the default ``immlib.docproc`` document
    processing object.

    The following code-block can be used to evaluate the code represented by
    ``...`` using the ``docrep.DocstringProcessor`` object ``docproc`` as the
    default ``immlib.docproc`` processor:

    .. code-block:: python

      with immlib.default_docproc(docproc):
          ...

    If the ``immlib.docproc`` value has accidentally been corrupted, then it
    can be reset using the following:

    .. code-block:: python

      immlib.default_docproc.reset()

    Parameters
    ----------
    docproc : docrep.DocstringProcessor object
        The docstring processing object to use as the default in ``immlib`` in
        the contextualized code.

    See Also
    --------
    docwrap : decorator that simplifies the use of the ``docrep`` library.

    """
    __slots__ = ('original', 'docproc')
    def __init__(self, docproc):
        if not isinstance(docproc, DocstringProcessor):
            raise TypeError("docproc must be a docrep.DocstringProcessor")
        object.__setattr__(self, 'original', None)
        object.__setattr__(self, 'docproc', docproc)
    def __enter__(self):
        import immlib
        object.__setattr__(self, 'original', immlib.docproc)
        immlib.docproc = self.docproc
        return self.ureg
    def __exit__(self, exc_type, exc_val, exc_tb):
        import immlib
        immlib.docproc = self.original
        return False
    def __setattr__(self, name, val):
        raise TypeError("default_docproc is immutable")
    @staticmethod
    def reset():
        """Resets the value of ``immlib.docproc`` to its value when the
        ``immlib`` library was originally loaded."""
        import immlib
        immlib.docproc = _initial_global_docproc


# Other Utilities #############################################################

@docwrap
def detect_indentation(text, /, skip_first=True, tabsize=8):
    """Given a block of text that is part of a docstring, guess the level of
    indentation used to write it.

    This function accepts a string that contains multiple lines and guesses the
    indentation level used to write it. It does this by splitting the lines and
    finding the line that starts with the smallest number of spaces. That
    number of spaces is the indentation guess.

    By default, this function skips the first line because it is customary to
    start docstrings out with an unindented line. This behavior can be changed
    by setting the optional argument `skip_first` to ``False``.

    Parameters
    ----------
    text : str
        The text whose indentation is to be guessed.
    skip_first : bool, optional
        Whether to skip the first line when detecting the indentation level.
        The default is ``True``.
    tabsize : int, optional
        The number of spaces in a tab-stop; used to replace the tab characters
        in each line using the method ``str.expandtabs``. The default is ``8``.

    Returns
    -------
    int
        The number of spaces of indentation detected.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"detect_indentation requires str but got {type(text)}")
    lns = text.split('\n')
    ident = None
    if skip_first:
        lns = lns[1:]
    for ln in lns:
        ln = ln.expandtabs(tabsize)
        if ln.strip() == '':
            continue
        ln_ident = len(ln) - len(ln.lstrip())
        if ident is None:
            ident = ln_ident
        elif ln_ident < ident:
            ident = ln_ident
    return ident
@docwrap
def reindent(text, new_indent=0, /,
             skip_first=True, tabsize=8, final_endline=True):
    """Returns a block of text with a different indentation.

    ``reindent(text, n)`` returns a copy of `text` after removing its current
    indentation level and uniformly reindenting the text with ``n`` spaces. The
    first line is skipped entirely, and the current indentation level is
    detected using ``detect_indentation``.

    Parameters
    ----------
    text : str
        The text that is to be reindented.
    new_indent : int, optional
        The new indentation level. If this is not provided, then the default is
        0, meaning that the text will be unindented.
    skip_first : bool, optional
        Whether or not to skip the first line.
    tabside : int, optional
        How large to consider tab characters in the text; this is used with the
        ``str.expandtabs`` method. The default is 8.
    final_endline : bool, optional
        Whether the returned string should end with a newline or not. The
        default is ``True``.

    Returns
    -------
    str
        A duplicate of `text` with updated indentation.
    """
    # Get the current indentation level:
    currind = detect_indentation(text, skip_first=skip_first, tabsize=tabsize)
    # Split the text into lines.
    lns = text.split('\n')
    # Remove all the current indentations:
    newlns = []
    if skip_first:
        newlns.append(lns[0].expandtabs(tabsize))
        lns = lns[1:]
    head = ' ' * currind
    newhead = ' ' * new_indent
    for ln in lns:
        ln = ln.expandtabs(tabsize)
        if ln.strip() == '':
            newlns.append('')
            continue
        if ln.startswith(head):
            ln = ln[currind:]
        newlns.append(newhead + ln)
    newtext = '\n'.join(newlns)
    if final_endline and newtext[-1] != '\n':
        newtext += '\n'
    return newtext
