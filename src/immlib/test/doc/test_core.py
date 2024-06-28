# -*- coding: utf-8 -*-
################################################################################
# immlib/test/doc/test_core.py
#
# Tests of the core documentation system in immlib: i.e., tests for the code in
# the immlib.doc._core module.


# Dependencies #################################################################

from unittest import TestCase

class TestDocCore(TestCase):
    """Tests the immlib.doc._core module.

    The only public functions in the module are the `make_docproc` function and
    the `docwrap` decorator.
    """
    def test_make_docproc(self):
        from immlib.doc._core import (make_docproc, docproc)
        from docrep import DocstringProcessor
        # make_docproc() takes no arguments.
        new_docproc = make_docproc()
        # It makes a new DocstringProcessor.
        self.assertIsInstance(new_docproc, DocstringProcessor)
        # That DocstringProcessor isn't the same as the original processor.
        self.assertIsNot(docproc, new_docproc)
    def test_docwrap(self):
        from immlib.doc._core import (docwrap, make_docproc)
        # For this test we will use a custom docproc.
        dp = make_docproc()
        # First, make sure we can duplicate parameter documentation.
        @docwrap('fn1', proc=dp)
        def fn1(a, b, c=None):
            """Documentation test function 1.

            This function tests the documentation formatter `@docwrap` of the
            `immlib` library.

            Parameters
            ----------
            a : object
                The first parameter to the function.
            b : str
                The second parameter to the function.
            c : str or None, optional
                The first optional parameter to the function. The default is
                `None`. Must be a string or `None`.

            Returns
            -------
            tuple
                A tuple of `(a, b, c)`.
            """
            return (a,b,c)
        @docwrap('fn2', proc=dp)        
        def fn2(a, b, c=None, d=None):
            """Documentation test function 1.

            This function tests the documentation formatter `@docwrap` of the
            `immlib` library.

            Parameters
            ----------
            %(fn1.parameters.a)s
            %(fn1.parameters.b)s
            %(fn1.parameters.c)s
            d : int or None, optional
                The second optional parameter to the function. The default is
                `None`. Must be an integer or `None`.

            Returns
            -------
            tuple
                A tuple of `(a, b, c, d)`.
            """
            return (a,b,c,d)
        # Make sure the appropriate text made it into the fn2 documentation.
        for s in ["a : object",
                  "The first parameter to the function.",
                  "b : str",
                  "The second parameter to the function.",
                  "c : str or None, optional",
                  "The first optional parameter to the function.",
                  "d : int or None, optional",
                  "The second optional parameter to the function."]:
            self.assertIn(s, fn2.__doc__)
        self.assertEqual(fn1(1,2,3), (1,2,3))
        self.assertEqual(fn2(1,2,3,4), (1,2,3,4))

