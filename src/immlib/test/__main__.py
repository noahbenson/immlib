################################################################################
# immlib/test/__main__.py
#
# Main-function wrapper for the test package for immlib, using unittest.

def run_tests(verbosity=2, **kwargs):
    from unittest import main
    return main("immlib.test", verbosity=verbosity, **kwargs)

run_tests()
