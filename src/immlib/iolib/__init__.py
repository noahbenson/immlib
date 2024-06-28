# -*- coding: utf-8 -*-
################################################################################
# pimms/iolib/__init__.py

"""Input/output tools managed by pimms; primarily the save and load functions.

The `pimms.iolib` module contains tools for saving and loading data to/from
paths or streams. This functionality is primarily supported via the `save` and
`load` objects that behave as general (de)serializers to which formats can be
registered.
"""

from ._core import (
    Save,
    save,
    Load,
    load,
)

__all__ = (
    'Save',
    'save',
    'Load',
    'load',
)

# Mark these as native to this module.
Save.__module__ = __name__
save.__module__ = __name__
Load.__module__ = __name__
load.__module__ = __name__
