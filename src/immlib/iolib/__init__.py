# -*- coding: utf-8 -*-
################################################################################
# immlib/iolib/__init__.py

"""Input/output tools managed by immlib; primarily the save and load functions.

The `immlib.iolib` module contains tools for saving and loading data to/from
paths or streams. This functionality is primarily supported via the `save` and
`load` objects that behave as general (de)serializers to which formats can be
registered.
"""

from ._core import (
    Save,
    save,
    Load,
    load)

__all__ = (
    'Save',
    'save',
    'Load',
    'load')
