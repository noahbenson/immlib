#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# docs/generate_api.py
#
# Generates the API-reference pages of the immlib documentation as MyST
# Markdown, one page per public module.
#
# Jupyter Book 2 (mystmd) has no Sphinx-style autodoc, so the API reference is
# generated here, before the book is built, from the objects' own docstrings.
# The docstrings are the single source of truth: every public object's
# docstring is already required to parse as NumPy-style documentation by
# immlib's own test suite (immlib.test.util.test_docs), so this script only has
# to render them, not to repair them.
#
# Run it with the same interpreter that can import immlib:
#
#     python docs/generate_api.py
#
# It writes docs/immlib/api/*.md and a docs/immlib/api.md index.

import inspect
import re
from pathlib import Path

from docshare import docparse

import immlib
import immlib.math
import immlib.pathlib
import immlib.iolib
import immlib.util
import immlib.workflow


# Modules to document, in the order the pages should appear.
MODULES = [
    ('immlib', immlib),
    ('immlib.math', immlib.math),
    ('immlib.workflow', immlib.workflow),
    ('immlib.pathlib', immlib.pathlib),
    ('immlib.iolib', immlib.iolib),
    ('immlib.util', immlib.util),
]

def page_slug(name):
    "Returns the file stem used for a module's page."
    return name.replace('.', '-')

def format_paragraph(lines):
    "Joins a string or list of lines of prose into one Markdown paragraph."
    if isinstance(lines, str):
        lines = lines.split('\n')
    text = ' '.join(ln.strip() for ln in lines)
    return ' '.join(text.split())

def render_docstring(obj):
    "Renders an object's docstring as MyST Markdown, or '' if it has none."
    if not (obj.__doc__ and obj.__doc__.strip()):
        return ''
    try:
        doc = docparse(obj, format='numpy',
                       custom=immlib.workflow.CALC_DOC_SECTIONS)
    except Exception:
        # A docstring that does not parse is a defect in the library, not in
        # the generator; fall back to the raw text rather than failing the
        # build (immlib's test suite is what catches the defect).
        return inspect.getdoc(obj) or ''
    out = []
    if doc.summary:
        out.append(format_paragraph(doc.summary))
    if doc.description:
        out.append(format_paragraph(doc.description))
    for section in doc.sections:
        body = []
        for item in section.items:
            names = ', '.join(f'``{n}``' for n in item.names)
            if item.type:
                head = f'{names} (*{item.type}*)' if names else f'*{item.type}*'
            else:
                head = names
            desc = format_paragraph(item.description)
            if desc:
                body.append(f'- **{head}** — {desc}')
            else:
                body.append(f'- **{head}**')
        if body:
            out.append(f'**{section.name}**\n\n' + '\n'.join(body))
    return '\n\n'.join(out)

def format_signature(name, obj):
    "Returns a fenced Python signature block for a callable or class."
    try:
        sig = str(inspect.signature(obj))
    except (TypeError, ValueError):
        sig = ''
    # immlib's modules use ``from __future__ import annotations``, so a
    # signature renders its annotations as string literals; strip the quotes
    # so the rendered signature reads as ordinary Python.
    sig = re.sub(r"'([^']*)'", r'\1', sig)
    keyword = 'class ' if inspect.isclass(obj) else ''
    return f'```python\n{keyword}{name}{sig}\n```'

def render_object(name, obj):
    "Renders one public object as a MyST section."
    lines = [f'### {name}', '']
    if callable(obj):
        lines += [format_signature(name, obj), '']
    doc = render_docstring(obj)
    if doc:
        lines += [doc, '']
    return '\n'.join(lines)

def render_module(name, module):
    "Renders a whole module's page."
    lines = [f'# {name}', '']
    moddoc = render_docstring(module)
    if moddoc:
        lines += [moddoc, '']
    for member in sorted(getattr(module, '__all__', ())):
        obj = getattr(module, member, None)
        if obj is None or getattr(obj, '__module__', name) is None:
            continue
        lines.append(render_object(member, obj))
    return '\n'.join(lines) + '\n'

def main():
    here = Path(__file__).resolve().parent / 'immlib'
    outdir = here / 'api'
    outdir.mkdir(parents=True, exist_ok=True)
    index = ['# API Reference', '',
             'The pages below are generated from the docstrings in the source '
             'by `docs/generate_api.py`; they are the same docstrings the '
             'library tests, so they cannot drift from the code.', '']
    for name, module in MODULES:
        text = render_module(name, module)
        (outdir / f'{page_slug(name)}.md').write_text(text)
        index.append(f'- [{name}](api/{page_slug(name)}.md)')
    (here / 'api.md').write_text('\n'.join(index) + '\n')
    print(f'wrote {len(MODULES)} API pages to {outdir}')

if __name__ == '__main__':
    main()
