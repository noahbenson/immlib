#! /usr/bin/env python
################################################################################

from setuptools import setup
from pathlib import Path


base_path = Path(__file__).parent
with (base_path / 'requirements.txt').open('rt') as fl:
    requirements = fl.read().split('\n')
    requirements = [k for k in requirements if k.strip() != '']
with (base_path / 'pyproject.toml').open('rt') as fl:
    toml_lines = fl.read().split('\n')
version = None
for ln in toml_lines:
    ln = ln.strip()
    if ln.startswith('version = '):
        version = ln.split('"')[1]
        break
with (base_path / 'immlib' / '__init__.py').open('rt') as fl:
    init_text = fl.read()
desc = init_text.split("'''")[1]

setup(
    name='immlib',
    version=version,
    description=desc,
    keywords='persistent immutable functional scientific workflow',
    author='Noah C. Benson',
    author_email='nben@uw.edu',
    url='https://github.com/noahbenson/immlib',
    download_url='https://github.com/noahbenson/immlib',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS'],
    license='MIT',
    package_dir={'': 'src'},
    packages=[
        'immlib.doc', 'immlib.util', 'immlib.workflow',
        'immlib.pathlib', 'immlib.iolib',
        'immlib',
        'immlib.test.doc', 'immlib.test.iolib', 'immlib.test.pathlib',
        'immlib.test.util', 'immlib.test.workflow',
        'immlib.test'],
    package_data={'': ['LICENSE']},
    include_package_data=True,
    install_requires=requirements)
