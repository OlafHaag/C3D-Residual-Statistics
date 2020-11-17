#!/usr/bin/env python
""" Installation script for c3dstats package """

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

from c3dstats import version

setup(name='c3dstats',
      version=version,
      description='Python package for extracting statistics from C3D motion capture files.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Olaf Haag',
      author_email='contact@olafhaag.com',
      url='https://github.com/olafhaag/C3D-Statistics',
      classifiers=['License :: OSI Approved :: MIT License',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Operating System :: OS Independent',
                   'Environment :: Console',
                   'Topic :: Scientific/Engineering',
                   'Intended Audience :: Science/Research',
                   'Development Status :: 3 - Alpha',
                   ],
      keywords='c3d motion-capture animation 3d statistics',
      packages=find_packages(),
      install_requires=['c3d >= 0.3.0',
                        'numpy',
                        'matplotlib >= 2.1.0'],
      extras_require={'dev': ['pytest', 'hypothesis'],
                      },
      entry_points={'console_scripts': ['c3dstats=c3dstats.c3dstats:main',
                                        ],
                    },
      project_urls={'Bug Reports': 'https://github.com/olafhaag/C3D-Statistics/issues',
                    'Source': 'https://github.com/olafhaag/C3D-Statistics/',
                    },
      )
