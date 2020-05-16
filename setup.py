# Copyright (c) 2019-present, Noe Casas
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

from setuptools import setup, find_packages

setup(name="seqp",
      version="0.1",
      author="Noe Casas",
      author_email="contact@noecasas.com",
      description="Sequence persistence library",
      keywords="sequence persistence deep learning",
      license="MIT",
      url="https://github.com/noe/seqp",
      packages=find_packages(exclude=["*.tests", "*.tests.*",
                                      "tests.*", "tests"]),
      python_requires='>=3.5.0',
      tests_require=['pytest'],
      install_requires=[
          'numpy',
          'h5py',
      ],
      classifiers=[
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Text Processing',
      ],
      )
