"""Adapted from https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='ops-analysis',
    version='0.1',
    python_requires='>=3.6',
    description='Analysis code for OPS.',  # Required
    # long_description=long_description,
    classifiers=[
        #'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],  
    packages=['image_analysis'],
)
