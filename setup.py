
import convst

from setuptools import setup, find_packages
from codecs import open
import numpy
import os

ROOT = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(ROOT, 'README.md'), encoding="utf-8") as f:
    README = f.read()
	
setup(
    name="convst",
    description="The Convolutional Shapelet Transform algorithm",
	long_description_content_type='text/markdown',
	long_description=README,
    author="Antoine Guillaume",
    packages=find_packages(),
	license='BSD 2',
	download_url = 'https://github.com/baraline/convst/archive/v0.1.2.tar.gz',
    version=convst.__version__,
	keywords = ['shapelets', 'time-series-classification', 'shapelet-transform','convolutional-kernels'],
	url="https://github.com/baraline/convst",
    author_email="antoine.guillaume45@gmail.com",
	python_requires='>=3.7',
    install_requires=[
        "matplotlib >= 3.1",
        "numba >= 0.50",
        "pandas >= 1.1",
        "scikit_learn >= 0.24",
        "scipy >= 1.5.0",
        "seaborn >= 0.11",
        "sktime >= 0.5",
        "numpy >= 1.18.5",
		"sphinx_gallery >= 0.8",
		"numpydoc >= 1.0"
    ],
    zip_safe=False
)
