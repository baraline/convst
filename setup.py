
from setuptools import setup, find_packages
from codecs import open
import numpy
import os

ROOT = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(ROOT, 'README.md'), encoding="utf-8") as f:
    README = f.read()
	
setup(
    name="cst",
    description="The Convolutional Shapelet Transform algorithm",
	long_description=README,
    long_description_content_type='text/markdown',
    author="Antoine Guillaume",
    packages=find_packages(),
	license='BSD 2',
	download_url = 'https://github.com/baraline/cst/archive/v_01.tar.gz'
	include_dirs=[numpy.get_include()],
    version="0.1",
	keywords = ['shapelets', 'time-series-classification', 'shapelet-transform','convolutional-kernels']
	url="https://github.com/baraline/CST",
    author_email="antoine.guillaume45@gmail.com",
	python_requires='>=3.7',
    install_requires=[
        "matplotlib >= 3.1",
        "numba >= 0.50",
        "pandas >= 1.1",
        "scikit_learn >= 0.24",
        "scipy >= 1.5.0",
        "seaborn >= 0.11",
        "sktime >= 0.5.3",
        "numpy >= 1.18.5"
    ],
    zip_safe=False
)