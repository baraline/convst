
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
    description="The Random Dilation Shapelet Transform algorithm",
	long_description_content_type='text/markdown',
	long_description=README,
    author="Antoine Guillaume",
    packages=find_packages(),
	license='BSD 2',
	#download_url = 'https://github.com/baraline/convst/archive/v0.1.4.tar.gz',
    version=convst.__version__,
	keywords = ['shapelets', 'time-series-classification',
             'shapelet-transform', 'time-series-transformations'],
	url="https://github.com/baraline/convst",
    author_email="antoine.guillaume45@gmail.com",
	python_requires='>=3.7',
    install_requires=[
        "matplotlib >= 3.5",
        "numba >= 0.54.1",
        "pandas >= 1.3.4",
        "scikit_learn >= 1.0.2",
        "joblib >= 1.1.0",
        "pyts >= 0.12"
        "scipy >= 1.7.2",
        "seaborn >= 0.11.2",
        "sktime >= 0.9",
        "numpy >= 1.19.3",
        "networkx >= 2.6.3",
        "pytest >= 6.2.5"
    ],
    zip_safe=False
)
