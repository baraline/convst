
import convst

from setuptools import setup, find_packages
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
	download_url = 'https://github.com/baraline/convst/archive/v0.1.5.1.tar.gz',
    version=convst.__version__,
	keywords = ['shapelets', 'time-series-classification',
             'shapelet-transform', 'time-series-transformations'],
	url="https://github.com/baraline/convst",
    author_email="antoine.guillaume45@gmail.com",
	python_requires='>=3.8',
    install_requires=[
        "matplotlib >= 3.5",
        "numba >= 0.55",
        "pandas >= 1.3",
        "scikit_learn >= 1.0",
        "joblib >= 1.1.0",
        "pyts >= 0.12",
        "scipy >= 1.7",
        "seaborn >= 0.11",
        "sktime >= 0.12",
        "numpy < 1.22, >=1.18",
        "networkx >= 2.6.3",
        #"pytest >= 7.0",
        #"sphinx >= 4.2.0",
        #"sphinx_gallery >= 0.10.1",
        #"numpydoc >= 1.1.0",
        #"alabaster >= 0.7.12"
    ],
    zip_safe=False
)
#TODO : remove docs and test packages, update versions and look for
# python versions compatibility.