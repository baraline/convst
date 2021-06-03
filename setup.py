
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

setup(
  name = 'cst',         # How you named your package folder (MyLib)
  packages = ['YOURPACKAGENAME'],   # Chose the same as "name"
  version = '0.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'TYPE YOUR DESCRIPTION HERE',   # Give a short description about your library
  author = 'YOUR NAME',                   # Type in your name
  author_email = 'your.email@domain.com',      # Type in your E-Mail
  url = 'https://github.com/user/reponame',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['SOME', 'MEANINGFULL', 'KEYWORDS'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'validators',
          'beautifulsoup4',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)