#! /usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = ["Antoine Guillaume"]

import codecs

import toml
from setuptools import find_packages, setup

pyproject = toml.load("pyproject.toml")

def long_description():
    with codecs.open("README.md", encoding="utf-8-sig") as f:
        return f.read()
    
def setup_package():
    """Set up package."""
    setup(
        author_email=pyproject["project"]["authors"][0]["email"],
        author=pyproject["project"]["authors"][0]["name"],
        classifiers=pyproject["project"]["classifiers"],
        description=pyproject["project"]["description"],
        download_url=pyproject["project"]["urls"]["download"],
        include_package_data=True,
        install_requires=pyproject["project"]["dependencies"],
        keywords=pyproject["project"]["keywords"],
        license=pyproject["project"]["license"],
        long_description=long_description(),
        name=pyproject["project"]["name"],
        package_data={
            "convst": [
                "*.csv",
                "*.csv.gz",
                "*.arff",
                "*.arff.gz",
                "*.txt",
                "*.ts",
                "*.tsv",
            ]
        },
        packages=find_packages(
            where=".",
            exclude=["tests", "tests.*"],
        ),
        project_urls=pyproject["project"]["urls"],
        python_requires=pyproject["project"]["requires-python"],
        setup_requires=pyproject["build-system"]["requires"],
        url=pyproject["project"]["urls"]["repository"],
        version=pyproject["project"]["version"],
        zip_safe=False,
    )


if __name__ == "__main__":
    setup_package()