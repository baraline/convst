import setuptools 

setuptools.setup(
    name="CST",
    description="Convolutional Shapelet Transform",
    author="Antoine Guillaume",
    packages=setuptools.find_packages(),
    version="0.1.0",
    author_email="antoine.guillaume45@gmail.com",
	python_requires='>=3.8',
    install_requires=[
        "matplotlib >= 3.1",
        "networkx >= 2.4",
        "numba >= 0.50",
        "pandas >= 1.1",
        "scikit_learn >= 0.24",
        "scipy >= 1.5.0",
        "seaborn >= 0.11",
        "sktime >= 0.5.3",
        "wildboar >= 1.0.9",
        "numpy >= 1.18.5"
    ],
	zip_safe=False
)