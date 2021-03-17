import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read() 

setuptools.setup(
    name='py-irt',
    version='0.1.1',
    author='John P. Lalor',
    author_email='john.lalor@nd.edu',
    description='Bayesian IRT models in Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='http://github.com/jplalor/py-irt',
    #packages=setuptools.find_packages(),
    packages=['py_irt', 'py_irt/models'],
    install_requires=[
        "numpy>=1.17",
        "pandas>=0.25",
        "scipy>=1.3",
        "pyro-ppl>=1.5.1",
        "codecov",
        "pytest",
        "pytest-cov"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
