import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read() 

setuptools.setup(
    name='py-irt',
    version='0.0.1',
    author='John P. Lalor',
    author_email='john.lalor@nd.edu',
    description='Bayesian IRT models in Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='http://github.com/jplalor/py-irt',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
