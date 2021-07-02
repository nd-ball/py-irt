[![Build Status](https://travis-ci.com/nd-ball/py-irt.svg?branch=master)](https://travis-ci.com/nd-ball/py-irt)
[![codecov.io](https://codecov.io/gh/nd-ball/py-irt/coverage.svg?branch=master)](https://codecov.io/gh/nd-ball/py-irt)

# py-irt

Bayesian IRT models in Python

## Overview

This repository includes code for fitting Item Response Theory (IRT) models using variational inference.

At present, the one parameter logistic (1PL) model, aka Rasch model, two parameter logistic model (2PL) and four parameter logistic model (4PL) have been implemented.
The user can specify whether vague or hierarchical priors are used.
The three-parameter logistic model is in the pipeline and will be added when available.

## License

py-irt is licensed under the [MIT license](https://opensource.org/licenses/MIT).

## Installation

py-irt is now available on PyPi!

### Pre-reqs

1. Install [PyTorch](https://pytorch.org/get-started/locally/).
2. Install [Pyro](https://pyro.ai/)
3. Install py-irt:

```shell
pip install py-irt
```

OR

```shell
git clone https://github.com/nd-ball/py-irt.git
cd py-irt
pip install -r requirements.txt
```

## Usage

Once you install from PyPI, you can use the following command to fit an IRT
model on the scored predictions of a dataset. For example, if you were to run py-irt with the 4PL model on the scored predictions of different transformer models on the SQuAD dataset, you'd do this:
`py-irt train 4pl ~/path/to/dataset/eg/squad.jsonlines /path/to/output/eg/test-4pl/`

## FAQ

1. What kind of output should I expect on running the command to train an IRT model?

You should see something like this when you run the command given above:
![image](https://user-images.githubusercontent.com/40918514/123986740-3eccd100-d9cf-11eb-8e58-ba5ad6c977ce.png)

2. I tried installing py-irt using pip from PyPI. But when I try to run the command `py-irt train 4pl ~/path/to/dataset/eg/squad.jsonlines /path/to/output/eg/test-4pl/`, I get an error that says `bash: py-irt: command not found`. How do I fix this?

You may be running into this error because of the old version on PyPI, which has not yet been configured for CLI usage. It's in the pipeline and you should be able to see it soon! In the meantime, do the following steps for replicating CLI function in the repo using these commands:

```shell
git clone https://github.com/nd-ball/py-irt.git
cd py-irt
mv ~/py-irt/py_irt/cli.py ~/py-irt/
python cli.py train 4pl ~/path/to/dataset/eg/squad.jsonlines /path/to/output/eg/test-4pl/
```

3. How do I evaluate a trained IRT model?

Evaluation of a trained IRT model is yet to be implemented. You're welcome to implement it and open a pull request!

## Citations

If you use this code, please consider citing the following paper:

```shell
@inproceedings{lalor2019emnlp,
  author    = {Lalor, John P and Wu, Hao and Yu, Hong},
  title     = {Learning Latent Parameters without Human Response Patterns: Item Response Theory with Artificial Crowds},
  year      = {2019},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
}
```

Implementation is based on the following paper:

```shell
@article{natesan2016bayesian,
  title={Bayesian prior choice in IRT estimation using MCMC and variational Bayes},
  author={Natesan, Prathiba and Nandakumar, Ratna and Minka, Tom and Rubright, Jonathan D},
  journal={Frontiers in psychology},
  volume={7},
  pages={1422},
  year={2016},
  publisher={Frontiers}
}
```

## Contributing

This is research code. Pull requests and issues are welcome!

## Questions?

Let me know if you have any requests, bugs, etc.

Email: john.lalor@nd.edu
