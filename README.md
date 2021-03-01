[![Build Status](https://travis-ci.com/nd-ball/py-irt.svg?branch=master)](https://travis-ci.com/nd-ball/py-irt)
[![codecov.io](https://codecov.io/gh/nd-ball/py-irt/coverage.svg?branch=master)](https://codecov.io/gh/nd-ball/py-irt)

# py-irt
Bayesian IRT models in Python

## Overview

This repository includes code for fitting Item Response Theory (IRT) models using variational inference. 

At present, the one parameter logistic (1PL) model, aka Rasch model, is implemented. 
The user can specify whether vague or hierarchical priors are used.
Two- and three-parameter logistic models are in the pipeline and will be added when available.

## License

py-irt is licensed under the []MIT license](https://opensource.org/licenses/MIT).

## Installation

py-irt is now available on PyPi!

### Pre-reqs

1. Install [PyTorch](https://pytorch.org/get-started/locally/). 
2. Install [Pyro](https://pyro.ai/) 
3. Install py-irt: 


```shell
pip install py-irt 
```


## Citations

If you use this code, please consider citing the following paper:

```
@inproceedings{lalor2019emnlp,
  author    = {Lalor, John P and Wu, Hao and Yu, Hong},
  title     = {Learning Latent Parameters without Human Response Patterns: Item Response Theory with Artificial Crowds},
  year      = {2019},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
}
```

Implementation is based on the following paper:

```
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

