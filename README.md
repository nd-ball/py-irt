# py-irt
Bayesian IRT models in Python

## Overview

This repository includes code for fitting Item Response Theory (IRT) models using variational inference. 

At present, the one parameter logistic (1PL) model, aka Rasch model, is implemented. 
The user can specify whether vague or hierarchical priors are used.
Two- and three-parameter logistic models are in the pipeline and will be added when available.


## Citations

If you use this code, please cite the following paper:

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

