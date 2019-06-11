# py-irt
Bayesian IRT models in Python

## Overview

This repository includes code for fitting Item Response Theory (IRT) models using variational inference. 

At present, the one parameter logistic (1PL) model, aka Rasch model, is implemented. 
The user can specify whether vague or hierarchical priors are used.
Two- and three-parameter logistic models are in the pipeline and will be added when available.


## Citation

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

If you use this code, please also cite the following extended abstract:

```
@inproceedings{lalor2019sivl,
  author    = {Lalor, John P. and Wu, Hao and Yu, Hong},
  title     = {{Learning Latent Parameters without Human Response Patterns: Item Response Theory with Artificial Crowds }},
  year      = {2019},
  maintitle={Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  booktitle = {Workshop on Shortcomings in Vision and Language (SiVL), Extended Abstracts},
}
```

