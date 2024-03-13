# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [0.6.0] - 2024-03-13

- Add graphviz for model viz
- Update to torch 2
- Make model more configurable
- Add a model for the tutorial

## [0.5.0] - 2024-01-05

- Update most dependencies with major version updates to pandas (1->2), pydantic (1->2)
- Drop support for Python 3.8

## [0.4.13] - 2023-12-21

- Address issue #54 regarding 3pl predictions. 

## [0.4.12] - 2023-12-21

- Write item IDs to disk for train_and_evaluate. 

## [0.4.11] - 2023-12-21

- Fix an issue with cli.evaluate.

## [0.4.10] - 2023-04-12

- Fix an issue with codecov, and also allow for Python 3.10 and 3.11.

## [0.4.9] - 2023-03-30

- Update dependencies so that we don't use torch 2.0.0 because of a breaking change in pyro.

## [0.4.8] - 2023-01-18

- Implement ordered sets to control subject and item creation
- Implement CLI argument to set a random seed

## [0.4.6] - 2022-09-26

- Fix bug in 3PL model

## [0.4.0] - 2022-02-02

- Implement 3PL and amortized 1PL models

## [0.3.4] - 2021-09-16

- Bugfix for CLI

## [0.3.3] - 2021-07-26

- Multidim model added

## [0.3.0] - 2021-07-14

- Add command line interfaces for evaluating a model and also training and evaluating together

## [0.2.1] - 2021-06-30

- Add command line interface for training model 
- Add 4PL model 
- Refactoring 

## [0.1.1] - 2021-03-17
 
### Added

- Added functions for estimating theta.
- Set up automatic builds, documentation, and tests.
    
### Changed
 
### Fixed
 
