[![Build Status](https://travis-ci.org/automl/pybnn.svg?branch=master)](https://travis-ci.org/automl/pybnn)
[![codecov](https://codecov.io/gh/automl/pybnn/branch/master/graph/badge.svg)](https://codecov.io/gh/automl/pybnn)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/automl/pybnn/blob/master/LICENSE)

# pybnn
Bayesian neural networks for Bayesian optimization based on pytorch.

 It contains implementations for:
 - [Scalable Bayesian Optimization Using Deep Neural Networks](https://arxiv.org/pdf/1502.05700.pdf) (DNGO which is BLR)
 - [Bayesian optimization with robust Bayesian neural networks](https://ml.informatik.uni-freiburg.de/papers/16-NIPS-BOHamiANN.pdf) (BOHAMIANN which is HMC-based)
 - [Dropout as a Bayesian approximation: Representing model uncertainty in deep learning](http://proceedings.mlr.press/v48/gal16.pdf)(MCDROP)
 - Our Loss-Callibrated BNN Method


# Installation

    git clone https://github.com/automl/pybnn.git
    cd pybnn
    python setup.py install

