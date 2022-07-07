# MIT License

# Copyright (c) 2019 John Lalor <john.lalor@nd.edu> and Pedro Rodriguez <me@pedro.ai>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
Functions to facilitate theta estimation
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import lognorm, norm
from scipy.special import expit


def theta_fn(difficulties, response_pattern):
    """Estimate theta for a given response pattern"""

    def fn(theta):
        theta = theta[0]
        probabilities = expit(theta - difficulties)
        log_likelihood = 0
        for i, rp in enumerate(response_pattern):
            if rp == 0:
                log_likelihood += np.log(1-probabilities[i])
            else:
                log_likelihood += np.log(probabilities[i])
        return -log_likelihood

    return fn


def calculate_theta(difficulties, response_pattern, num_obs=-1):
    """
    Given learned item difficulties and a model response pattern, estimate theta
    if num_obs > 0, then sample from the observed values for a computational speedup
    """

    if num_obs > 0:
        samples = np.random.choice(len(difficulties), num_obs)
        difficulties = [difficulties[s] for s in samples]
        response_pattern = [response_pattern[s] for s in samples]

    fn = theta_fn(difficulties, response_pattern)
    result = minimize(fn, [0.1], method="BFGS")
    return result["x"]


def calculate_diff_threshold(p_correct, theta):
    """
    Calculate the difficulty threshold where the probability correct given theta is equal to p_correct
    p_correct: the desired probability threshold
    theta: estimated model ability at current timestep
    """
    return np.log(1 / p_correct - 1) + theta
