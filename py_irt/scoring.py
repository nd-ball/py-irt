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


def theta_fn(difficulties, response_pattern, discriminations=None, guessing=None):
    """
    Build the negative log-likelihood function of theta for a response pattern.

    Supports the 1PL, 2PL, and 3PL models:
        p(correct) = c + (1 - c) * sigmoid(a * (theta - b))
    where b = difficulty, a = discrimination (1PL: a = 1),
    and c = guessing / lower asymptote (1PL, 2PL: c = 0).

    difficulties: item difficulties (b)
    response_pattern: dichotomous responses (0/1), one per item
    discriminations: item discriminations (a); if None, all 1 (1PL)
    guessing: item guessing parameters (c); if None, all 0 (1PL/2PL)
    """
    difficulties = np.asarray(difficulties, dtype=float)
    responses = np.asarray(response_pattern, dtype=float)
    if discriminations is None:
        discriminations = np.ones_like(difficulties)
    else:
        discriminations = np.asarray(discriminations, dtype=float)
    if guessing is None:
        guessing = np.zeros_like(difficulties)
    else:
        guessing = np.asarray(guessing, dtype=float)

    def fn(theta):
        theta = theta[0]
        p_star = expit(discriminations * (theta - difficulties))
        probabilities = guessing + (1 - guessing) * p_star
        # clip to avoid log(0) at extreme theta values
        probabilities = np.clip(probabilities, 1e-10, 1 - 1e-10)
        log_likelihood = np.sum(
            responses * np.log(probabilities) + (1 - responses) * np.log(1 - probabilities)
        )
        return -log_likelihood

    return fn


def calculate_theta(
    difficulties, response_pattern, num_obs=-1, discriminations=None, guessing=None
):
    """
    Given learned item parameters and a model response pattern, estimate theta.
    If num_obs > 0, then sample from the observed values for a computational speedup.

    Pass only difficulties for a 1PL model, add discriminations for 2PL,
    and add guessing parameters for 3PL.
    """

    if num_obs > 0:
        samples = np.random.choice(len(difficulties), num_obs)
        difficulties = [difficulties[s] for s in samples]
        response_pattern = [response_pattern[s] for s in samples]
        if discriminations is not None:
            discriminations = [discriminations[s] for s in samples]
        if guessing is not None:
            guessing = [guessing[s] for s in samples]

    fn = theta_fn(difficulties, response_pattern, discriminations, guessing)
    result = minimize(fn, [0.1], method="BFGS")
    return result["x"]


def calculate_diff_threshold(p_correct, theta):
    """
    Calculate the difficulty threshold where the probability correct given theta is equal to p_correct
    p_correct: the desired probability threshold
    theta: estimated model ability at current timestep
    """
    return np.log(1 / p_correct - 1) + theta
