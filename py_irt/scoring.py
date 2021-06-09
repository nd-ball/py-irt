"""
Functions to facilitate theta estimation
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import lognorm, norm
from scipy.special import expit


def theta_fn(difficulties, student_prior, response_pattern):
    """Estimate theta for a given response pattern"""

    def fn(theta):
        theta = theta[0]
        probabilities = expit(theta - difficulties)
        # print(probabilities)
        log_likelihood = student_prior.logpdf(theta)
        for i, rp in enumerate(response_pattern):
            log_likelihood += np.log1p((2 * probabilities[i] - 1) * rp)
        # print(log_likelihood)
        return -log_likelihood

    return fn


def calculate_theta(difficulties, response_pattern, num_obs=-1):
    """
    Given learned item difficulties and a model response pattern, estimate theta
    if num_obs > 0, then sample from the observed values for a computational speedup
    """

    student_prior = norm(loc=0.0, scale=1.0)
    if num_obs > 0:
        samples = np.random.choice(len(difficulties), num_obs)
        difficulties = [difficulties[s] for s in samples]
        response_pattern = [response_pattern[s] for s in samples]

    fn = theta_fn(difficulties, student_prior, response_pattern)
    result = minimize(fn, [0.1], method="Nelder-Mead")
    return result["x"]


def calculate_diff_threshold(p_correct, theta):
    """
    Calculate the difficulty threshold where the probability correct given theta is equal to p_correct
    p_correct: the desired probability threshold
    theta: estimated model ability at current timestep
    """
    return np.log(1 / p_correct - 1) + theta
