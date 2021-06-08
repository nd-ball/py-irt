# preliminaries
import numpy as np
import pyro
import torch
import torch.nn as nn

import unittest
from scipy.special import expit

# import model for testing
from py_irt.models.two_param_logistic import TwoParamLog


class TestTwoPL(unittest.TestCase):
    def test_fitting(self):
        device = torch.device("cpu")
        models = []
        items = []
        responses = []
        num_subjects = 2000
        num_items = 100
        real_theta = np.random.normal(size=[num_subjects])
        real_diff = np.random.normal(size=[num_items])
        real_slope = np.random.normal(size=[num_items])
        obs = []
        for i in range(len(real_theta)):
            for j in range(len(real_diff)):
                y = np.random.binomial(1, expit(real_slope[j] * (real_theta[i] - real_diff[j])))
                models.append(i)
                items.append(j)
                responses.append(y)
        num_subjects = len(set(models))
        num_items = len(set(items))
        self.models = torch.tensor(models, dtype=torch.long, device=device)
        self.items = torch.tensor(items, dtype=torch.long, device=device)
        self.responses = torch.tensor(responses, dtype=torch.float, device=device)

    def test_priors(self):
        with self.assertRaises(ValueError):
            m = TwoParamLog("testing", "cpu", 100, 100, False)

    def test_device(self):
        with self.assertRaises(ValueError):
            m = TwoParamLog("vague", "zpu", 100, 100, False)

    def test_num_items(self):
        with self.assertRaises(ValueError):
            m = TwoParamLog("vague", "cpu", -100, 100, False)

    def test_num_subjects(self):
        with self.assertRaises(ValueError):
            m = TwoParamLog("vague", "cpu", 100, -100, False)
