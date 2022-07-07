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

# preliminaries
from py_irt.config import IrtConfig
from py_irt.training import IrtModelTrainer
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

    def test_training(self):
        config = IrtConfig(model_type="2pl", epochs=100, priors="hierarchical")
        trainer = IrtModelTrainer(config=config, data_path="test_fixtures/minitest.jsonlines")
        trainer.train(device="cpu")
        trainer.save("/tmp/parameters.json")

    def test_priors(self):
        with self.assertRaises(ValueError):
            m = TwoParamLog(
                priors="testing", device="cpu", num_items=100, num_subjects=100, verbose=False
            )

    def test_device(self):
        with self.assertRaises(ValueError):
            m = TwoParamLog(
                priors="vague", device="zpu", num_items=100, num_subjects=100, verbose=False
            )

    def test_num_items(self):
        with self.assertRaises(ValueError):
            m = TwoParamLog(
                priors="vague", device="cpu", num_items=-100, num_subjects=100, verbose=False
            )

    def test_num_subjects(self):
        with self.assertRaises(ValueError):
            m = TwoParamLog(
                priors="vague", device="cpu", num_items=100, num_subjects=-100, verbose=False
            )
