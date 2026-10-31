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


import unittest

import numpy as np
from scipy.special import expit

from py_irt.scoring import calculate_theta

NUM_ITEMS = 1000
TRUE_THETAS = [-2.0, -0.5, 0.0, 0.5, 2.0]
TOLERANCE = 0.35


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.difficulties = self.rng.normal(size=NUM_ITEMS)
        self.discriminations = np.exp(self.rng.normal(scale=0.3, size=NUM_ITEMS))
        self.guessing = self.rng.uniform(0.0, 0.3, size=NUM_ITEMS)

    def simulate(self, theta, discriminations=None, guessing=None):
        if discriminations is None:
            discriminations = np.ones(NUM_ITEMS)
        if guessing is None:
            guessing = np.zeros(NUM_ITEMS)
        probs = guessing + (1 - guessing) * expit(
            discriminations * (theta - self.difficulties)
        )
        return self.rng.binomial(1, probs)

    def test_1pl_theta_recovery(self):
        for true_theta in TRUE_THETAS:
            responses = self.simulate(true_theta)
            est = calculate_theta(self.difficulties, responses)[0]
            self.assertAlmostEqual(est, true_theta, delta=TOLERANCE)

    def test_2pl_theta_recovery(self):
        for true_theta in TRUE_THETAS:
            responses = self.simulate(true_theta, discriminations=self.discriminations)
            est = calculate_theta(
                self.difficulties, responses, discriminations=self.discriminations
            )[0]
            self.assertAlmostEqual(est, true_theta, delta=TOLERANCE)

    def test_3pl_theta_recovery(self):
        for true_theta in TRUE_THETAS:
            responses = self.simulate(
                true_theta, discriminations=self.discriminations, guessing=self.guessing
            )
            est = calculate_theta(
                self.difficulties,
                responses,
                discriminations=self.discriminations,
                guessing=self.guessing,
            )[0]
            self.assertAlmostEqual(est, true_theta, delta=TOLERANCE)

    def test_discrimination_changes_estimate(self):
        # with all discriminations = 1, the 2PL reduces to the 1PL
        responses = self.simulate(0.5)
        est_1pl = calculate_theta(self.difficulties, responses)[0]
        est_2pl = calculate_theta(
            self.difficulties, responses, discriminations=np.ones(NUM_ITEMS)
        )[0]
        self.assertAlmostEqual(est_1pl, est_2pl, places=5)

    def test_num_obs_subsampling(self):
        responses = self.simulate(0.0, self.discriminations, self.guessing)
        est = calculate_theta(
            list(self.difficulties),
            list(responses),
            num_obs=500,
            discriminations=list(self.discriminations),
            guessing=list(self.guessing),
        )[0]
        self.assertTrue(np.isfinite(est))

    def test_extreme_response_patterns(self):
        # all-correct and all-incorrect patterns should not produce nan/inf
        all_correct = np.ones(NUM_ITEMS)
        all_wrong = np.zeros(NUM_ITEMS)
        for responses in (all_correct, all_wrong):
            est = calculate_theta(
                self.difficulties,
                responses,
                discriminations=self.discriminations,
                guessing=self.guessing,
            )[0]
            self.assertTrue(np.isfinite(est))


if __name__ == "__main__":
    unittest.main()
