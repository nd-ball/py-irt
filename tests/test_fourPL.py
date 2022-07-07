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
import unittest

# import model for testing
from py_irt.models.four_param_logistic import FourParamLog
from py_irt.training import IrtModelTrainer


class TestFourPL(unittest.TestCase):
    def test_training(self):
        config = IrtConfig(model_type="4pl", epochs=100)
        trainer = IrtModelTrainer(config=config, data_path="test_fixtures/minitest.jsonlines")
        trainer.train(device="cpu")
        trainer.save("/tmp/parameters.json")

    def test_device(self):
        with self.assertRaises(ValueError):
            m = FourParamLog(device="zpu", num_items=100, num_subjects=100, verbose=False)

    def test_num_items(self):
        with self.assertRaises(ValueError):
            m = FourParamLog(device="cpu", num_items=-100, num_subjects=100, verbose=False)

    def test_num_subjects(self):
        with self.assertRaises(ValueError):
            m = FourParamLog(device="cpu", num_items=100, num_subjects=-100, verbose=False)
