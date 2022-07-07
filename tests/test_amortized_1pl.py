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

from py_irt.config import IrtConfig
from py_irt.training import IrtModelTrainer

# import model for testing
from py_irt.models.amortized_1pl import Amortized1PL 



class TestAmortized1PL(unittest.TestCase):        
    def test_training(self):
        config = IrtConfig(model_type="amortized_1pl", epochs=2)
        trainer = IrtModelTrainer(config=config, data_path="test_fixtures/minitest.amortized.jsonlines")
        trainer.train(device="cpu")
        trainer.save("/tmp/parameters.json")

    def test_device(self):
        with self.assertRaises(ValueError):
            m = Amortized1PL(
                priors="vague", 
                device="zpu", 
                num_items=100, 
                num_subjects=100, 
                verbose=False,
                vocab_size=100,
                dropout=0.5,
                hidden=100
            )

    def test_num_items(self):
        with self.assertRaises(ValueError):
            m = Amortized1PL(
                priors="vague", 
                device="cpu", 
                num_items=-100, 
                num_subjects=100, 
                verbose=False,
                vocab_size=100,
                dropout=0.5,
                hidden=100
            )

    def test_num_subjects(self):
        with self.assertRaises(ValueError):
            m = Amortized1PL(
                priors="vague", 
                device="cpu", 
                num_items=100, 
                num_subjects=-100, 
                verbose=False,
                vocab_size=100,
                dropout=0.5,
                hidden=100
            )
