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


from py_irt.initializers import DifficultySignInitializer
from py_irt.dataset import Dataset
from py_irt.training import IrtModelTrainer
from py_irt.config import IrtConfig


def test_parsing():
    config = IrtConfig(model_type="4pl", initializers=[])
    # This loads the initializer, so is a fine test
    trainer = IrtModelTrainer(data_path="test_fixtures/minitest.jsonlines", config=config)
    assert len(trainer._initializers) == 0

    config = IrtConfig(model_type="4pl", initializers=["difficulty_sign"])
    trainer = IrtModelTrainer(data_path="test_fixtures/minitest.jsonlines", config=config)
    assert len(trainer._initializers) == 1
    assert isinstance(trainer._initializers[0], DifficultySignInitializer)

    config = IrtConfig(
        model_type="4pl", initializers=[{"name": "difficulty_sign", "magnitude": 5.0}]
    )
    trainer = IrtModelTrainer(data_path="test_fixtures/minitest.jsonlines", config=config)
    assert len(trainer._initializers) == 1
    assert isinstance(trainer._initializers[0], DifficultySignInitializer)
    assert trainer._initializers[0]._magnitude == 5.0
