import unittest

from py_irt.config import IrtConfig
from py_irt.training import IrtModelTrainer


class TestAmortized1PL(unittest.TestCase):
    def test_training(self):
        config = IrtConfig(model_type="amortized_1pl", epochs=100, dims=3)
        trainer = IrtModelTrainer(config=config, data_path="test_fixtures/minitest.amortized.jsonlines")
        trainer.train(device="cpu")
        trainer.save("/tmp/parameters.json")
