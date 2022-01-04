import unittest

from py_irt.config import IrtConfig
from py_irt.training import IrtModelTrainer


class TestMultidim2PL(unittest.TestCase):
    def test_training(self):
        config = IrtConfig(model_type="multidim_2pl", epochs=100, dims=3)
        trainer = IrtModelTrainer(config=config, data_path="test_fixtures/minitest.jsonlines")
        trainer.train(device="cpu")
        trainer.save("/tmp/parameters.json")
