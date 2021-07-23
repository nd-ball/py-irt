# preliminaries
from py_irt.config import IrtConfig
import unittest

# import model for testing
from py_irt.models.three_param_logistic import ThreeParamLog
from py_irt.training import IrtModelTrainer


class TestThreePL(unittest.TestCase):
    def test_training(self):
        config = IrtConfig(model_type="3pl", epochs=100)
        trainer = IrtModelTrainer(config=config, data_path="test_fixtures/minitest.jsonlines")
        trainer.train(device="cpu")
        trainer.save("/tmp/parameters.json")

    def test_priors(self):
        with self.assertRaises(NotImplementedError):
            m = ThreeParamLog("testing", "cpu", 100, 100, False)

    def test_device(self):
        with self.assertRaises(ValueError):
            m = ThreeParamLog("hierarchical", "zpu", 100, 100, False)

    def test_num_items(self):
        with self.assertRaises(ValueError):
            m = ThreeParamLog("hierarchical", "cpu", -100, 100, False)

    def test_num_subjects(self):
        with self.assertRaises(ValueError):
            m = ThreeParamLog("hierarchical", "cpu", 100, -100, False)
