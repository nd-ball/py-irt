# preliminaries
import unittest

# import model for testing
from py_irt.models.three_param_logistic import ThreeParamLog
from py_irt.training import IrtModelTrainer


class TestThreePL(unittest.TestCase):
    def test_training(self):
        trainer = IrtModelTrainer(model_type="3pl", data_path="test_fixtures/minitest.jsonlines")
        trainer.train(iterations=100, device="cpu")
        trainer.save("/tmp/parameters.json")

    def test_priors(self):
        with self.assertRaises(NotImplementedError):
            m = FourParamLog("testing", "cpu", 100, 100, False)

    def test_device(self):
        with self.assertRaises(ValueError):
            m = FourParamLog("hierarchical", "zpu", 100, 100, False)

    def test_num_items(self):
        with self.assertRaises(ValueError):
            m = FourParamLog("hierarchical", "cpu", -100, 100, False)

    def test_num_subjects(self):
        with self.assertRaises(ValueError):
            m = FourParamLog("hierarchical", "cpu", 100, -100, False)
