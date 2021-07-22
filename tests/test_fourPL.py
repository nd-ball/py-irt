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
