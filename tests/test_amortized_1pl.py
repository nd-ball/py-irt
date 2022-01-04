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
