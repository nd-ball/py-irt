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
