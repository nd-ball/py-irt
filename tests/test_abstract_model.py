import pandas as pd
from py_irt.dataset import Dataset
from py_irt.models import *


def test_train_models():
    df = pd.DataFrame({
        'subject_id': ["joe", "sarah", "juan", "julia"],
        'item_1': [0, 1, 1, 1],
        'item_2': [0, 1, 0, 1],
        'item_3': [1, 0, 1, 0],
    })

    data = Dataset.from_pandas(df, subject_column="subject_id", item_columns=["item_1", "item_2", "item_3"])

    for model in [OneParamLog, TwoParamLog, ThreeParamLog, Multidim2PL]:
        trainer = model.train(data, epochs=10)
        trainer.irt_model.export()
