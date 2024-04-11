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

from py_irt.training import Dataset
import pandas as pd
import numpy as np
from pytest import raises


def test_jsonlines_load():
    dataset = Dataset.from_jsonlines("test_fixtures/minitest.jsonlines")
    assert len(dataset.item_ids) == 4
    assert len(dataset.subject_ids) == 4
    assert "pedro" in dataset.subject_ids
    assert "pinguino" in dataset.subject_ids
    assert "ken" in dataset.subject_ids
    assert "burt" in dataset.subject_ids
    assert "q1" in dataset.item_ids
    assert "q2" in dataset.item_ids
    assert "q3" in dataset.item_ids
    assert "q4" in dataset.item_ids

def test_from_pandas():
    df = pd.DataFrame({
        'subject_id': ["joe", "sarah", "juan", "julia"],
        'item_1': [0, 1, 1, 1],
        'item_2': [0, 1, 0, 1],
        'item_3': [1, 0, 1, 0],
    })

    subject_column = 'subject_id'
    item_columns = ['item_1', 'item_2', 'item_3']
    dataset = Dataset.from_pandas(df, subject_column, item_columns)

    assert set(dataset.item_ids) == {'item_1', 'item_2', 'item_3'}
    assert set(dataset.subject_ids) == {'joe', 'sarah', 'juan', 'julia'}
    assert dataset.ix_to_item_id[dataset.item_id_to_ix['item_1']] == 'item_1'
    assert dataset.ix_to_subject_id[dataset.subject_id_to_ix['joe']] == 'joe'
    assert len(dataset.observations) == 12

def test_to_pandas():
    df = pd.DataFrame({
        'subject_id': ["joe", "sarah", "juan", "julia"],
        'item_1': [0, 1, 2, 3],
        'item_2': [4, 5, 6, 7],
        'item_3': [8, 9, 10, 11],
    })
    dataset = Dataset.from_pandas(df, 'subject_id', ['item_1', 'item_2', 'item_3'])
    long = dataset.to_pandas(wide=False)
    assert set(long.columns) == {'subject', 'item', 'response', 'item_ix', 'subject_ix'}
    assert long[long.subject == 'joe'].response.values.tolist() == [0, 4, 8]

    wide = dataset.to_pandas(wide=True)
    assert set(wide.columns) == {'subject', 'item_1', 'item_2', 'item_3'}
    assert wide[wide.subject == 'joe'].item_1.values.tolist() == [0]

def test_from_pandas_with_missing():
    df = pd.DataFrame({
        'subject_id': ["joe", "sarah", "juan", "julia"],
        'item_1': [0, np.nan, 1, 1],
        'item_2': [0, 1, 0, 1],
        'item_3': [1, 0, np.nan, 0],
    })

    dataset = Dataset.from_pandas(df, 'subject_id', ['item_1', 'item_2', 'item_3'])
    long = dataset.to_pandas(wide=False)
    assert long[(long.subject == "sarah") & (long.item == "item_1")].values.size == 0
    wide = dataset.to_pandas()
    assert wide[wide.subject == "sarah"].item_1.isna().all()

def test_from_pandas_defaults():
    df = pd.DataFrame({
        'item_1': [3, np.nan, 1, 1],
        'item_2': [0, 1, 2, 3],
        'item_3': [1, 0, np.nan, 0],
    })
    ds = Dataset.from_pandas(df).to_pandas()
    assert set(ds.columns) == {"subject", "item_1", "item_2", "item_3"}
    assert set(ds.subject.values.tolist()) == {'0', '1', '2', '3'}
    assert ds[ds.index == 0].item_1.tolist() == [3]
    assert set(Dataset.from_pandas(df[["item_1", "item_2", "item_3"]], subject_column="item_2").to_pandas().columns) == {"subject", "item_1", "item_3"}

def test_to_pandas_errors():
    df = pd.DataFrame({
        'subject_id': ["joe", "sarah", "juan", "julia"],
        'item_1': [0, np.nan, 1, 1],
        'item_2': [0, 1, 0, 1],
        'item_3': [1, 0, np.nan, 0],
    })

    with raises(ValueError):
        Dataset.from_pandas(df, subject_column=0)
    
    with raises(KeyError):
        Dataset.from_pandas(df, subject_column="foo")
    
    with raises(ValueError):
        Dataset.from_pandas(df, subject_column="subject_id", item_columns=["subject_id", "item_1"])
    
    with raises(ValueError):
        Dataset.from_pandas(df, subject_column="item_1") # it has nans
    
    with raises(ValueError):
        Dataset.from_pandas(df, subject_column='item_2') # it has duplicates
