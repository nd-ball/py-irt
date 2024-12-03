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

from typing import Set, Dict, List, Union
from pathlib import Path
from pydantic import BaseModel
from py_irt.io import read_jsonlines
from sklearn.feature_extraction.text import CountVectorizer
from ordered_set import OrderedSet
from rich.console import Console
import pandas as pd

console = Console()

class ItemAccuracy(BaseModel):
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self):
        return self.correct / max(1, self.total)


class Dataset(BaseModel):
    item_ids: Union[Set[str], OrderedSet]
    subject_ids: Union[Set[str], OrderedSet]
    item_id_to_ix: Dict[str, int] # encoding values for each item
    ix_to_item_id: Dict[int, str]
    subject_id_to_ix: Dict[str, int] # encoding values for each subject
    ix_to_subject_id: Dict[int, str]

    # observation_subjects and observation_items refers to indices
    observation_subjects: List[int] # subjects encoded as integers
    observation_items: List # items encoded as integers

    # Actual response value, usually an integer
    observations: List[float]

    # should this example be included in training? 
    training_example: List[bool]

    class Config:
        arbitrary_types_allowed = True

    def get_item_accuracies(self) -> Dict[str, ItemAccuracy]:
        item_accuracies = {}
        for ix, response in enumerate(self.observations):
            item_ix = self.observation_items[ix]
            item_id = self.ix_to_item_id[item_ix]
            if item_id not in item_accuracies:
                item_accuracies[item_id] = ItemAccuracy()

            item_accuracies[item_id].correct += response
            item_accuracies[item_id].total += 1

        return item_accuracies

    @classmethod
    def from_jsonlines(cls, data_path: Path, train_items: dict = None, amortized: bool = False):
        """Parse IRT dataset from jsonlines, formatted in the following way:
        * The dataset is in jsonlines format, each line representing the responses of a subject
        * Each row looks like this:
        {"subject_id": "<subject_id>", "responses": {"<item_id>": <response>}}
        * Where <subject_id> is a string, <item_id> is a string, and <response> is a number (usually integer)
        """
        item_ids = OrderedSet()
        subject_ids = OrderedSet()
        item_id_to_ix = {}
        ix_to_item_id = {}
        subject_id_to_ix = {}
        ix_to_subject_id = {}
        input_data = read_jsonlines(data_path)
        for line in input_data:
            subject_id = line["subject_id"]
            subject_ids.add(subject_id)
            responses = line["responses"]
            for item_id in responses.keys():
                item_ids.add(item_id)

        for idx, item_id in enumerate(item_ids):
            item_id_to_ix[item_id] = idx
            ix_to_item_id[idx] = item_id

        for idx, subject_id in enumerate(subject_ids):
            subject_id_to_ix[subject_id] = idx
            ix_to_subject_id[idx] = subject_id
        
        if amortized:
            vectorizer = CountVectorizer(max_df=0.5, min_df=20, stop_words='english')
            vectorizer.fit(item_ids)

        
        observation_subjects = []
        observation_items = []
        observations = []
        training_example = []
        console.log(f'amortized: {amortized}')
        for idx, line in enumerate(input_data):
            subject_id = line["subject_id"]
            for item_id, response in line["responses"].items():
                observations.append(response)
                observation_subjects.append(subject_id_to_ix[subject_id])
                if not amortized:
                    observation_items.append(item_id_to_ix[item_id])
                else:
                    observation_items.append(vectorizer.transform([item_id]).todense().tolist()[0])
                if train_items is not None:
                    training_example.append(train_items[subject_id][item_id])
                else:
                    training_example.append(True)

        return cls(
            item_ids=item_ids,
            subject_ids=subject_ids,
            item_id_to_ix=item_id_to_ix,
            ix_to_item_id=ix_to_item_id,
            subject_id_to_ix=subject_id_to_ix,
            ix_to_subject_id=ix_to_subject_id,
            observation_subjects=observation_subjects,
            observation_items=observation_items,
            observations=observations,
            training_example=training_example,
        )

    @classmethod
    def from_pandas(cls, df, subject_column=None, item_columns=None):
        """Build a Dataset object from a pandas DataFrame

        Rows represent subjects. Columns represent items. Values represent responses. Nan values are treated as missing data.

        E.g.
        ```python
        df = pd.DataFrame({
            'user_id': ["joe", "sarah", "juan", "julia"],
            'item_1': [0, 1, 1, 1],
            'item_2': [0, 1, 0, 1],
            'item_3': [1, 0, 1, 0],
        })
        subject_column = 'user_id'
        item_columns = ['item_1', 'item_2', 'item_3']
        ```

        Args:
            df (pd.DataFrame): A DataFrame containing the data
            subject_column (str, optional): The name of the column containing the subject ids, defaults to using the index
            item_columns (list of str, optional): The names of the columns containing the item ids, defaults to every column
        Returns:
            Dataset: The dataset object
        """
        if subject_column is not None:
            if item_columns is not None and subject_column in item_columns:
                raise ValueError("subject_column cannot be in item_columns")
            if not isinstance(subject_column, str):
                raise ValueError("subject_column must be a string if provided")
        if item_columns is not None:
            if isinstance(item_columns, str):
                raise ValueError("item_columns must be an iterable of strings if provided")
            try:
                item_columns = list(item_columns)
            except TypeError:
                raise ValueError("item_columns must be an iterable of strings if provided")
        
        # default value for subject columns is the index
        if subject_column is None:
            subject_column = "subject_name"
            i = 0
            while subject_column in df.columns:
                subject_column = f"subject_name: {i}"
                i += 1
            
            df[subject_column] = df.index.astype(str)

        if df[subject_column].isna().any():
            raise ValueError("subject column cannot contain nan")
        if df[subject_column].unique().size < df[subject_column].values.size:
            raise ValueError("subject column cannot contain duplicates")
        if df[subject_column].values.dtype != str:
            df[subject_column] = df[subject_column].astype(str)
        
        # default value for item columns is all columns except the subject column
        if item_columns is None:
            item_columns = [c for c in df.columns if c != subject_column]
        df[item_columns] = df[item_columns].astype(float)

        melted = pd.melt(
            df[[subject_column] + item_columns],
            id_vars=[subject_column],
            value_vars=item_columns,
            var_name="item_name",
            value_name="outcome"
            ).rename(columns={subject_column: "subject_name"})
        
        # na values code for unkown data that should not be included in training
        melted = melted.dropna(axis=0)

        item_ids = pd.DataFrame({
            "item_name": item_columns,
            "item_id": range(len(item_columns))
        })
        subject_ids = pd.DataFrame({
            "subject_name": df[subject_column].unique(),
            "subject_id": range(len(df[subject_column].unique()))
        })
        merged = pd.merge(
            pd.merge(melted, item_ids, how="left", on="item_name"),
            subject_ids, how="left", on="subject_name"
            )

        return cls(
            item_ids = OrderedSet([str(x) for x in merged.item_name.values]),
            subject_ids = OrderedSet([str(x) for x in merged.subject_name.values]),
            observation_subjects = list(merged.subject_id.values),
            observation_items = list(merged.item_id.values),
            observations = list(merged.outcome.values),
            training_example = [True for _ in range(merged.shape[0])],
            item_id_to_ix = dict(zip(item_ids.item_name, item_ids.item_id)),
            ix_to_item_id = dict(zip(item_ids.item_id, item_ids.item_name)),
            subject_id_to_ix = dict(zip(subject_ids.subject_name, subject_ids.subject_id)),
            ix_to_subject_id = dict(zip(subject_ids.subject_id, subject_ids.subject_name))
        )
    
    def to_pandas(self, wide=True):
        """Convert the dataset to a pandas DataFrame

        If returned in long format, the columns will be "subject", "item", "subject_ix", "item_ix", "response".
        If returned in wide format, the columns will be "subject" and the names of the items.

        Args:
            wide (bool, optional): Whether to return the dataset in wide format (default) or long format. Defaults to True.

        Returns:
            pd.DataFrame: The dataset as a DataFrame
        """
        subject_list = list(zip(*[[k, v] for k, v in self.ix_to_subject_id.items()]))
        item_list = list(zip(*[[k, v] for k, v in self.ix_to_item_id.items()]))
        subjects = pd.DataFrame({"subject": subject_list[1]}, index=subject_list[0])
        items = pd.DataFrame({"item": item_list[1]}, index=item_list[0])

        long = pd.DataFrame({
            "subject_ix": self.observation_subjects,
            "item_ix": self.observation_items,
            "response": self.observations
        }).join(subjects, on="subject_ix").join(items, on="item_ix")
        long = long[["subject", "subject_ix", "item", "item_ix", "response"]]

        if not wide:
            return long
        return subjects.join(long.pivot(index="subject_ix", columns="item", values="response"))
