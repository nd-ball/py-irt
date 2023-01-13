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
    item_id_to_ix: Dict[str, int]
    ix_to_item_id: Dict[int, str]
    subject_id_to_ix: Dict[str, int]
    ix_to_subject_id: Dict[int, str]
    # observation_subjects and observation_items refers to indices
    observation_subjects: List[int]
    observation_items: List
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
