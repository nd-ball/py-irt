from typing import Set, Dict, List
from pathlib import Path
from pydantic import BaseModel
from py_irt.io import read_jsonlines
from sklearn.feature_extraction.text import CountVectorizer

class ItemAccuracy(BaseModel):
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self):
        return self.correct / max(1, self.total)


class Dataset(BaseModel):
    item_ids: Set[str]
    subject_ids: Set[str]
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
        item_ids = set()
        subject_ids = set()
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
        print(amortized)
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
