from typing import Dict, Set, Optional, List, Union
from pathlib import Path
import typer
import torch
from pydantic import BaseModel
from py_irt.models import one_param_logistic, two_param_logistic, four_param_logistic
from pedroai.io import read_jsonlines, safe_file, write_json


training_app = typer.Typer()


class Dataset(BaseModel):
    item_ids: Set[str]
    subject_ids: Set[str]
    item_id_to_ix: Dict[str, int]
    ix_to_item_id: Dict[int, str]
    subject_id_to_ix: Dict[str, int]
    ix_to_subject_id: Dict[int, str]
    # observation_subjects and observation_items refers to indices
    observation_subjects: List[int]
    observation_items: List[int]
    # Actual response value, usually an integer
    observations: List[float]

    @classmethod
    def from_jsonlines(cls, data_path: Path):
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

        observation_subjects = []
        observation_items = []
        observations = []
        for idx, line in enumerate(input_data):
            subject_id = line["subject_id"]
            for item_id, response in line["responses"].items():
                observations.append(response)
                observation_subjects.append(subject_id_to_ix[subject_id])
                observation_items.append(item_id_to_ix[item_id])

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
        )


IRT_MODELS = {
    "1pl": one_param_logistic.OneParamLog,
    "2pl": two_param_logistic.TwoParamLog,
    "4pl": four_param_logistic.FourParamLog,
}


class IrtModelTrainer:
    def __init__(
        self, *, data_path: Path, model_type: str, dataset: Optional[Dataset] = None
    ) -> None:
        self._data_path = data_path
        if model_type not in IRT_MODELS:
            raise ValueError(f"{model_type} not {IRT_MODELS.keys()}")
        self._model_type = model_type
        self._priors = None
        self._device = None
        self._iterations = None
        if dataset is None:
            self._dataset = Dataset.from_jsonlines(data_path)
        else:
            self._dataset = dataset

    def train(self, *, iterations: int = 1000, priors="hierarchical", device: str = "cpu") -> None:
        self._device = device
        self._priors = priors
        self._iterations = iterations
        self._pyro_model = IRT_MODELS[self._model_type](
            priors=priors,
            device=device,
            num_items=len(self._dataset.ix_to_item_id),
            num_subjects=len(self._dataset.ix_to_subject_id),
        )
        device = torch.device(device)

        self._pyro_model.fit(
            torch.tensor(self._dataset.observation_subjects, dtype=torch.long, device=device),
            torch.tensor(self._dataset.observation_items, dtype=torch.long, device=device),
            torch.tensor(self._dataset.observations, dtype=torch.float, device=device),
            iterations,
        )

    def export(self):
        results = self._pyro_model.export()
        results["irt_model"] = self._model_type
        results["item_ids"] = self._dataset.ix_to_item_id
        results["subject_ids"] = self._dataset.ix_to_subject_id
        return results

    def save(self, output_path: Union[str, Path]):
        write_json(safe_file(output_path), self.export())
