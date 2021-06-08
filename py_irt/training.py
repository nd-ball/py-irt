import typer
import torch
from pathlib import Path
from pydantic import BaseModel
from py_irt.models import one_param_logistic, two_param_logistic, four_param_logistic
from pedroai.io import read_jsonlines


training_app = typer.Typer()


class Dataset(BaseModel):
    pass


IRT_MODELS = {"1pl": one_param_logistic, "2pl": two_param_logistic, "4pl": four_param_logistic}


class IrtModelTrainer:
    def __init__(self, *, data_path: Path, model_type: str, python_data=None) -> None:
        self._data_path = data_path
        if model_type not in IRT_MODELS:
            raise ValueError(f"{model_type} not {IRT_MODELS.keys()}")
        self._model_type = model_type
        self._priors = None
        self._device = None
        self._iterations = None
        if python_data is None:
            self._all_subjects = read_jsonlines(data_path)
        else:
            self._all_subjects = python_data

        self.item_ids = set()
        self.subject_ids = set()
        self.item_id_to_ix = {}
        self.ix_to_item_id = {}
        self.subject_id_to_ix = {}
        self.ix_to_subject_id = {}

    def train(self, *, iterations: int = 1000, priors="hierarchical", device: str = "cpu") -> None:
        self._device = device
        self._priors = priors
        self._iterations = iterations
        device = torch.device(device)
        self._pyro_model = IRT_MODELS[self.model](
            priors=priors,
            device=device,
            num_items=len(self.ix_to_example_id),
            num_subjects=self.n_subjects,
        )

        self._pyro_model.fit(
            torch.tensor(self.student_to_obs, dtype=torch.long, device=device),
            torch.tensor(self.question_to_obs, dtype=torch.long, device=device),
            torch.tensor(self.observations, dtype=torch.float, device=device),
            iterations,
        )

    def export(self):
        results = self._pyro_model.export()
        results["irt_model"] = self._model_type
        results["example_ids"] = self.ix_to_example_id
        results["model_ids"] = self.ix_to_model_id
        return results


@training_app.command()
def train():
    pass
