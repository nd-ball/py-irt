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

from typing import Optional, Union, Dict
from pathlib import Path
import contextlib

import typer
import torch

from pyro.infer import SVI, Trace_ELBO
import pyro

from rich.console import Console
from rich.live import Live
from rich.table import Table

from sklearn.feature_extraction.text import CountVectorizer


# This import is necessary to have @register run
# pylint: disable=unused-import
import py_irt.models

from py_irt.models import abstract_model
from py_irt.io import safe_file, write_json
from py_irt.dataset import Dataset
from py_irt.initializers import INITIALIZERS, IrtInitializer
from py_irt.config import IrtConfig
from py_irt.models.abstract_model import IrtModel


training_app = typer.Typer()
console = Console()


class IrtModelTrainer:
    def __init__(
        self,
        *,
        data_path: Path,
        config: IrtConfig,
        dataset: Optional[Dataset] = None,
        verbose: bool = True,
    ) -> None:
        self._data_path = data_path
        self._config = config
        if isinstance(config.model_type, str):
            IrtModel.validate_name(config.model_type)
            self.amortized = "amortized" in self._config.model_type
        else:
            self.amortized = False
        self._priors = None
        self._device = None
        self._epochs = None
        self.irt_model: Optional[abstract_model.IrtModel] = None
        self._pyro_model = None
        self._pyro_guide = None
        self._verbose = verbose
        console.quiet = not self._verbose
        self.best_params = None
        if dataset is None:
            self._dataset = Dataset.from_jsonlines(data_path, amortized=self.amortized)
        else:
            self._dataset = dataset

        if self.amortized:
            self._config.vocab_size = len(self._dataset.observation_items[0])
        console.log(f"Vocab size: {self._config.vocab_size}")

        # filter out test data
        training_idx = [
            i
            for i in range(len(self._dataset.training_example))
            if self._dataset.training_example[i]
        ]
        self._dataset.observation_subjects = [
            self._dataset.observation_subjects[i] for i in training_idx
        ]
        self._dataset.observation_items = [
            self._dataset.observation_items[i] for i in training_idx
        ]
        self._dataset.observations = [
            self._dataset.observations[i] for i in training_idx
        ]
        self._dataset.training_example = [
            self._dataset.training_example[i] for i in training_idx
        ]

        if config.initializers is None:
            initializers = []
        else:
            initializers = config.initializers

        self._initializers = []
        for init in initializers:
            if isinstance(init, IrtInitializer):
                self._initializers.append(init)
            elif isinstance(init, str):
                self._initializers.append(INITIALIZERS[init](self._dataset))
            elif isinstance(init, Dict):
                name = init.pop("name")
                self._initializers.append(INITIALIZERS[name](self._dataset, **init))
            else:
                raise TypeError("invalid initializer type")

    def train(self, *, epochs: Optional[int] = None, device: str = "cpu") -> None:
        model_type = self._config.model_type
        if epochs is None:
            epochs = self._config.epochs
        self._device = device
        self._priors = self._config.priors
        self._epochs = epochs
        args = {
            "device": device,
            "num_items": len(self._dataset.ix_to_item_id),
            "num_subjects": len(self._dataset.ix_to_subject_id),
        }
        console.log(f"args: {args}")
        # TODO: Find a better solution to this
        if self._config.priors is not None:
            args["priors"] = self._config.priors
        else:
            args["priors"] = "vague"

        if self._config.dims is not None:
            args["dims"] = self._config.dims
        args["dropout"] = self._config.dropout
        args["hidden"] = self._config.hidden
        args["vocab_size"] = self._config.vocab_size

        console.log(f"Parsed Model Args: {args}")
        if isinstance(model_type, str):
            self.irt_model = IrtModel.from_name(model_type)(**args)
        else:
            self.irt_model = model_type(**args)
            assert isinstance(self.irt_model, IrtModel)
        pyro.clear_param_store()
        self._pyro_model = self.irt_model.get_model()
        self._pyro_guide = self.irt_model.get_guide()
        device = torch.device(device)
        scheduler = pyro.optim.ExponentialLR(
            {
                "optimizer": torch.optim.Adam,
                "optim_args": {"lr": self._config.lr},
                "gamma": self._config.lr_decay,
            }
        )
        svi = SVI(self._pyro_model, self._pyro_guide, scheduler, loss=Trace_ELBO())
        subjects = torch.tensor(
            self._dataset.observation_subjects, dtype=torch.long, device=device
        )
        items = torch.tensor(
            self._dataset.observation_items, dtype=torch.long, device=device
        )
        responses = torch.tensor(
            self._dataset.observations, dtype=torch.float, device=device
        )
        # Don't take a step here, just make sure params are initialized
        # so that initializers can modify the params
        _ = self._pyro_model(subjects, items, responses)
        _ = self._pyro_guide(subjects, items, responses)
        for init in self._initializers:
            init.initialize()

        table = Table()
        table.add_column("Epoch")
        table.add_column("Loss")
        table.add_column("Best Loss")
        table.add_column("New LR")
        loss = float("inf")
        best_loss = loss
        current_lr = self._config.lr
        
        if self._verbose:
            live = Live(table)
            live.console.print(f"Training Pyro IRT Model for {epochs} epochs")
        else:
            live = contextlib.nullcontext()

        with live:
            for epoch in range(epochs):
                loss = svi.step(subjects, items, responses)
                if loss < best_loss:
                    best_loss = loss
                    self.best_params = self.export(items)
                scheduler.step()
                current_lr = current_lr * self._config.lr_decay
                if epoch % self._config.log_every == 0:
                    table.add_row(
                        f"{epoch + 1}",
                        "%.4f" % loss,
                        "%.4f" % best_loss,
                        "%.4f" % current_lr,
                    )

            table.add_row(
                f"{epoch + 1}", "%.4f" % loss, "%.4f" % best_loss, "%.4f" % current_lr
            )
            self.last_params = self.export(items)

    def export(self, items):
        if self.amortized:
            vectorizer = CountVectorizer(max_df=0.5, min_df=20, stop_words="english")
            inputs = list(self._dataset.item_ids)
            vectorizer.fit(inputs)
            inputs = vectorizer.transform(inputs).todense().tolist()
            results = self.irt_model.export(inputs)
        else:
            results = self.irt_model.export()
        results["irt_model"] = self._config.model_type
        results["item_ids"] = self._dataset.ix_to_item_id
        results["subject_ids"] = self._dataset.ix_to_subject_id
        return results

    def save(self, output_path: Union[str, Path]):
        write_json(safe_file(output_path), self.last_params)
