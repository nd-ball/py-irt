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


import random
from typing import Optional, List
import time
import json
import copy
from pathlib import Path

import torch
import pyro
import numpy as np
import typer
import toml
from rich.console import Console
from sklearn.model_selection import train_test_split

from py_irt.training import IrtModelTrainer
from py_irt.config import IrtConfig
from py_irt.dataset import Dataset
from py_irt.models.abstract_model import IrtModel
from py_irt.io import write_json, write_jsonlines, read_jsonlines, read_json

console = Console()
app = typer.Typer()


@app.command()
def train(
    model_type: str,
    data_path: str,
    output_dir: str,
    epochs: Optional[int] = None,
    priors: Optional[str] = None,
    dims: Optional[int] = None,
    lr: Optional[float] = None,
    lr_decay: Optional[float] = None,
    device: str = "cpu",
    initializers: Optional[List[str]] = None,
    config_path: Optional[str] = None,
    dropout: Optional[float] = 0.5,
    hidden: Optional[int] = 100,
    seed: Optional[int] = None,
    deterministic: bool = False,
    log_every: int = 100,
):
    if config_path is None:
        parsed_config = {}
    else:
        if config_path.endswith(".toml"):
            with open(config_path) as f:
                parsed_config = toml.load(f)
        else:
            parsed_config = read_json(config_path)

    args_config = {
        "priors": priors,
        "dims": dims,
        "lr": lr,
        "lr_decay": lr_decay,
        "epochs": epochs,
        "initializers": initializers,
        "model_type": model_type,
        "dropout": dropout,
        "hidden": hidden,
        "log_every": log_every,
        "deterministic": deterministic,
        "seed": seed,
    }
    for key, value in args_config.items():
        if value is not None:
            parsed_config[key] = value

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        pyro.set_rng_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(True)

    if model_type != parsed_config["model_type"]:
        raise ValueError("Mismatching model types in args and config")

    config = IrtConfig(**parsed_config)
    console.log(f"config: {config}")

    console.log(f"data_path: {data_path}")
    console.log(f"output directory: {output_dir}")
    start_time = time.time()
    trainer = IrtModelTrainer(config=config, data_path=data_path)
    output_dir = Path(output_dir)
    console.log("Training Model...")
    trainer.train(device=device)
    trainer.save(output_dir / "parameters.json")
    write_json(output_dir / "best_parameters.json", trainer.best_params)
    end_time = time.time()
    elapsed_time = end_time - start_time
    console.log("Train time:", elapsed_time)


@app.command()
def train_and_evaluate(
    model_type: str,
    data_path: str,
    output_dir: str,
    epochs: int = 2000,
    priors: Optional[str] = None,
    dims: Optional[int] = None,
    device: str = "cpu",
    lr: Optional[float] = None,
    lr_decay: Optional[float] = None,
    initializers: Optional[List[str]] = None,
    evaluation: str = "heldout",
    seed: int = 42,
    train_size: float = 0.9,
    config_path: Optional[str] = None,
    dropout: Optional[float] = 0.5,
    hidden: Optional[int] = 100
):
    if config_path is None:
        parsed_config = {}
    else:
        if config_path.endswith(".toml"):
            with open(config_path) as f:
                parsed_config = toml.load(f)
        else:
            parsed_config = read_json(config_path)

    args_config = {
        "priors": priors,
        "dims": dims,
        "lr": lr,
        "lr_decay": lr_decay,
        "epochs": epochs,
        "initializers": initializers,
        "model_type": model_type,
        "dropout": dropout,
        "hidden": hidden,
    }
    for key, value in args_config.items():
        if value is not None:
            parsed_config[key] = value
    if 'amortized' in model_type:
        amortized = True
    else:
        amortized = False

    if model_type != parsed_config["model_type"]:
        raise ValueError("Mismatching model types in args and config")
    start_time = time.time()
    config = IrtConfig(**parsed_config)
    console.log(f"config: {config}")

    console.log(f"data_path: {data_path}")
    console.log(f"output directory: {output_dir}")

#    config = IrtConfig(model_type=model_type, epochs=epochs, initializers=initializers)
    if evaluation == "heldout":
        with open(data_path) as f:
            items = []
            for line in f:
                submission = json.loads(line)
                model_id = submission["subject_id"]
                for example_id in submission["responses"].keys():
                    items.append((model_id, example_id))
            train, test = train_test_split(
                items, train_size=train_size, random_state=seed)
            training_dict = {}
            for model_id, example_id in train:
                training_dict.setdefault(model_id, dict())
                training_dict[model_id][example_id] = True
            for model_id, example_id in test:
                training_dict.setdefault(model_id, dict())
                training_dict[model_id][example_id] = False
        dataset = Dataset.from_jsonlines(
            data_path, train_items=training_dict, amortized=amortized)
    else:
        dataset = Dataset.from_jsonlines(data_path, amortized=amortized)

    # deep copy for training
    training_data = copy.deepcopy(dataset)
    trainer = IrtModelTrainer(
        config=config, dataset=training_data, data_path=data_path)
    output_dir = Path(output_dir)
    console.log("Training Model...")
    trainer.train(device=device)
    trainer.save(output_dir / "parameters.json")
    write_json(output_dir / "best_parameters.json", trainer.best_params)

    # get validation data
    # filter out test data
    console.log("Evaluating Model...")
    testing_idx = [
        i for i in range(len(dataset.training_example)) if not dataset.training_example[i]
    ]
    if len(testing_idx) > 0:
        dataset.observation_subjects = [
            dataset.observation_subjects[i] for i in testing_idx]
        dataset.observation_items = [
            dataset.observation_items[i] for i in testing_idx]
        dataset.observations = [dataset.observations[i] for i in testing_idx]
        dataset.training_example = [
            dataset.training_example[i] for i in testing_idx]

    preds = trainer.irt_model.predict(
        dataset.observation_subjects, dataset.observation_items)
    outputs = []
    for i in range(len(preds)):
        outputs.append(
            {
                "subject_id": dataset.observation_subjects[i],
                "example_id": dataset.observation_items[i],
                "response": dataset.observations[i],
                "prediction": preds[i],
            }
        )
    write_jsonlines(f"{output_dir}/model_predictions.jsonlines", outputs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    console.log("Evaluation time:", elapsed_time)


@app.command()
def evaluate(
    model_type: str,
    parameter_path: str,
    test_pairs_path: str,
    output_dir: str,
    epochs: int = 2000,
    device: str = "cpu",
    initializers: Optional[List[str]] = None,
    evaluation: str = "heldout",
    seed: int = 42,
    train_size: float = 0.9,
):
    console.log(
        f"model_type: {model_type}, parameter_path: {parameter_path}, test_pairs_path: {test_pairs_path}"
    )
    console.log(f"output directory: {output_dir}")
    start_time = time.time()
    console.log("Evaluating Model...")
    # load saved params
    irt_params = read_json(parameter_path)

    # reverse dict for lookup
    subjectIDX = {}
    for (key, item) in irt_params["subject_ids"].items():
        subjectIDX[item] = int(key)

    itemIDX = {}
    for (key, item) in irt_params["item_ids"].items():
        itemIDX[item] = int(key)

    # load subject, item pairs we want to test
    subject_item_pairs = read_jsonlines(test_pairs_path)

    # calculate predictions and write them to disk
    config = IrtConfig(model_type=model_type, epochs=epochs,
                       initializers=initializers)
    
    observation_subjects = [subjectIDX[entry["subject_id"]]
                            for entry in subject_item_pairs]
    observation_items = [itemIDX[entry["item_id"]] for entry in subject_item_pairs]

    irt_model = IrtModel.from_name(model_type)(
        priors="vague",
        device=device,
        num_items=len(set(observation_items)),
        num_subjects=len(set(observation_subjects)),
    )

    preds = irt_model.predict(observation_subjects,
                              observation_items, irt_params)
    outputs = []
    for i in range(len(preds)):
        outputs.append(
            {
                "subject_id": observation_subjects[i],
                "example_id": observation_items[i],
                "prediction": preds[i],
            }
        )
    write_jsonlines(f"{output_dir}/model_predictions.jsonlines", outputs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    console.log("Evaluation time:", elapsed_time)


if __name__ == "__main__":
    app()
