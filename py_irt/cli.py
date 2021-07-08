from py_irt.dataset import Dataset
from typing import Optional, List
from py_irt.io import write_json, write_jsonlines
import typer
import time
from pathlib import Path
from rich.console import Console
from py_irt.training import IrtModelTrainer
from py_irt.config import IrtConfig
import json 
import random 
from sklearn.model_selection import train_test_split
import copy 


console = Console()
app = typer.Typer()


@app.command()
def train(
    model_type: str,
    data_path: str,
    output_dir: str,
    epochs: int = 2000,
    device: str = "cpu",
    initializers: Optional[List[str]] = None,
):
    console.log(f"model_type: {model_type} data_path: {data_path}")
    start_time = time.time()
    config = IrtConfig(model_type=model_type, epochs=epochs, initializers=initializers)
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
def evaluate(
    model_type: str, 
    data_path: str, 
    output_dir: str, 
    epochs: int = 2000, 
    device: str = "cpu",
    initializers: Optional[List[str]] = None,
    evaluation: str = "heldout",
    seed: int = 42,
    train_size: float = 0.9,
):

    console.log(f"model_type: {model_type} data_path: {data_path}")
    start_time = time.time()
    config = IrtConfig(model_type=model_type, epochs=epochs, initializers=initializers)
    if evaluation == "heldout":
        with open(data_path) as f:
            items = []
            for line in f:
                submission = json.loads(line)
                model_id = submission["subject_id"]
                for example_id in submission["responses"].keys():
                    items.append((model_id, example_id))
            random.seed(seed)
            train, test = train_test_split(items, train_size=train_size)
            training_dict = {}
            for model_id, example_id in train:
                training_dict.setdefault(model_id, dict())
                training_dict[model_id][example_id] = True 
            for model_id, example_id in test:
                training_dict.setdefault(model_id, dict())
                training_dict[model_id][example_id] = False 
        dataset = Dataset.from_jsonlines(data_path, train_items=training_dict)
    else:
        dataset = Dataset.from_jsonlines(data_path)

    # deep copy for training
    training_data = copy.deepcopy(dataset) 
    trainer = IrtModelTrainer(config=config, dataset=training_data, data_path=data_path)
    output_dir = Path(output_dir)
    console.log("Training Model...")
    trainer.train(device=device)
    trainer.save(output_dir / "parameters.json")
    write_json(output_dir / "best_parameters.json", trainer.best_params)

    # get validation data
    # filter out test data
    console.log("Evaluating Model...")
    testing_idx = [i for i in range(len(dataset.training_example)) if not dataset.training_example[i]]
    if len(testing_idx) > 0:
        dataset.observation_subjects = [dataset.observation_subjects[i] for i in testing_idx]
        dataset.observation_items= [dataset.observation_items[i] for i in testing_idx]
        dataset.observations = [dataset.observations[i] for i in testing_idx]
        dataset.training_example = [dataset.training_example[i] for i in testing_idx]

    preds = trainer._irt_model.predict(dataset.observation_subjects, dataset.observation_items)
    print(preds)
    outputs = []
    for i in range(len(preds)):
        outputs.append({
            "subject_id": dataset.observation_subjects[i],
            "example_id": dataset.observation_items[i],
            "response": dataset.observations[i],
            "prediction": preds[i] 
        })
    write_jsonlines(f"{output_dir}/model_predictions.jsonlines", outputs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    console.log("Evaluation time:", elapsed_time)


if __name__ == "__main__":
    app()
