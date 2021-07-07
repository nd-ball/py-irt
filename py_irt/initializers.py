"""
A set of initializers to modify how IRT models are initialized.

For example, the expression disc * (skill - diff) permits two equivalent
solutions, one where disc/skill/diff are "normal", and another one where the
sign on each is flipped. Initializing difficulty helps push towards the intuitive
solution.
"""
import abc
import torch
import pyro
from rich.console import Console
from py_irt.dataset import Dataset, ItemAccuracy


console = Console()
INITIALIZERS = {}


def register(name: str):
    def decorator(class_):
        INITIALIZERS[name] = class_
        return class_

    return decorator


class IrtInitializer(abc.ABC):
    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    def initialize(self) -> None:
        pass


@register("difficulty_sign")
class DifficultySignInitializer(IrtInitializer):
    def __init__(self, dataset: Dataset, magnitude: float = 3.0, n_to_init: int = 4):
        super().__init__(dataset)
        self._magnitude = magnitude
        self._n_to_init = n_to_init

    def initialize(self) -> None:
        """
        Initialize the hardest and easiest (by accuracy) n_to_init item difficulties.
        Set to magnitude.
        """
        item_accuracies = {}
        for item_ix, response in zip(self._dataset.observation_items, self._dataset.observations):
            if item_ix not in item_accuracies:
                item_accuracies[item_ix] = ItemAccuracy()

            item_accuracies[item_ix].correct += response
            item_accuracies[item_ix].total += 1

        sorted_item_accuracies = sorted(
            list(item_accuracies.items()), key=lambda kv: kv[1].accuracy
        )

        diff = pyro.param("loc_diff")
        for item_ix, accuracy in sorted_item_accuracies[: self._n_to_init]:
            item_id = self._dataset.ix_to_item_id[item_ix]
            console.log(f"Low Accuracy: {accuracy}, ix={item_ix} id={item_id}")
            diff.data[item_ix] = torch.tensor(
                self._magnitude, dtype=diff.data.dtype, device=diff.data.device
            )

        for item_ix, accuracy in sorted_item_accuracies[-self._n_to_init :]:
            item_id = self._dataset.ix_to_item_id[item_ix]
            console.log(f"High Accuracy: {accuracy}, ix={item_ix} id={item_id}")
            diff.data[item_ix] = torch.tensor(
                -self._magnitude, dtype=diff.data.dtype, device=diff.data.device
            )
