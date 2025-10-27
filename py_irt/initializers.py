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
from py_irt.config import NEAR_ZERO_SCALE


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


@register("anchor_items")
class AnchorItemInitializer(IrtInitializer):
    """Initializer for setting fixed values for anchor items.
    
    This initializer sets the parameter values for anchor items and ensures they
    remain fixed during training by zeroing out their gradients and variance parameters.
    """
    
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        if dataset.anchor_items is None or len(dataset.anchor_items) == 0:
            raise ValueError("Dataset must have anchor items defined")
    
    def initialize(self) -> None:
        """Initialize anchor item parameters with their fixed values."""
        if self._dataset.anchor_items is None:
            return
        
        # Get parameter tensors from Pyro's param store
        loc_diff = pyro.param("loc_diff")
        scale_diff = pyro.param("scale_diff")
        
        # Check if discrimination parameters exist (2PL, 3PL, 4PL models)
        has_disc = "loc_slope" in pyro.get_param_store().keys()
        if has_disc:
            loc_disc = pyro.param("loc_slope")
            scale_disc = pyro.param("scale_slope")
        
        # Check if guessing parameters exist (3PL, 4PL models)
        has_guess = "loc_guess" in pyro.get_param_store().keys()
        if has_guess:
            loc_guess = pyro.param("loc_guess")
            scale_guess = pyro.param("scale_guess")
        
        console.log(f"Initializing {len(self._dataset.anchor_items)} anchor items:")
        
        # Create masks for anchor items
        anchor_indices = self._dataset.get_anchor_indices()
        
        for anchor in self._dataset.anchor_items:
            item_ix = anchor.item_ix
            item_id = anchor.item_id
            
            # Set difficulty (all models have this)
            if anchor.difficulty is not None:
                # Detach and set the value
                with torch.no_grad():
                    loc_diff[item_ix] = anchor.difficulty
                    # Set scale to very small value (near zero variance)
                    scale_diff[item_ix] = NEAR_ZERO_SCALE
                console.log(f"  {item_id} (ix={item_ix}): difficulty={anchor.difficulty}")
            
            # Set discrimination if available
            if has_disc and anchor.discrimination is not None:
                with torch.no_grad():
                    loc_disc[item_ix] = anchor.discrimination
                    scale_disc[item_ix] = NEAR_ZERO_SCALE
                console.log(f"  {item_id} (ix={item_ix}): discrimination={anchor.discrimination}")
            
            # Set guessing if available
            if has_guess and anchor.guessing is not None:
                with torch.no_grad():
                    loc_guess[item_ix] = anchor.guessing
                    scale_guess[item_ix] = NEAR_ZERO_SCALE
                console.log(f"  {item_id} (ix={item_ix}): guessing={anchor.guessing}")
