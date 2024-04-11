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


import abc
from typing import Dict, Any

_IRT_REGISTRY = {}


class IrtModel(abc.ABC):
    def __init__(
        self, *, num_items: int, num_subjects: int, verbose: bool = False, device: str = "cpu"
    ) -> None:
        super().__init__()
        if device not in ["cpu", "cuda"]:
            raise ValueError("Options for device are cpu and cuda")
        if num_items <= 0:
            raise ValueError("Number of items must be greater than 0")
        if num_subjects <= 0:
            raise ValueError("Number of subjects must be greater than 0")
        self.device = device
        self.num_items = num_items
        self.num_subjects = num_subjects
        self.verbose = verbose

    @classmethod
    def register(cls, name: str):
        def add_to_registry(class_):
            if class_ in _IRT_REGISTRY:
                raise ValueError(f"Model name already in registry: {class_}")
            _IRT_REGISTRY[name] = class_
            return class_

        return add_to_registry

    @classmethod
    def from_name(cls, name: str):
        if name not in _IRT_REGISTRY:
            raise ValueError(f"Unknown model name: {name}, Registry:\n{_IRT_REGISTRY}")
        return _IRT_REGISTRY[name]

    @classmethod
    def validate_name(cls, name: str):
        if name not in _IRT_REGISTRY:
            raise ValueError(f"Unknown model name: {name}, Registry:\n{_IRT_REGISTRY}")
        return

    @abc.abstractmethod
    def get_model(self):
        pass

    @abc.abstractmethod
    def get_guide(self):
        pass

    @abc.abstractmethod
    def export(self) -> Dict[str, Any]:
        pass

    @classmethod
    def train(cls, dataset, **kw):
        # import here to avoid circular import
        from py_irt.config import IrtConfig
        from py_irt.training import IrtModelTrainer

        inv_irt_registry = {v: k for k, v in _IRT_REGISTRY.items()}
        try:
            my_name = inv_irt_registry[cls]
        except KeyError:
            raise KeyError("model not found in registry")

        config = IrtConfig(model_type=my_name, **kw)
        trainer = IrtModelTrainer(dataset=dataset, data_path=None, config=config)
        trainer.train()
        
        return trainer
