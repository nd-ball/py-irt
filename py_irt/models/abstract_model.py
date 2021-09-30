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
