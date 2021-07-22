import abc
from typing import Dict, Any


_IRT_REGISTRY = {}


class IrtModel(abc.ABC):
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
            raise ValueError(f"Unknown model name: {name}")
        return _IRT_REGISTRY[name]

    @classmethod
    def validate_name(cls, name: str):
        if name not in _IRT_REGISTRY:
            raise ValueError(f"Unknown model name: {name}")
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
