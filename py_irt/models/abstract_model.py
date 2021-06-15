import abc
from typing import Dict, Any


class IrtModel(abc.ABC):
    @abc.abstractmethod
    def get_model(self):
        pass

    @abc.abstractmethod
    def get_guide(self):
        pass

    @abc.abstractmethod
    def export(self) -> Dict[str, Any]:
        pass
