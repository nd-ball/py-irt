from typing import List, Dict, Union, Optional
from pydantic import BaseModel

# This registers all models with the registry
# pylint: disable=unused-import
from py_irt.models import *


class IrtConfig(BaseModel):
    model_type: str
    epochs: int = 2000
    priors: Optional[str] = None
    initializers: Optional[List[Union[str, Dict]]] = None
    dims: Optional[int] = None
    lr: float = 0.1
    lr_decay: float = 0.9999
    dropout: float = 0.5
    hidden: int = 100
    vocab_size: Optional[int] = None
