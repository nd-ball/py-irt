from typing import List, Dict, Union, Optional
from pydantic import BaseModel

# pylint: disable=unused-import
from py_irt.models import *


class IrtConfig(BaseModel):
    model_type: str
    epochs: int = 2000
    priors: str = "hierarchical"
    initializers: Optional[List[Union[str, Dict]]] = None
