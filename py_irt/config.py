from typing import List, Dict, Union, Optional
from pydantic import BaseModel


class IrtConfig(BaseModel):
    model_type: str
    epochs: int = 2000
    priors: str = "hierarchical"
    initializers: Optional[List[Union[str, Dict]]] = None
