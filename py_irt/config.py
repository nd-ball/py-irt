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


from typing import List, Dict, Union, Optional, Callable
from pydantic import BaseModel, ConfigDict

# This registers all models with the registry
# pylint: disable=unused-import
from py_irt.models import *


class IrtConfig(BaseModel):
    model_type: Union[str, Callable]
    epochs: int = 2000
    priors: Optional[str] = None
    initializers: Optional[List[Union[str, Dict]]] = None
    dims: Optional[int] = None
    lr: float = 0.1
    lr_decay: float = 0.9999
    dropout: float = 0.5
    hidden: int = 100
    vocab_size: Optional[int] = None
    log_every: int = 100
    seed: Optional[int] = None
    deterministic: bool = False
    model_config = ConfigDict(protected_namespaces=())
