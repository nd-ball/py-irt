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


from typing import Any, Union, List
from pathlib import Path
import os
import json
from pydantic import BaseModel


def read_json(path: Union[str, Path]):
    """
    Read a json file from a string path
    """
    with open(path) as f:
        return json.load(f)


def write_json(path: Union[str, Path], obj: Any):
    """
    Write an object to a string path as json.
    If the object is a pydantic model, export it to json
    """
    if isinstance(obj, BaseModel):
        with open(path, "w") as f:
            f.write(obj.json())
    else:
        with open(path, "w") as f:
            json.dump(obj, f)


def _read_jsonlines_list(path: Union[str, Path]):
    """
    Read a jsonlines file into memory all at once
    """
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
    return out


def _read_jsonlines_lazy(path: Union[str, Path]):
    """
    Lazily return the contents of a jsonlines file
    """
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def read_jsonlines(path: Union[str, Path], lazy: bool = False):
    """
    Read a jsonlines file as a list/iterator of json objects
    """
    if lazy:
        return _read_jsonlines_lazy(path)
    else:
        return _read_jsonlines_list(path)


def write_jsonlines(path: Union[str, Path], elements: List[Any]):
    """
    Write a list of json serialiazable objects to the path given
    """
    with open(path, "w") as f:
        for e in elements:
            f.write(json.dumps(e))
            f.write("\n")


def safe_file(path: Union[str, Path]) -> Union[str, Path]:
    """
    Ensure that the path to the file exists, then return the path.
    For example, if the path passed in is /home/entilzha/stuff/stuff/test.txt,
    this function will run the equivalent of mkdir -p /home/entilzha/stuff/stuff/
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path
