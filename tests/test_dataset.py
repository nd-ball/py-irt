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

from py_irt.training import Dataset


def test_jsonlines_load():
    dataset = Dataset.from_jsonlines("test_fixtures/minitest.jsonlines")
    assert len(dataset.item_ids) == 4
    assert len(dataset.subject_ids) == 4
    assert "pedro" in dataset.subject_ids
    assert "pinguino" in dataset.subject_ids
    assert "ken" in dataset.subject_ids
    assert "burt" in dataset.subject_ids
    assert "q1" in dataset.item_ids
    assert "q2" in dataset.item_ids
    assert "q3" in dataset.item_ids
    assert "q4" in dataset.item_ids
