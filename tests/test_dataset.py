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
