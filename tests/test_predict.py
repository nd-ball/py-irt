"""Tests for ``predict`` on the non-amortized IRT models."""

import json

import numpy as np
import pyro
import pytest
from scipy.special import expit
from typer.testing import CliRunner

from py_irt.cli import app
from py_irt.config import IrtConfig
from py_irt.models.abstract_model import IrtModel
from py_irt.training import IrtModelTrainer

# the tutorial model is not exported from py_irt.models; importing it registers "tutorial"
import py_irt.models.tutorial_model  # noqa: F401  pylint: disable=unused-import

FIXTURE = "test_fixtures/minitest.jsonlines"
MODEL_TYPES = ["1pl", "2pl", "3pl", "4pl", "multidim_2pl", "tutorial"]

# two subjects, three items; queries cover every subject/item at least once
SUBJECTS = [0, 1, 1, 0]
ITEMS = [0, 1, 2, 2]
ABILITY = [-0.5, 1.2]
DIFF = [0.3, -1.0, 0.8]
DISC = [1.5, 0.7, 2.0]
LAMBDAS = [0.2, 0.9, 0.35]


def _model(model_type, **kwargs):
    return IrtModel.from_name(model_type)(
        priors="vague", device="cpu", num_items=len(DIFF), num_subjects=len(ABILITY), **kwargs
    )


def _unidim_logits(disc=None):
    disc = disc if disc is not None else [1.0] * len(DIFF)
    return np.array([disc[j] * (ABILITY[i] - DIFF[j]) for i, j in zip(SUBJECTS, ITEMS)])


def test_predict_1pl():
    params = {"ability": ABILITY, "diff": DIFF}
    preds = _model("1pl").predict(SUBJECTS, ITEMS, params)
    np.testing.assert_allclose(preds, expit(_unidim_logits()))


@pytest.mark.parametrize("model_type", ["2pl", "tutorial"])
def test_predict_2pl(model_type):
    params = {"ability": ABILITY, "diff": DIFF, "disc": DISC}
    preds = _model(model_type).predict(SUBJECTS, ITEMS, params)
    np.testing.assert_allclose(preds, expit(_unidim_logits(DISC)))


def test_predict_3pl():
    params = {"ability": ABILITY, "diff": DIFF, "disc": DISC, "lambdas": LAMBDAS}
    preds = _model("3pl").predict(SUBJECTS, ITEMS, params)
    lam = np.array([LAMBDAS[j] for j in ITEMS])
    np.testing.assert_allclose(preds, lam + (1 - lam) * expit(_unidim_logits(DISC)))


def test_predict_4pl():
    params = {"ability": ABILITY, "diff": DIFF, "disc": DISC, "lambdas": LAMBDAS}
    preds = _model("4pl").predict(SUBJECTS, ITEMS, params)
    lam = np.array([LAMBDAS[j] for j in ITEMS])
    np.testing.assert_allclose(preds, lam * expit(_unidim_logits(DISC)))


def test_predict_multidim_2pl():
    ability = [[-0.5, 0.4], [1.2, -0.3]]
    diff = [[0.3, 0.1], [-1.0, 0.6], [0.8, -0.2]]
    disc = [[1.5, 0.5], [0.7, 1.1], [2.0, 0.9]]
    params = {"ability": ability, "diff": diff, "disc": disc}
    preds = _model("multidim_2pl", dims=2).predict(SUBJECTS, ITEMS, params)

    # logits are summed across dimensions before the sigmoid
    expected = [
        expit(sum(disc[j][d] * (ability[i][d] - diff[j][d]) for d in range(2)))
        for i, j in zip(SUBJECTS, ITEMS)
    ]
    assert preds.shape == (len(SUBJECTS),)
    np.testing.assert_allclose(preds, expected)


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_predict_after_training(model_type):
    pyro.set_rng_seed(0)
    dims = 3 if model_type == "multidim_2pl" else None
    config = IrtConfig(model_type=model_type, epochs=10, dims=dims)
    trainer = IrtModelTrainer(config=config, data_path=FIXTURE)
    trainer.train(device="cpu")

    dataset = trainer._dataset
    preds = trainer.irt_model.predict(dataset.observation_subjects, dataset.observation_items)
    assert preds.shape == (len(dataset.observations),)
    assert np.all((preds > 0) & (preds < 1))


def test_cli_train_and_evaluate_multidim(tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "train-and-evaluate",
            "multidim_2pl",
            FIXTURE,
            str(tmp_path),
            "--epochs",
            "10",
            "--dims",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output

    with open(tmp_path / "model_predictions.jsonlines") as f:
        rows = [json.loads(line) for line in f]
    assert rows
    assert all(0 < row["prediction"] < 1 for row in rows)
