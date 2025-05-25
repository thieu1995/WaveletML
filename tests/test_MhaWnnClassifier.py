#!/usr/bin/env python
# Created by "Thieu" at 10:33, 25/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pytest
import numpy as np
from sklearn.datasets import make_classification
from waveletml import MhaWnnClassifier


def test_binary_classification_fit_predict_score():
    X, y = make_classification(n_samples=120, n_features=6, n_classes=2, random_state=42)

    model = MhaWnnClassifier(
        size_hidden=5,
        optim="BaseGA",  # Use Genetic Algorithm for diversity
        optim_params={"epoch": 20, "pop_size": 20},
        obj_name="AS",  # Accuracy Score
        seed=42,
        verbose=False
    )

    model.fit(X, y)
    y_pred = model.predict(X)
    assert y_pred.shape == (X.shape[0],)
    assert np.all(np.isin(y_pred, [0, 1]))

    score = model.score(X, y)
    assert isinstance(score, float)
    assert 0 <= score <= 1


def test_multiclass_classification_fit_predict():
    X, y = make_classification(n_samples=100, n_features=5, n_classes=3, n_informative=3, n_redundant=0,
                               random_state=42)

    model = MhaWnnClassifier(
        size_hidden=6,
        optim="OriginalPSO",  # Particle Swarm Optimization
        optim_params={"epoch": 20, "pop_size": 20},
        obj_name="AS",
        seed=1,
        verbose=False
    )

    model.fit(X, y)
    y_pred = model.predict(X)
    assert y_pred.shape == (X.shape[0],)
    assert set(np.unique(y_pred)).issubset(set(np.unique(y)))


def test_predict_proba_output_shape():
    X, y = make_classification(n_samples=80, n_features=4, n_classes=2, random_state=0)

    model = MhaWnnClassifier(
        optim="OriginalDE",
        optim_params={"epoch": 20, "pop_size": 20},
        obj_name="AS",
        verbose=False
    )

    model.fit(X, y)
    probs = model.predict_proba(X)
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (X.shape[0], 1) or probs.shape[1] > 1


def test_invalid_obj_name_raises_error():
    X, y = make_classification(
        n_samples=60,
        n_features=3,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        random_state=0
    )

    model = MhaWnnClassifier(
        optim="BaseGA",
        optim_params={"epoch": 20, "pop_size": 20},
        obj_name="INVALID_METRIC",
        verbose=False
    )

    with pytest.raises(ValueError,
                       match="obj_name is not supported. Please check the library: permetrics to see the supported objective function."):
        model.fit(X, y)


def test_predict_proba_invalid_task():
    X, y = make_classification(
        n_samples=60,
        n_features=3,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        random_state=0
    )

    model = MhaWnnClassifier(
        optim="BaseGA",
        optim_params={"epoch": 25, "pop_size": 20},
        obj_name="AS",
        verbose=False
    )

    model.fit(X, y)
    model.task = "regression"  # Force task to be incorrect
    with pytest.raises(ValueError, match="predict_proba is only available for classification tasks."):
        model.predict_proba(X)
