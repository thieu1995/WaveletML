#!/usr/bin/env python
# Created by "Thieu" at 10:29, 25/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from waveletml import GdWnnClassifier


@pytest.mark.parametrize("n_classes", [2, 3])
def test_fit_predict_score(n_classes):
    # Create synthetic classification dataset
    X, y = make_classification(n_samples=200, n_features=5, n_informative=3,
                               n_classes=n_classes, random_state=42)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize classifier
    model = GdWnnClassifier(epochs=10, batch_size=16, seed=42, verbose=False)

    # Fit model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    assert y_pred.shape == y_test.shape
    assert set(np.unique(y_pred)).issubset(set(np.unique(y_train)))

    # Predict probabilities
    y_proba = model.predict_proba(X_test)
    assert y_proba.shape[0] == X_test.shape[0]
    if n_classes == 2:
        assert y_proba.shape[1] == 1 or len(y_proba.shape) == 1
    else:
        assert y_proba.shape[1] == n_classes

    # Check accuracy score
    score = model.score(X_test, y_test)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_invalid_valid_rate():
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    model = GdWnnClassifier(valid_rate=1.5)
    with pytest.raises(ValueError, match="Validation rate must be between 0 and 1."):
        model.fit(X, y)


def test_invalid_device():
    with pytest.raises(ValueError, match="GPU is not available"):
        GdWnnClassifier(device="gpu")  # Force failure if CUDA is not available


def test_metrics_output():
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    model = GdWnnClassifier(epochs=5, batch_size=16, seed=42, verbose=False)
    model.fit(X, y)
    scores = model.scores(X, y)
    assert isinstance(scores, dict)
    assert "AS" in scores or "RS" in scores
