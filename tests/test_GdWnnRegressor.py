#!/usr/bin/env python
# Created by "Thieu" at 10:31, 25/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from waveletml import GdWnnRegressor


def test_fit_predict_score_regression():
    # Generate synthetic regression data
    X, y = make_regression(n_samples=150, n_features=6, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Initialize regressor
    model = GdWnnRegressor(epochs=10, batch_size=16, seed=42, verbose=False)

    # Fit model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    assert y_pred.shape == (X_test.shape[0], 1)

    # Score
    score = model.score(X_test, y_test)
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0  # R² score range

    # Scores (evaluate)
    result = model.scores(X_test, y_test)
    assert isinstance(result, dict)
    assert "MSE" in result or "MAE" in result


def test_multi_target_regression():
    # Generate synthetic multi-output regression data
    X = np.random.rand(100, 5)
    y = np.random.rand(100, 3)

    model = GdWnnRegressor(epochs=5, batch_size=16, seed=0, verbose=False)
    model.fit(X, y)

    y_pred = model.predict(X)
    assert y_pred.shape == y.shape


def test_invalid_valid_rate_regressor():
    X, y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=42)
    model = GdWnnRegressor(valid_rate=2.0)
    with pytest.raises(ValueError, match="Validation rate must be between 0 and 1."):
        model.fit(X, y)


def test_invalid_device_regressor():
    with pytest.raises(ValueError, match="GPU is not available"):
        GdWnnRegressor(device="gpu")  # Force GPU failure if not available
