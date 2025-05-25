#!/usr/bin/env python
# Created by "Thieu" at 10:35, 25/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pytest
import numpy as np
from sklearn.datasets import make_regression
from waveletml import MhaWnnRegressor


def test_single_output_regression_fit_predict_score():
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

    model = MhaWnnRegressor(
        size_hidden=6,
        optim="BaseGA",
        optim_params={"epoch": 25, "pop_size": 20},
        obj_name="R2",  # R2 Score
        seed=42,
        verbose=False
    )

    model.fit(X, y)
    y_pred = model.predict(X)
    assert y_pred.shape == (X.shape[0], 1) or y_pred.shape == (X.shape[0],)
    assert isinstance(y_pred, np.ndarray)

    score = model.score(X, y)
    assert isinstance(score, float)
    assert -1 <= score <= 1


def test_multi_output_regression_fit_predict():
    X, y = make_regression(n_samples=80, n_features=4, n_targets=3, noise=0.2, random_state=0)

    model = MhaWnnRegressor(
        size_hidden=4,
        optim="OriginalPSO",
        optim_params={"epoch": 25, "pop_size": 20},
        obj_name="RMSE",  # Root Mean Squared Error
        verbose=False
    )

    model.fit(X, y)
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    assert isinstance(y_pred, np.ndarray)


def test_invalid_obj_name_raises_error():
    X, y = make_regression(n_samples=60, n_features=3, noise=0.3, random_state=0)

    model = MhaWnnRegressor(
        size_hidden=5,
        optim="OriginalDE",
        optim_params={"epoch": 25, "pop_size": 20},
        obj_name="INVALID_METRIC",
        verbose=False
    )

    with pytest.raises(ValueError, match="obj_name is not supported. Please check the library: permetrics to see the supported objective function."):
        model.fit(X, y)


def test_score_consistency_with_r2_score():
    from sklearn.metrics import r2_score

    X, y = make_regression(n_samples=50, n_features=3, noise=0.5, random_state=1)

    model = MhaWnnRegressor(
        size_hidden=5,
        optim="BaseGA",
        optim_params={"epoch": 25, "pop_size": 20},
        obj_name="R2",
        seed=1,
        verbose=False
    )

    model.fit(X, y)
    y_pred = model.predict(X)
    score1 = model.score(X, y)
    score2 = r2_score(y, y_pred)
    np.testing.assert_almost_equal(score1, score2, decimal=4)
