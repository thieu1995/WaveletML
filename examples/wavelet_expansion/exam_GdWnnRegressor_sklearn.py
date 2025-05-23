#!/usr/bin/env python
# Created by "Thieu" at 16:36, 23/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from waveletml import DataTransformer, GdWnnRegressor, CustomWaveletExpansionNetwork


def get_cross_val_score(X, y, cv=3):
    ## Train and test
    model = GdWnnRegressor(size_hidden=10, wavelet_fn="morlet", act_output=None,
                        epochs=100, batch_size=16, optim="Adam", optim_params=None,
                        valid_rate=0.1, seed=42, verbose=True, device=None, wnn_type=CustomWaveletExpansionNetwork)
    return cross_val_score(model, X, y, cv=cv)


def get_pipe_line(X, y):
    ## Split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    ## Train and test
    model = GdWnnRegressor(size_hidden=10, wavelet_fn="morlet", act_output=None,
                        epochs=100, batch_size=16, optim="Adam", optim_params=None,
                        valid_rate=0.1, seed=42, verbose=True, device=None, wnn_type=CustomWaveletExpansionNetwork)

    pipe = Pipeline([
        ("dt", DataTransformer(scaling_methods=("standard", "minmax"))),
        ("grnn", model)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    return model.evaluate(y_true=y_test, y_pred=y_pred, list_metrics=["MAE", "RMSE", "R", "NNSE", "KGE", "R2"])


def get_grid_search(X, y):
    ## Split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    para_grid = {
        'wavelet_fn': ("morlet", "mexican_hat", "haar"),
        'size_hidden': [10, 20, 30]
    }

    ## Create a gridsearch
    model = GdWnnRegressor(epochs=50, batch_size=16, optim="Adam", optim_params=None,
                           valid_rate=0.1, seed=42, wnn_type=CustomWaveletExpansionNetwork)
    clf = GridSearchCV(model, para_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
    clf.fit(X_train, y_train)
    print("Best parameters found: ", clf.best_params_)
    print("Best model: ", clf.best_estimator_)
    print("Best training score: ", clf.best_score_)
    print(clf)

    ## Predict
    y_pred = clf.predict(X_test)
    return model.evaluate(y_true=y_test, y_pred=y_pred, list_metrics=["MAE", "RMSE", "R", "NNSE", "KGE", "R2"])


## Load data object
X, y = load_diabetes(return_X_y=True)

print(get_cross_val_score(X, y, cv=3))
print(get_pipe_line(X, y))
print(get_grid_search(X, y))
