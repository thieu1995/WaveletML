#!/usr/bin/env python
# Created by "Thieu" at 21:45, 20/05/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import inspect
import pickle
import pprint
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from permetrics import RegressionMetric, ClassificationMetric


class BaseModel(BaseEstimator):
    """
    Base class for all models
    """

    def __init__(self):
        self.loss_train = None

    def __repr__(self, **kwargs):
        """Pretty-print parameters like scikit-learn's Estimator."""
        param_order = list(inspect.signature(self.__init__).parameters.keys())
        param_dict = {k: getattr(self, k) for k in param_order}

        param_str = ", ".join(f"{k}={repr(v)}" for k, v in param_dict.items())
        if len(param_str) <= 80:
            return f"{self.__class__.__name__}({param_str})"
        else:
            formatted_params = ",\n  ".join(f"{k}={pprint.pformat(v)}" for k, v in param_dict.items())
            return f"{self.__class__.__name__}(\n  {formatted_params}\n)"

    def fit(self, X, y):
        """
        Fit the model to the training data.
        """
        raise NotImplementedError("fit method not implemented")

    def predict(self, X):
        """
        Predict using the trained model.
        """
        raise NotImplementedError("predict method not implemented")

    def score(self, X, y):
        """
        Evaluate the model's performance on the given data.
        """
        raise NotImplementedError("score method not implemented")

    def _evaluate_reg(self, y_true, y_pred, list_metrics=("MSE", "MAE")):
        """
        Evaluate regression performance metrics.

        Parameters
        ----------
        y_true : array-like
            True target values.
        y_pred : array-like
            Predicted values.
        list_metrics : tuple of str, list of str
            List of metrics for evaluation (e.g., "MSE" and "MAE").

        Returns
        -------
        dict
            Dictionary of calculated metric values.
        """
        rm = RegressionMetric(y_true=y_true, y_pred=y_pred)
        return rm.get_metrics_by_list_names(list_metrics)

    def _evaluate_cls(self, y_true, y_pred, list_metrics=("AS", "RS")):
        """
        Evaluate classification performance metrics.

        Parameters
        ----------
        y_true : array-like
            True target values.
        y_pred : array-like
            Predicted labels.
        list_metrics : tuple of str, list of str
            List of metrics for evaluation (e.g., "AS" and "RS").

        Returns
        -------
        dict
            Dictionary of calculated metric values.
        """
        cm = ClassificationMetric(y_true, y_pred)
        return cm.get_metrics_by_list_names(list_metrics)

    def evaluate(self, y_true, y_pred, list_metrics=None):
        """
        Evaluate the model using specified metrics.

        Parameters
        ----------
        y_true : array-like
            True target values.
        y_pred : array-like
            Model's predicted values.
        list_metrics : list of str, optional
            Names of metrics for evaluation (e.g., "MSE", "MAE").

        Returns
        -------
        dict
            Evaluation metrics and their values.
        """
        pass

    def save_training_loss(self, save_path="history", filename="loss.csv"):
        """
        Save training loss history to a CSV file.

        Parameters
        ----------
        save_path : str, optional
            Path to save the file (default: "history").
        filename : str, optional
            Filename for saving loss history (default: "loss.csv").
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        if self.loss_train is None:
            print(f"{self.__class__.__name__} model doesn't have training loss!")
        else:
            data = {"epoch": list(range(1, len(self.loss_train) + 1)), "loss": self.loss_train}
            pd.DataFrame(data).to_csv(f"{save_path}/{filename}", index=False)

    def save_evaluation_metrics(self, y_true, y_pred, list_metrics=("RMSE", "MAE"),
                                save_path="history", filename="metrics.csv"):
        """
        Save evaluation metrics to a CSV file.

        Parameters
        ----------
        y_true : array-like
            Ground truth values.
        y_pred : array-like
            Model predictions.
        list_metrics : list of str, optional
            Metrics for evaluation (default: ("RMSE", "MAE")).
        save_path : str, optional
            Path to save the file (default: "history").
        filename : str, optional
            Filename for saving metrics (default: "metrics.csv").
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        results = self.evaluate(y_true, y_pred, list_metrics)
        df = pd.DataFrame.from_dict(results, orient='index').T
        df.to_csv(f"{save_path}/{filename}", index=False)

    def save_y_predicted(self, X, y_true, save_path="history", filename="y_predicted.csv"):
        """
        Save true and predicted values to a CSV file.

        Parameters
        ----------
        X : array-like or torch.Tensor
            Input features.
        y_true : array-like
            True values.
        save_path : str, optional
            Path to save the file (default: "history").
        filename : str, optional
            Filename for saving predicted values (default: "y_predicted.csv").
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        y_pred = self.predict(X)
        data = {"y_true": np.squeeze(np.asarray(y_true)), "y_pred": np.squeeze(np.asarray(y_pred))}
        pd.DataFrame(data).to_csv(f"{save_path}/{filename}", index=False)

    def save_model(self, save_path="history", filename="model.pkl"):
        """
        Save the trained model to a pickle file.

        Parameters
        ----------
        save_path : str, optional
            Path to save the model (default: "history").
        filename : str, optional
            Filename for saving model, with ".pkl" extension (default: "model.pkl").
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        pickle.dump(self, open(f"{save_path}/{filename}", 'wb'))

    @staticmethod
    def load_model(load_path="history", filename="model.pkl"):
        """
        Load a model from a pickle file.

        Parameters
        ----------
        load_path : str, optional
            Path to load the model from (default: "history").
        filename : str, optional
            Filename of the saved model (default: "model.pkl").

        Returns
        -------
        BaseMlp
            The loaded model.
        """
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        return pickle.load(open(f"{load_path}/{filename}", 'rb'))
