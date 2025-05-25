#!/usr/bin/env python
# Created by "Thieu" at 16:17, 23/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import torch
from sklearn.metrics import accuracy_score, r2_score
from sklearn.base import ClassifierMixin, RegressorMixin
from permetrics import ClassificationMetric, RegressionMetric
from mealpy import get_optimizer_by_name, Optimizer, get_all_optimizers, FloatVar
from waveletml.helpers.evaluator import get_all_regression_metrics, get_all_classification_metrics
from waveletml.models.base_model import BaseModel
from waveletml.models import custom_wnn as cwnn


class BaseMhaWnnModel(BaseModel):
    """
    Base class for Fully Metaheuristic-based Wavelet Neural Network (GdWNN) models.
    This class provides common functionality for both classifiers and regressors.
    """

    SUPPORTED_OPTIMIZERS = list(get_all_optimizers().keys())
    SUPPORTED_CLS_OBJECTIVES = get_all_classification_metrics()
    SUPPORTED_REG_OBJECTIVES = get_all_regression_metrics()

    def __init__(self, size_hidden=10, wavelet_fn="morlet", act_output=None,
                 optim="Adam", optim_params=None, obj_name=None,
                 seed=42, verbose=True, wnn_type=None):
        super().__init__()
        self.size_hidden = size_hidden
        self.wavelet_fn = wavelet_fn
        self.act_output = act_output
        self.optim = optim
        self.optim_params = optim_params
        self.obj_name = obj_name
        self.seed = seed
        self.verbose = verbose
        self.wnn_type = wnn_type

        # Initialize model parameters
        self.size_input, self.size_output = None, None
        self.network, self.optimizer = None, None
        self.metric_class, self.loss_train = None, []
        self.minmax = None

    def _set_optimizer(self, optim=None, optim_params=None):
        if isinstance(optim, str):
            opt_class = get_optimizer_by_name(optim)
            if isinstance(optim_params, dict):
                return opt_class(**optim_params)
            else:
                return opt_class(epoch=300, pop_size=30)
        elif isinstance(optim, Optimizer):
            if isinstance(optim_params, dict):
                if "name" in optim_params:  # Check if key exists and remove it
                    optim.name = optim_params.pop("name")
                optim.set_parameters(optim_params)
            return optim
        else:
            raise TypeError(f"optimizer needs to set as a string and supported by Mealpy library.")

    def get_name(self):
        """
        Generate a descriptive name for the MLP model based on the optimizer.
        """
        return f"{self.optimizer.name}-MLP-{self.optim_params}"

    def build_model(self):
        """
        Builds the model architecture and sets the optimizer and loss function based on the task.

        Raises
        ------
        ValueError
            If the task is not recognized.
        """
        
        if self.wnn_type is None:
            self.wnn_model = cwnn.CustomWaveletWeightedLinearNetwork
        elif issubclass(self.wnn_type, cwnn.BaseCustomWNN):
            self.wnn_model = self.wnn_type
        elif isinstance(self.wnn_type, str):
            self.wnn_model = getattr(cwnn, self.wnn_type, cwnn.CustomWaveletWeightedLinearNetwork)
        else:
            raise ValueError("wnn_type must be a string or an instance of BaseCustomWNN.")
        # Define model, optimizer, and loss criterion based on task
        self.network = self.wnn_model(input_dim=self.size_input, hidden_dim=self.size_hidden,
                                      output_dim=self.size_output, wavelet_fn=self.wavelet_fn,
                                      act_output=self.act_output, seed=self.seed)
        self.optimizer = self._set_optimizer(self.optim, self.optim_params)

    def _set_lb_ub(self, lb=None, ub=None, n_dims=None):
        if isinstance(lb, (list, tuple, np.ndarray)) and isinstance(ub, (list, tuple, np.ndarray)):
            if len(lb) == len(ub):
                if len(lb) == 1:
                    lb = np.array(lb * n_dims, dtype=float)
                    ub = np.array(ub * n_dims, dtype=float)
                    return lb, ub
                elif len(lb) == n_dims:
                    return lb, ub
                else:
                    raise ValueError(f"Invalid lb and ub. Their length should be equal to 1 or {n_dims}.")
            else:
                raise ValueError(f"Invalid lb and ub. They should have the same length.")
        elif isinstance(lb, (int, float)) and isinstance(ub, (int, float)):
            lb = (float(lb),) * n_dims
            ub = (float(ub),) * n_dims
            return lb, ub
        else:
            raise ValueError(f"Invalid lb and ub. They should be a number of list/tuple/np.ndarray with size equal to {n_dims}")

    def objective_function(self, solution=None):
        """
        Evaluates the fitness function for classification metrics based on the provided solution.

        Parameters
        ----------
        solution : np.ndarray, default=None
            The proposed solution to evaluate.

        Returns
        -------
        result : float
            The fitness value, representing the loss for the current solution.
        """
        X_train, y_train = self.data
        self.network.set_weights(solution)
        y_pred = self.predict(X_train)  # Get predictions from the model
        loss_train = self.metric_class(y_train, y_pred).get_metric_by_name(self.obj_name)[self.obj_name]
        return np.mean([loss_train])

    def _fit(self, data, lb=(-1.0,), ub=(1.0,), mode='single', n_workers=None,
             termination=None, save_population=False, **kwargs):
        # Get data
        n_dims = self.network.get_weights_size()
        lb, ub = self._set_lb_ub(lb, ub, n_dims)
        self.data = data

        log_to = "console" if self.verbose else "None"
        problem = {
            "obj_func": self.objective_function,
            "bounds": FloatVar(lb=lb, ub=ub),
            "minmax": self.minmax,
            "log_to": log_to,
            "save_population": save_population,
        }
        if termination is None:
            self.optimizer.solve(problem, mode=mode, n_workers=n_workers, seed=self.seed)
        else:
            self.optimizer.solve(problem, mode=mode, n_workers=n_workers, termination=termination, seed=self.seed)
        self.network.set_weights(self.optimizer.g_best.solution)
        self.loss_train = np.array(self.optimizer.history.list_global_best_fit)
        return self

