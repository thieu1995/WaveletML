#!/usr/bin/env python
# Created by "Thieu" at 16:17, 23/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from typing import Optional, Union, Type
import numbers
import numpy as np
import torch
from sklearn.metrics import accuracy_score, r2_score
from sklearn.base import ClassifierMixin, RegressorMixin
from permetrics import ClassificationMetric, RegressionMetric
from mealpy import get_optimizer_by_class, Optimizer, get_all_optimizers, FloatVar
from waveletml.helpers.evaluator import get_all_regression_metrics, get_all_classification_metrics
from waveletml.models.base_model import BaseModel
from waveletml.models import custom_wnn as cwnn


class BaseMhaWnnModel(BaseModel):
    """
    Base class for Metaheuristic-Optimized Wavelet Neural Network (MhaWNN) models.

    This class serves as a foundation for constructing Wavelet Neural Networks (WNNs)
    optimized using various metaheuristic algorithms from the Mealpy library. It supports
    both regression and classification tasks, offering flexible model configuration,
    optimization setup, and training management.

    Parameters
    ----------
    size_hidden : int, optional (default=10)
        Number of hidden neurons in the WNN.
    wavelet_fn : str, optional (default="morlet")
        Name of the wavelet basis function used in hidden layers.
    act_output : callable or None, optional (default=None)
        Activation function for the output layer.
    optim : str, optional (default="Adam")
        Name of the metaheuristic optimizer. Must be supported by Mealpy.
    optim_params : dict, optional (default=None)
        Dictionary of parameters to configure the optimizer.
    obj_name : str or None, optional (default=None)
        Name of the objective function or performance metric (e.g., "mse", "accuracy").
    seed : int, optional (default=42)
        Random seed for reproducibility.
    verbose : bool, optional (default=True)
        If True, prints optimization progress during training.
    wnn_type : str or subclass of BaseCustomWNN, optional (default=None)
        Type or custom implementation of the Wavelet Neural Network architecture.
    lb : float, int, list, or np.ndarray, optional (default=None)
        Lower bounds for the model's trainable parameters.
    ub : float, int, list, or np.ndarray, optional (default=None)
        Upper bounds for the model's trainable parameters.
    mode : str, optional (default='single')
        Mode for optimization ('single' or 'multi').
    n_workers : int or None, optional (default=None)
        Number of parallel workers used in optimization (if supported).
    termination : dict or callable, optional (default=None)
        Termination condition for the optimization process.

    Attributes
    ----------
    size_input : int or None
        Number of features in the input dataset.
    size_output : int or None
        Number of output targets.
    network : torch.nn.Module or None
        Constructed wavelet neural network.
    optimizer : Optimizer or None
        Metaheuristic optimizer instance.
    metric_class : callable or None
        Metric computation class based on selected objective.
    loss_train : list
        History of training loss over optimization iterations.
    minmax : str or None
        Indicates whether the optimization is minimizing or maximizing the objective.
    data : tuple
        Training data (X, y) used during optimization.

    Methods
    -------
    _set_optimizer(optim=None, optim_params=None)
        Initializes the optimizer using a string or an Optimizer instance.

    get_name()
        Returns a string describing the model and optimizer configuration.

    build_model()
        Constructs the WNN architecture, initializes optimizer and loss functions.

    _set_lb_ub(lb=None, ub=None, n_dims=None)
        Normalizes and validates lower and upper bounds for optimization.

    objective_function(solution=None)
        Evaluates the loss/fitness of the model given a parameter solution.

    _fit(X, y)
        Executes the training process using metaheuristic optimization on (X, y).
    """

    SUPPORTED_OPTIMIZERS = list(get_all_optimizers(verbose=False).keys())
    SUPPORTED_CLS_OBJECTIVES = get_all_classification_metrics()
    SUPPORTED_REG_OBJECTIVES = get_all_regression_metrics()

    def __init__(self, size_hidden=10, wavelet_fn="morlet", act_output=None,
                 optim="Adam", optim_params=None, obj_name=None,
                 seed=42, verbose=True, wnn_type=None,
                 lb=None, ub=None, mode='single', n_workers=None, termination=None):
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
        self.lb = lb
        self.ub = ub
        self.mode = mode
        self.n_workers = n_workers
        self.termination = termination

        # Initialize model parameters
        self.size_input, self.size_output = None, None
        self.network, self.optimizer = None, None
        self.metric_class, self.loss_train = None, []
        self.minmax = None

    def _set_optimizer(self, optim=None, optim_params=None):
        if isinstance(optim, str):
            opt_class = get_optimizer_by_class(optim)
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
        return f"{self.optimizer.name}-WNN-{self.optim_params}"

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
        """
        Validates and sets the lower and upper bounds for optimization.

        Parameters
        ----------
        lb : list, tuple, np.ndarray, int, or float, optional
            The lower bounds for weights and biases in network.
        ub : list, tuple, np.ndarray, int, or float, optional
            The upper bounds for weights and biases in network.
        n_dims : int
            The number of dimensions.

        Returns
        -------
        tuple
            A tuple containing validated lower and upper bounds.

        Raises
        ------
        ValueError
            If the bounds are not valid.
        """
        if lb is None:
            lb = (-1.,) * n_dims
        elif isinstance(lb, numbers.Number):
            lb = (lb, ) * n_dims
        elif isinstance(lb, (list, tuple, np.ndarray)):
            if len(lb) == 1:
                lb = np.array(lb * n_dims, dtype=float)
            else:
                lb = np.array(lb, dtype=float).ravel()

        if ub is None:
            ub = (1.,) * n_dims
        elif isinstance(ub, numbers.Number):
            ub = (ub, ) * n_dims
        elif isinstance(ub, (list, tuple, np.ndarray)):
            if len(ub) == 1:
                ub = np.array(ub * n_dims, dtype=float)
            else:
                ub = np.array(ub, dtype=float).ravel()

        if len(lb) != len(ub):
            raise ValueError(f"Invalid lb and ub. Their length should be equal to 1 or {n_dims}.")

        return np.array(lb).ravel(), np.array(ub).ravel()

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

    def _fit(self, X, y):
        # Get data
        n_dims = self.network.get_weights_size()
        lb, ub = self._set_lb_ub(self.lb, self.ub, n_dims)
        self.data = (X, y)

        log_to = "console" if self.verbose else "None"
        problem = {
            "obj_func": self.objective_function,
            "bounds": FloatVar(lb=lb, ub=ub),
            "minmax": self.minmax,
            "log_to": log_to,
        }
        self.optimizer.solve(problem, mode=self.mode, n_workers=self.n_workers,
                             termination=self.termination, seed=self.seed)
        self.network.set_weights(self.optimizer.g_best.solution)
        self.loss_train = np.array(self.optimizer.history.list_global_best_fit)
        return self


class MhaWnnClassifier(BaseMhaWnnModel, ClassifierMixin):
    """
    Metaheuristic-based Wavelet Neural Network (MhaWNN) Classifier.

    This class implements a wavelet-based neural network for classification tasks,
    leveraging metaheuristic optimization for training.

    Parameters
    ----------
    size_hidden : int, optional
        Number of hidden neurons in the wavelet neural network (default is 10).
    wavelet_fn : str, optional
        Name of the wavelet function to use (default is "morlet").
    act_output : callable or None, optional
        Activation function for the output layer (default is None).
    optim : str, optional
        Name of the optimizer to use (default is "Adam").
    optim_params : dict, optional
        Parameters for the optimizer (default is None).
    obj_name : str or None, optional
        Name of the objective function for optimization (default is None).
    seed : int, optional
        Random seed for reproducibility (default is 42).
    verbose : bool, optional
        Whether to print training progress (default is True).
    wnn_type : str or Type[BaseCustomWNN] or None, optional
        Specify the type of Wavelet Neural Network to use.
        Can be:
        - A string name of a class defined in `waveletml.models.custom_wnn`
        - A class inheriting from `BaseCustomWNN`
        - None (defaults to `CustomWaveletWeightedLinearNetwork`)

    Attributes
    ----------
    size_hidden : int
        Number of hidden neurons in the wavelet neural network.
    wavelet_fn : str
        Name of the wavelet function to use.
    act_output : callable or None
        Activation function for the output layer.
    optim : str
        Name of the optimizer to use.
    optim_params : dict
        Parameters for the optimizer.
    obj_name : str or None
        Name of the objective function for optimization.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Whether to print training progress.
    wnn_type : str or None
        Type of wavelet neural network to use.
    size_input : int or None
        Number of input features.
    size_output : int or None
        Number of output features.
    network : torch.nn.Module or None
        Neural network instance.
    optimizer : Optimizer or None
        Optimizer instance.
    metric_class : callable
        Metric class for evaluation.
    classes_ : np.ndarray or None
        Unique class labels in the dataset.
    task : str
        Classification task type ('classification' or 'binary_classification').
    minmax : str or None
        Optimization direction ('min' or 'max').

    Methods
    -------
    fit(X, y, lb=(-1.0,), ub=(1.0,), mode='single', n_workers=None, termination=None, save_population=False, **kwargs)
        Fits the model to the training data.
    predict(X)
        Predicts the class labels for the provided input data.
    predict_proba(X)
        Computes the probability estimates for each class.
    score(X, y)
        Computes the accuracy score of the model based on predictions.
    evaluate(y_true, y_pred, list_metrics=("AS", "RS"))
        Evaluates the model using specified metrics for classification tasks.
    """

    def __init__(self, size_hidden=10, wavelet_fn="morlet", act_output=None,
                 optim="Adam", optim_params=None, obj_name=None,
                 seed=42, verbose=True, wnn_type: Optional[Union[str, Type[cwnn.BaseCustomWNN]]] = None,
                 lb=None, ub=None, mode='single', n_workers=None, termination=None):
        """
        Initializes the MhaWnnClassifier with specified parameters.
        """
        super().__init__(size_hidden=size_hidden, wavelet_fn=wavelet_fn, act_output=act_output,
                         optim=optim, optim_params=optim_params, obj_name=obj_name,
                         seed=seed, verbose=verbose, wnn_type=wnn_type,
                         lb=lb, ub=ub, mode=mode, n_workers=n_workers, termination=termination)
        self.classes_ = None  # Initialize classes to None
        self.metric_class = ClassificationMetric  # Set the metric class for evaluation

    def fit(self, X, y):
        """
        Fits the model to the training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
        lb : tuple, optional
            Lower bounds for optimization (default is (-1.0,)).
        ub : tuple, optional
            Upper bounds for optimization (default is (1.0,)).
        mode : str, optional
            Mode for optimization (default is 'single').
        n_workers : int, optional
            Number of workers for parallel processing (default is None).
        termination : any, optional
            Termination criteria for optimization (default is None).
        save_population : bool, optional
            Whether to save the population during optimization (default is False).
        **kwargs : additional parameters
            Additional parameters for fitting.

        Returns
        -------
        self : MhaWnnClassifier
            Returns the instance of the fitted model.
        """
        ## Check the parameters
        self.size_input = X.shape[1]  # Number of features
        y = np.squeeze(np.array(y))  # Convert y to a numpy array and squeeze dimensions
        if y.ndim != 1:
            y = np.argmax(y, axis=1)  # Convert to 1D if it’s not already
        self.classes_ = np.unique(y)  # Get unique classes from y
        if len(self.classes_) == 2:
            self.task = "binary_classification"  # Set task for binary classification
            self.size_output = 1  # Output size for binary classification
        else:
            self.task = "classification"  # Set task for multi-class classification
            self.size_output = len(self.classes_)  # Output size for multi-class

        ## Check objective function
        if type(self.obj_name) == str and self.obj_name in self.SUPPORTED_CLS_OBJECTIVES.keys():
            self.minmax = self.SUPPORTED_CLS_OBJECTIVES[self.obj_name]
        else:
            raise ValueError("obj_name is not supported. Please check the library: "
                             "permetrics to see the supported objective function.")

        ## Process data
        X_tensor = torch.tensor(X, dtype=torch.float32)  # Convert input data to tensor

        ## Build model
        self.build_model()  # Build the model architecture

        ## Fit the data
        self._fit(X_tensor, y)  # Fit the model

        return self  # Return the fitted model

    def predict(self, X):
        """
        Predicts the class labels for the provided input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data for prediction.

        Returns
        -------
        np.ndarray
            Predicted class labels for each sample.
        """
        if not isinstance(X, (torch.Tensor)):
            X = torch.tensor(X, dtype=torch.float32)  # Convert input data to tensor
        self.network.eval()  # Set model to evaluation mode
        with torch.no_grad():
            output = self.network(X)  # Get model predictions
            if self.task =="classification":        # Multi-class classification
                _, predicted = torch.max(output, 1)
            else:       # Binary classification
                predicted = (output > 0.5).int().squeeze()
        return predicted.numpy()  # Return as a numpy array

    def score(self, X, y):
        """
        Computes the accuracy score of the model based on predictions.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data for scoring.
        y : array-like, shape (n_samples,)
            True labels for comparison.

        Returns
        -------
        float
            Accuracy score of the model.
        """
        y_pred = self.predict(X)  # Get predictions
        return accuracy_score(y, y_pred)  # Calculate and return accuracy score

    def predict_proba(self, X):
        """
        Computes the probability estimates for each class (for classification tasks only).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data for which to predict probabilities.

        Returns
        -------
        np.ndarray
            Probability predictions for each class.

        Raises
        ------
        ValueError
            If the task is not a classification task.
        """
        if not isinstance(X, (torch.Tensor)):
            X = torch.tensor(X, dtype=torch.float32)  # Convert input data to tensor
        if self.task not in ["classification", "binary_classification"]:
            raise ValueError(
                "predict_proba is only available for classification tasks.")  # Raise error if task is invalid
        self.network.eval()  # Ensure model is in evaluation mode
        with torch.no_grad():
            probs = self.network.forward(X)  # Get the output from forward pass
        return probs.numpy()  # Return probabilities as a numpy array

    def evaluate(self, y_true, y_pred, list_metrics=("AS", "RS")):
        """
        Return the list of performance metrics on the given test data and labels.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values for `X`.

        list_metrics : list, default=("AS", "RS")
            List of metrics to compute using Permetrics library:
            https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            A dictionary containing the results of the requested metrics.
        """
        return self._evaluate_cls(y_true, y_pred, list_metrics)  # Call evaluation method


class MhaWnnRegressor(BaseMhaWnnModel, RegressorMixin):
    """
    Metaheuristic-based Wavelet Neural Network (MhaWNN) Regressor.

    This class implements a wavelet-based neural network for regression tasks,
    leveraging metaheuristic optimization for training.

    Parameters
    ----------
    size_hidden : int, optional
        Number of hidden neurons in the wavelet neural network (default is 10).
    wavelet_fn : str, optional
        Name of the wavelet function to use (default is "morlet").
    act_output : callable or None, optional
        Activation function for the output layer (default is None).
    optim : str, optional
        Name of the optimizer to use (default is "Adam").
    optim_params : dict, optional
        Parameters for the optimizer (default is None).
    obj_name : str or None, optional
        Name of the objective function for optimization (default is None).
    seed : int, optional
        Random seed for reproducibility (default is 42).
    verbose : bool, optional
        Whether to print training progress (default is True).
    wnn_type : str or Type[BaseCustomWNN] or None, optional
        Specify the type of Wavelet Neural Network to use.
        Can be:
        - A string name of a class defined in `waveletml.models.custom_wnn`
        - A class inheriting from `BaseCustomWNN`
        - None (defaults to `CustomWaveletWeightedLinearNetwork`)

    Attributes
    ----------
    size_hidden : int
        Number of hidden neurons in the wavelet neural network.
    wavelet_fn : str
        Name of the wavelet function to use.
    act_output : callable or None
        Activation function for the output layer.
    optim : str
        Name of the optimizer to use.
    optim_params : dict
        Parameters for the optimizer.
    obj_name : str or None
        Name of the objective function for optimization.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Whether to print training progress.
    wnn_type : str or None
        Type of wavelet neural network to use.
    size_input : int or None
        Number of input features.
    size_output : int or None
        Number of output features.
    network : torch.nn.Module or None
        Neural network instance.
    optimizer : Optimizer or None
        Optimizer instance.
    metric_class : callable
        Metric class for evaluation.
    task : str
        Regression task type ('regression' or 'multi_regression').
    minmax : str or None
        Optimization direction ('min' or 'max').

    Methods
    -------
    fit(X, y, lb=(-1.0,), ub=(1.0,), mode='single', n_workers=None, termination=None, save_population=False, **kwargs)
        Fits the model to the training data.
    predict(X)
        Predicts the output values for the provided input data.
    score(X, y)
        Computes the R2 score of the model based on predictions.
    evaluate(y_true, y_pred, list_metrics=("AS", "RS"))
        Evaluates the model using specified metrics for regression tasks.
    """

    def __init__(self, size_hidden=10, wavelet_fn="morlet", act_output=None,
                 optim="Adam", optim_params=None, obj_name=None,
                 seed=42, verbose=True, wnn_type: Optional[Union[str, Type[cwnn.BaseCustomWNN]]] = None,
                 lb=None, ub=None, mode='single', n_workers=None, termination=None):
        """
        Initializes the MhaWnnRegressor with specified parameters.
        """
        super().__init__(size_hidden=size_hidden, wavelet_fn=wavelet_fn, act_output=act_output,
                         optim=optim, optim_params=optim_params, obj_name=obj_name,
                         seed=seed, verbose=verbose, wnn_type=wnn_type,
                         lb=lb, ub=ub, mode=mode, n_workers=n_workers, termination=termination)
        self.metric_class = RegressionMetric  # Set the metric class for evaluation

    def fit(self, X, y):
        """
        Fits the model to the training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Target values.
        lb : tuple, optional
            Lower bounds for optimization (default is (-1.0,)).
        ub : tuple, optional
            Upper bounds for optimization (default is (1.0,)).
        mode : str, optional
            Mode for optimization (default is 'single').
        n_workers : int, optional
            Number of workers for parallel processing (default is None).
        termination : any, optional
            Termination criteria for optimization (default is None).
        save_population : bool, optional
            Whether to save the population during optimization (default is False).
        **kwargs : additional parameters
            Additional parameters for fitting.

        Returns
        -------
        self : MhaWnnRegressor
            Returns the instance of the fitted model.
        """
        ## Check the parameters
        self.size_input = X.shape[1]  # Number of input features
        y = np.squeeze(np.array(y))  # Convert y to a numpy array and squeeze dimensions
        self.size_output = 1  # Default output size for single-output regression
        self.task = "regression"  # Default task is regression

        if y.ndim == 2:
            self.task = "multi_regression"  # Set task for multi-output regression
            self.size_output = y.shape[1]  # Update output size for multi-output

        ## Check objective function
        if type(self.obj_name) == str and self.obj_name in self.SUPPORTED_REG_OBJECTIVES.keys():
            self.minmax = self.SUPPORTED_REG_OBJECTIVES[self.obj_name]
        else:
            raise ValueError("obj_name is not supported. Please check the library: permetrics to see the supported objective function.")

        ## Process data
        X_tensor = torch.tensor(X, dtype=torch.float32)  # Convert input data to tensor

        ## Build model
        self.build_model()  # Build the model architecture

        ## Fit the data
        self._fit(X_tensor, y)

        return self  # Return the fitted model

    def predict(self, X):
        """
        Predicts the output values for the provided input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data for prediction.

        Returns
        -------
        np.ndarray
            Predicted output values for each sample.
        """
        if not isinstance(X, (torch.Tensor)):
            X = torch.tensor(X, dtype=torch.float32)  # Convert input data to tensor
        self.network.eval()  # Set model to evaluation mode
        with torch.no_grad():
            predicted = self.network(X)  # Get model predictions
        return predicted.detach().cpu().numpy()  # Return predictions as a numpy array

    def score(self, X, y):
        """
        Computes the R2 score of the model based on predictions.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data for scoring.
        y : array-like, shape (n_samples,)
            True labels for comparison.

        Returns
        -------
        float
            R2 score of the model.
        """
        y_pred = self.predict(X)  # Get predictions
        return r2_score(y, y_pred)  # Calculate and return R^2 score

    def evaluate(self, y_true, y_pred, list_metrics=("AS", "RS")):
        """
        Return the list of performance metrics on the given test data and labels.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values for `X`.

        list_metrics : list, default=("AS", "RS")
            List of metrics to compute using Permetrics library:
            https://github.com/thieu1995/permetrics

        Returns
        -------
        results : dict
            A dictionary containing the results of the requested metrics.
        """
        return self._evaluate_reg(y_true, y_pred, list_metrics)  # Call evaluation method
