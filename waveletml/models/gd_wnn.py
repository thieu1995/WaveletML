#!/usr/bin/env python
# Created by "Thieu" at 03:57, 19/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch
from waveletml.helpers.callbacks import BaseCallback, PrintLossCallback
from waveletml.models.base_model import BaseModel
from waveletml.models import custom_wnn as cwnn


class BaseGdWnnModel(BaseModel):
    """
    Base class for Fully Gradient-based Wavelet Neural Network (GdWNN) models.

    This class provides common functionality for both classifiers and regressors, including
    data processing, training, and callback handling.

    Attributes:
        size_hidden (int): Number of hidden neurons in the wavelet neural network.
        wavelet_fn (str): Name of the wavelet function to use.
        act_output (callable or None): Activation function for the output layer.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        optim (str): Name of the optimizer to use.
        optim_params (dict): Parameters for the optimizer.
        valid_rate (float): Proportion of data to use for validation.
        seed (int): Random seed for reproducibility.
        verbose (bool): Whether to print training progress.
        device (str): Device to use for training ('cpu' or 'gpu').
        callbacks (list): List of callback instances for training.
        wnn_model (class): Wavelet neural network model class.
        kwargs (dict): Additional keyword arguments.
        size_input (int or None): Number of input features.
        size_output (int or None): Number of output features.
        network (torch.nn.Module or None): Neural network instance.
        optimizer (torch.optim.Optimizer or None): Optimizer instance.
        criterion (torch.nn.Module or None): Loss function.
        valid_mode (bool): Whether validation mode is enabled.
        loss_train (list): List of training losses for each epoch.

    Methods:
        _process_data(X, y): Processes the input data for training.
        _train(X, y): Trains the model on the given data.
    """

    def __init__(self, size_hidden=10, wavelet_fn="morlet", act_output=None,
                 epochs=1000, batch_size=16, optim="Adam", optim_params=None,
                 valid_rate=0.1, seed=42, verbose=True, device=None, **kwargs):
        super().__init__()
        self.size_hidden = size_hidden
        self.wavelet_fn = wavelet_fn
        self.act_output = act_output
        self.epochs = epochs
        self.batch_size = batch_size
        self.optim = optim
        self.optim_params = optim_params
        if optim_params is None:
            self.optim_params = {}
        self.valid_rate = valid_rate
        self.seed = seed
        self.verbose = verbose
        if device == 'gpu':
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                raise ValueError("GPU is not available. Please set device to 'cpu'.")
        else:
            self.device = "cpu"

        callbacks = kwargs.get("callbacks")
        if callbacks is None:
            if verbose:
                self.callbacks = [PrintLossCallback()]
            else:
                self.callbacks = []
        elif type(callbacks) == list:
            flag = False
            for callback in callbacks:
                if isinstance(callback, PrintLossCallback):
                    flag = True
                if not isinstance(callback, BaseCallback):
                    raise ValueError("All callbacks must be instances of BaseCallback.")
            if (not flag) and verbose:
                callbacks.append(PrintLossCallback())
            self.callbacks = callbacks
        else:
            raise ValueError("Callbacks must be a list of BaseCallback instances.")

        wnn_type = kwargs.get("wnn_type")
        if wnn_type is None:
            self.wnn_model = cwnn.CustomWaveletWeightedLinearNetwork
        elif issubclass(wnn_type, cwnn.BaseCustomWNN):
            self.wnn_model = wnn_type
        elif isinstance(wnn_type, str):
            self.wnn_model = getattr(cwnn, wnn_type, cwnn.CustomWaveletWeightedLinearNetwork)
        else:
            raise ValueError("wnn_type must be a string or an instance of BaseCustomWNN.")
        self.kwargs = kwargs
        self.size_input, self.size_output = None, None
        self.network, self.optimizer, self.criterion = None, None, None
        self.valid_mode, self.loss_train = False, []

    def _process_data(self, X, y):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def _train(self, X, y):
        # Set input and output sizes based on data and initialize task
        train_loader, X_valid_tensor, y_valid_tensor = self._process_data(X, y)

        # Check callbacks on_train_begin
        for cb in self.callbacks:
            cb.on_train_begin()

        # Training loop
        for epoch in range(self.epochs):

            # Check callbacks on_epoch_begin
            for cb in self.callbacks:
                cb.on_epoch_begin(epoch)

            self.network.train()  # Set model to training mode
            train_loss = 0.0
            for batch_idx, (xb, yb) in enumerate(train_loader):
                # Check callbacks on_batch_begin
                for cb in self.callbacks:
                    cb.on_batch_begin(batch_idx)

                self.optimizer.zero_grad()  # Clear gradients
                # Forward pass
                pred = self.network(xb)
                loss = self.criterion(pred, yb)
                # Backpropagation and optimization
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()  # Accumulate batch loss
                # Check callbacks on_batch_end
                for cb in self.callbacks:
                    cb.on_batch_end(batch_idx)

            # Calculate average training loss for this epoch
            avg_train_loss = train_loss / len(train_loader)
            self.loss_train.append(avg_train_loss)

            # Perform validation if validation mode is enabled
            avg_val_loss = None
            if self.valid_mode:
                self.network.eval()  # Set model to evaluation mode
                with torch.no_grad():
                    val_output = self.network(X_valid_tensor)
                    val_loss = self.criterion(val_output, y_valid_tensor)
                    avg_val_loss = val_loss.item()

            logs = {
                "loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "model_state_dict": self.network.state_dict()
            }

            # Check callbacks on_epoch_end
            for cb in self.callbacks:
                cb.on_epoch_end(epoch, logs=logs)

            if any(getattr(cb, "stop_training", False) for cb in self.callbacks):
                break

        # Check callbacks on_train_end
        for cb in self.callbacks:
            cb.on_train_end()


class GdWnnClassifier(BaseGdWnnModel, ClassifierMixin):
    """
    Gradient-based Wavelet Neural Network (GdWNN) Classifier.

    This class implements a wavelet-based neural network for classification tasks,
    leveraging gradient-based optimization for training.

    Parameters:
    ----------
    size_hidden : int, optional
        Number of hidden neurons in the wavelet neural network (default is 10).
    wavelet_fn : str, optional
        Name of the wavelet function to use (default is "morlet").
    act_output : callable or None, optional
        Activation function for the output layer (default is None).
    epochs : int, optional
        Number of training epochs (default is 1000).
    batch_size : int, optional
        Batch size for training (default is 16).
    optim : str, optional
        Name of the optimizer to use (default is "Adam").
    optim_params : dict, optional
        Parameters for the optimizer (default is None).
    valid_rate : float, optional
        Proportion of data to use for validation (default is 0.1).
    seed : int, optional
        Random seed for reproducibility (default is 42).
    verbose : bool, optional
        Whether to print training progress (default is True).
    device : str, optional
        Device to use for training ('cpu' or 'gpu', default is None).
    kwargs : dict, optional
        Additional keyword arguments.

    Attributes:
        size_hidden (int): Number of hidden neurons in the wavelet neural network.
        wavelet_fn (str): Name of the wavelet function to use.
        act_output (callable or None): Activation function for the output layer.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        optim (str): Name of the optimizer to use.
        optim_params (dict): Parameters for the optimizer.
        valid_rate (float): Proportion of data to use for validation.
        seed (int): Random seed for reproducibility.
        verbose (bool): Whether to print training progress.
        device (str): Device to use for training ('cpu' or 'gpu').
        classes_ (np.ndarray or None): Unique class labels in the dataset.
        task (str): Classification task type ('classification' or 'binary_classification').

    Methods:
        _process_data(X, y): Processes the input data for training and validation.
        fit(X, y): Fits the model to the training data.
        predict(X): Predicts class labels for the input data.
        predict_proba(X): Predicts class probabilities for the input data.
        score(X, y): Computes the accuracy of the model on the given data.
        evaluate(y_true, y_pred, list_metrics): Evaluates the model using specified metrics.
        scores(X, y, list_metrics): Computes evaluation metrics on the given data.
    """

    def __init__(self, size_hidden=10, wavelet_fn="morlet", act_output=None,
                 epochs=1000, batch_size=16, optim="Adam", optim_params=None,
                 valid_rate=0.1, seed=42, verbose=True, device=None, **kwargs):
        """
        Initializes the GdWnnClassifier with specified parameters.
        """
        super().__init__(size_hidden, wavelet_fn, act_output, epochs, batch_size,
                         optim, optim_params, valid_rate, seed, verbose, device, **kwargs)
        self.classes_, self.task = None, "classification"

    def _process_data(self, X, y):
        X_valid_tensor, y_valid_tensor, X_valid, y_valid = None, None, None, None

        # Split data into training and validation sets based on valid_rate
        if self.valid_rate is not None:
            if 0 < self.valid_rate < 1:
                # Activate validation mode if valid_rate is set between 0 and 1
                self.valid_mode = True
                X, X_valid, y, y_valid = train_test_split(X, y, test_size=self.valid_rate,
                                                          random_state=self.seed, shuffle=True, stratify=y)
            else:
                raise ValueError("Validation rate must be between 0 and 1.")

        # Convert data to tensors and set up DataLoader
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        if self.task == "binary_classification":
            y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)
        else:
            y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        # y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=self.batch_size, shuffle=True)

        if self.valid_mode:
            X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).to(self.device)
            # y_valid_tensor = torch.tensor(y_valid, dtype=torch.long).to(self.device)
            if self.task == "binary_classification":
                y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).view(-1, 1).to(self.device)
            else:
                y_valid_tensor = torch.tensor(y_valid, dtype=torch.long).to(self.device)
        return train_loader, X_valid_tensor, y_valid_tensor

    def fit(self, X, y):
        """
        Fits the GdWnnClassifier to the training data.
        """
        # Set input and output sizes based on data and initialize task
        self.size_input = X.shape[1]
        self.classes_ = np.unique(y)
        if len(self.classes_) == 2:
            self.task = "binary_classification"
            self.size_output = 1
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.task = "classification"
            self.size_output = len(self.classes_)
            self.criterion = nn.CrossEntropyLoss()

        # Define model, optimizer, and loss criterion based on task
        self.network = self.wnn_model(input_dim=self.size_input, hidden_dim=self.size_hidden,
                                      output_dim=self.size_output, wavelet_fn=self.wavelet_fn,
                                      act_output=self.act_output, seed=self.seed).to(self.device)
        self.optimizer = getattr(torch.optim, self.optim)(self.network.parameters(), **self.optim_params)
        self._train(X, y)  # Call the training method
        return self

    def predict(self, X):
        """
        Predict class labels for the input data.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.network.eval()
        with torch.no_grad():
            logits = self.network(X_tensor)  # Get model predictions
            if self.task == "classification":  # Multi-class classification
                _, predicted = torch.max(logits, 1)
            else:  # Binary classification
                predicted = (logits > 0.5).int().squeeze()
        return predicted.cpu().numpy()

    def predict_proba(self, X):
        """
        Predict class probabilities for the input data.
        """
        self.network.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.network(X_tensor)
            if self.task == "binary_classification":
                probs = torch.sigmoid(logits)
            else:
                probs = torch.softmax(logits, dim=1)
            return probs.cpu().numpy()

    def score(self, X, y):
        """Return the accuracy on the given test data and labels."""
        return accuracy_score(y, self.predict(X))

    def evaluate(self, y_true, y_pred, list_metrics=("AS", "RS")):
        """
        Evaluates the model using specified metrics.
        """
        return self._evaluate_cls(y_true=y_true, y_pred=y_pred, list_metrics=list_metrics)

    def scores(self, X, y, list_metrics=("AS", "RS")):
        """
        Computes evaluation metrics on the given data.
        """
        y_pred = self.predict(X)
        return self.evaluate(y, y_pred, list_metrics)


class GdWnnRegressor(BaseGdWnnModel, RegressorMixin):
    """
    Gradient-based Wavelet Neural Network (GdWNN) Regressor.

    This class implements a wavelet-based neural network for regression tasks,
    leveraging gradient-based optimization for training.

    Parameters
    ----------
    size_hidden : int, optional
        Number of hidden neurons in the wavelet neural network (default is 10).
    wavelet_fn : str, optional
        Name of the wavelet function to use (default is "morlet").
    act_output : callable or None, optional
        Activation function for the output layer (default is None).
    epochs : int, optional
        Number of training epochs (default is 1000).
    batch_size : int, optional
        Batch size for training (default is 16).
    optim : str, optional
        Name of the optimizer to use (default is "Adam").
    optim_params : dict, optional
        Parameters for the optimizer (default is None).
    valid_rate : float, optional
        Proportion of data to use for validation (default is 0.1).
    seed : int, optional
        Random seed for reproducibility (default is 42).
    verbose : bool, optional
        Whether to print training progress (default is True).
    device : str, optional
        Device to use for training ('cpu' or 'gpu', default is None).
    kwargs : dict, optional
        Additional keyword arguments.

    Attributes
    ----------
    size_hidden : int
        Number of hidden neurons in the wavelet neural network.
    wavelet_fn : str
        Name of the wavelet function to use.
    act_output : callable or None
        Activation function for the output layer.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.
    optim : str
        Name of the optimizer to use.
    optim_params : dict
        Parameters for the optimizer.
    valid_rate : float
        Proportion of data to use for validation.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Whether to print training progress.
    device : str
        Device to use for training ('cpu' or 'gpu').
    task : str
        Regression task type ('regression' or 'multi_regression').
    size_input : int or None
        Number of input features.
    size_output : int or None
        Number of output features.
    network : torch.nn.Module or None
        Neural network instance.
    optimizer : torch.optim.Optimizer or None
        Optimizer instance.
    criterion : torch.nn.Module or None
        Loss function.
    valid_mode : bool
        Whether validation mode is enabled.
    loss_train : list
        List of training losses for each epoch.

    Methods
    -------
    _process_data(X, y)
        Processes the input data for training and validation.
    fit(X, y)
        Fits the model to the training data.
    predict(X)
        Predicts the output values for the provided input data.
    score(X, y)
        Computes the R2 score of the model on the given data.
    evaluate(y_true, y_pred, list_metrics=("MSE", "MAE"))
        Evaluates the model using specified metrics.
    scores(X, y, list_metrics=("MSE", "MAE"))
        Computes evaluation metrics on the given data.
    """

    def __init__(self, size_hidden=10, wavelet_fn="morlet", act_output=None,
                 epochs=1000, batch_size=16, optim="Adam", optim_params=None,
                 valid_rate=0.1, seed=42, verbose=True, device=None, **kwargs):
        super().__init__(size_hidden, wavelet_fn, act_output, epochs, batch_size,
                         optim, optim_params, valid_rate, seed, verbose, device, **kwargs)
        self.task = "regression"

    def _process_data(self, X, y):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        X_valid_tensor, y_valid_tensor, X_valid, y_valid = None, None, None, None

        # Split data into training and validation sets based on valid_rate
        if self.valid_rate is not None:
            if 0 < self.valid_rate < 1:
                # Activate validation mode if valid_rate is set between 0 and 1
                self.valid_mode = True
                X, X_valid, y, y_valid = train_test_split(X, y, test_size=self.valid_rate,
                                                          random_state=self.seed, shuffle=True)
            else:
                raise ValueError("Validation rate must be between 0 and 1.")

        # Convert data to tensors and set up DataLoader
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=self.batch_size, shuffle=True)

        if self.valid_mode:
            X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).to(self.device)
            y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).to(self.device)
        return train_loader, X_valid_tensor, y_valid_tensor

    def fit(self, X, y):
        """
        Fits the GdWnnRegressor to the training data.
        """
        # Set up data
        self.size_input = X.shape[1]
        self.size_output = 1
        y = np.squeeze(np.array(y))
        if y.ndim == 2:     # Adjust task if multi-dimensional target is provided
            self.task = "multi_regression"
            self.size_output = y.shape[1]

        # Define model, optimizer, and loss criterion based on task
        self.network = self.wnn_model(input_dim=self.size_input, hidden_dim=self.size_hidden,
                                      output_dim=self.size_output, wavelet_fn=self.wavelet_fn,
                                      act_output=self.act_output, seed=self.seed).to(self.device)
        self.optimizer = getattr(torch.optim, self.optim)(self.network.parameters(), **self.optim_params)
        self.criterion = nn.MSELoss()
        self._train(X, y)  # Call the training method
        return self

    def predict(self, X):
        """
        Predicts the output for the given input data.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.network.eval()  # Set model to evaluation mode
        with torch.no_grad():
            predicted = self.network(X_tensor)  # Forward pass to get predictions
        return predicted.cpu().numpy()  # Convert predictions to numpy array

    def score(self, X, y):
        """
        Returns the coefficient of determination R2 of the prediction.
        """
        return r2_score(y, self.predict(X))

    def evaluate(self, y_true, y_pred, list_metrics=("MSE", "MAE")):
        """
        Evaluates the model using specified metrics for regression tasks.
        """
        return self._evaluate_reg(y_true, y_pred, list_metrics)  # Call the evaluation method

    def scores(self, X, y, list_metrics=("MSE", "MAE")):
        """
        Computes evaluation metrics on the given data.
        """
        y_pred = self.predict(X)
        return self.evaluate(y, y_pred, list_metrics)
