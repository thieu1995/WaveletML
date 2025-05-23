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
    This class provides common functionality for both classifiers and regressors.
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

    def __init__(self, size_hidden=10, wavelet_fn="morlet", act_output=None,
                 epochs=1000, batch_size=16, optim="Adam", optim_params=None,
                 valid_rate=0.1, seed=42, verbose=True, device=None, **kwargs):
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
        return self._evaluate_cls(y_true=y_true, y_pred=y_pred, list_metrics=list_metrics)

    def scores(self, X, y, list_metrics=("AS", "RS")):
        y_pred = self.predict(X)
        return self.evaluate(y, y_pred, list_metrics)


class GdWnnRegressor(BaseGdWnnModel, RegressorMixin):

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
        return self._evaluate_reg(y_true, y_pred, list_metrics)  # Call the evaluation method

    def scores(self, X, y, list_metrics=("MSE", "MAE")):
        y_pred = self.predict(X)
        return self.evaluate(y, y_pred, list_metrics)
