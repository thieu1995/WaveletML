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
from waveletml.models.base_wnn import CustomWNN


class GdWnnClassifier(BaseModel, ClassifierMixin):

    def __init__(self, size_hidden=10, wavelet_fn="morlet", act_output=None,
                 epochs=1000, batch_size=16, optim="Adam", optim_paras=None,
                 valid_rate=0.1, seed=42, verbose=True, device=None, callbacks=None):
        super().__init__()
        self.size_hidden = size_hidden
        self.wavelet_fn = wavelet_fn
        self.act_output = act_output
        self.epochs = epochs
        self.batch_size = batch_size
        self.optim = optim
        self.optim_paras = optim_paras if optim_paras else {}
        self.valid_rate = valid_rate
        self.seed = seed
        self.verbose = verbose
        self.callbacks = callbacks if callbacks else []
        if callbacks is None:
            if verbose:
                self.callbacks = [PrintLossCallback()]
            else:
                self.callbacks = []
        elif type(callbacks) == list:
            for callback in callbacks:
                if not isinstance(callback, BaseCallback):
                    raise ValueError("All callbacks must be instances of BaseCallback.")
            self.callbacks = callbacks
        else:
            raise ValueError("Callbacks must be a list of BaseCallback instances.")

        if device == 'gpu':
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                raise ValueError("GPU is not available. Please set device to 'cpu'.")
        else:
            self.device = "cpu"

        self.size_input, self.size_output = None, None
        self.network, self.optimizer, self.criterion = None, None, None
        self.valid_mode, self.loss_train = False, []

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
        # y = np.squeeze(np.array(y))
        # if y.ndim != 1:
        #     y = np.argmax(y, axis=1)

        self.classes_ = np.unique(y)
        if len(self.classes_) == 2:
            self.task = "binary_classification"
            self.size_output = 1
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.task = "classification"
            self.size_output = len(self.classes_)
            self.criterion = nn.CrossEntropyLoss()

        train_loader, X_valid_tensor, y_valid_tensor = self._process_data(X, y)

        # Define model, optimizer, and loss criterion based on task
        self.network = CustomWNN(size_input=self.size_input, size_hidden=self.size_hidden,
                                 size_output=self.size_output, wavelet_fn=self.wavelet_fn,
                                 act_output=self.act_output, seed=self.seed).to(self.device)
        self.optimizer = getattr(torch.optim, self.optim)(self.network.parameters(), **self.optim_paras)

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
                train_loss += loss.item()   # Accumulate batch loss
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

