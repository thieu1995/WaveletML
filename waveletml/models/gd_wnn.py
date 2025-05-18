#!/usr/bin/env python
# Created by "Thieu" at 03:57, 19/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import torch


class GdWaveletRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, in_features, hidden_features=10, out_features=1,
                 wavelet_fn=morlet, epochs=100, lr=0.01, batch_size=32):
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.wavelet_fn = wavelet_fn
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

    def fit(self, X, y):
        self.model_ = WNNBase(self.in_features, self.hidden_features, self.out_features, self.wavelet_fn)
        self.model_.train()

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for xb, yb in loader:
                pred = self.model_(xb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.model_.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.model_(X_tensor).numpy()


class GdWaveletClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, in_features, hidden_features=10, out_features=2,
                 wavelet_fn=morlet, epochs=100, lr=0.01, batch_size=32):
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.wavelet_fn = wavelet_fn
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

    def fit(self, X, y):
        self.model_ = WNNBase(self.in_features, self.hidden_features, self.out_features, self.wavelet_fn)
        self.model_.train()

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for xb, yb in loader:
                pred = self.model_(xb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.model_.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model_(X_tensor)
            return torch.argmax(logits, dim=1).numpy()

    def predict_proba(self, X):
        self.model_.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model_(X_tensor)
            probs = torch.softmax(logits, dim=1)
            return probs.numpy()
