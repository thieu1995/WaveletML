#!/usr/bin/env python
# Created by "Thieu" at 03:57, 19/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import torch
import torch.nn as nn
from waveletml.helpers import wavelet_funcs as wf


class WaveletLayer(nn.Module):
    def __init__(self, in_features, out_features, wavelet_fn):
        super(WaveletLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_fn = wavelet_fn  # a callable function

        self.centers = nn.Parameter(torch.randn(out_features, in_features))
        self.scales = nn.Parameter(torch.ones(out_features, in_features))

    def forward(self, x):
        # x: [batch_size, in_features]
        x = x.unsqueeze(1)  # [batch_size, 1, in_features]
        centers = self.centers.unsqueeze(0)  # [1, out_features, in_features]
        scales = self.scales.unsqueeze(0)  # [1, out_features, in_features]

        u = (x - centers) / scales  # broadcasting
        # apply wavelet function element-wise
        wave = self.wavelet_fn(u)
        return wave.view(x.shape[0], -1)  # Flatten to [batch_size, out_features * in_features]


class CustomWNN(nn.Module):

    SUPPORTED_ACTIVATIONS = [
        "Threshold", "ReLU", "RReLU", "Hardtanh", "ReLU6",
        "Sigmoid", "Hardsigmoid", "Tanh", "SiLU", "Mish", "Hardswish", "ELU",
        "CELU", "SELU", "GLU", "GELU", "Hardshrink", "LeakyReLU",
        "LogSigmoid", "Softplus", "Softshrink", "MultiheadAttention", "PReLU",
        "Softsign", "Tanhshrink", "Softmin", "Softmax", "Softmax2d", "LogSoftmax",
    ]

    SUPPORTED_WAVELETS = [
        "morlet", "mexican_hat", "haar", "db1", "db2", "sym2", "coif1",
        "bior1.3", "bior1.5", "rbio1.3", "rbio1.5", "dmey", "cmor",
    ]

    def __init__(self, size_input, size_hidden, size_output, wavelet_fn="morlet", act_output=None, seed=None):
        """
        Initialize a customizable multi-layer perceptron (MLP) model.
        """
        super(CustomWNN, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Ensure hidden_layers is a int
        self.size_hidden = size_hidden

        # Determine activation for the output layer based on the task
        if act_output is None or act_output == "None":
            self.act_out = nn.Identity()
        elif act_output == "Softmax":
            self.act_out = nn.Softmax(dim=1)
        else:
            self.act_out = getattr(nn.modules.activation, act_output)()

        self.wavelet_fn = getattr(wf, wavelet_fn) if wavelet_fn in self.SUPPORTED_WAVELETS else None
        self.wavelet = WaveletLayer(size_input, size_hidden, self.wavelet_fn)
        self.linear = nn.Linear(size_input * size_hidden, size_output)

    def forward(self, x):
        """
        Forward pass through the MLP model.

        Parameters:
            - x (torch.Tensor): The input tensor.

        Returns:
            - torch.Tensor: The output of the MLP model.
        """
        z = self.wavelet(x)  # apply wavelet transformation
        z = self.act_out(z)  # apply activation function
        return self.linear(z) # linear output layer

    def set_weights(self, solution):
        """
        Set network weights based on a given solution vector.

        Parameters:
            - solution (np.ndarray): A flat array of weights to set in the model.
        """
        with torch.no_grad():
            idx = 0
            for param in self.network.parameters():
                param_size = param.numel()
                # Ensure dtype and device consistency
                param.copy_(torch.tensor(solution[idx:idx + param_size], dtype=param.dtype, device=param.device).view(param.shape))
                idx += param_size

    def get_weights(self):
        """
        Retrieve network weights as a flattened array.

        Returns:
            - np.ndarray: Flattened array of the model's weights.
        """
        return np.concatenate([param.data.cpu().numpy().flatten() for param in self.network.parameters()])

    def get_weights_size(self):
        """
        Calculate the total number of trainable parameters in the model.

        Returns:
            - int: Total number of parameters.
        """
        return sum(param.numel() for param in self.parameters() if param.requires_grad)
