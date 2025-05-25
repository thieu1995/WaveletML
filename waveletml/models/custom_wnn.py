#!/usr/bin/env python
# Created by "Thieu" at 15:45, 22/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import torch
import torch.nn as nn
from waveletml.helpers import wavelet_funcs as wf
from waveletml.helpers.wavelet_layers import (WaveletWeightedLinearLayer, WaveletProductLayer,
                                              WaveletSummationLayer, WaveletExpansionLayer)


class BaseCustomWNN(nn.Module):
    """
    Base class for custom wavelet neural networks.
    This class is not meant to be used directly.
    """
    SUPPORTED_WAVELETS = [
        "morlet", "mexican_hat", "haar", "db1", "db2", "sym2", "coif1",
        "bior1.3", "bior1.5", "rbio1.3", "rbio1.5", "dmey", "cmor",
    ]
    SUPPORTED_ACTIVATIONS = [
        "Threshold", "ReLU", "RReLU", "Hardtanh", "ReLU6",
        "Sigmoid", "Hardsigmoid", "Tanh", "SiLU", "Mish", "Hardswish", "ELU",
        "CELU", "SELU", "GLU", "GELU", "Hardshrink", "LeakyReLU",
        "LogSigmoid", "Softplus", "Softshrink", "MultiheadAttention", "PReLU",
        "Softsign", "Tanhshrink", "Softmin", "Softmax", "Softmax2d", "LogSoftmax",
    ]

    def __init__(self, input_dim, hidden_dim, output_dim, wavelet_fn="morlet", act_output=None, seed=None):
        super(BaseCustomWNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Determine activation for the output layer based on the task
        if act_output is None or act_output == "None":
            self.act_out = nn.Identity()
        elif act_output == "Softmax":
            self.act_out = nn.Softmax(dim=1)
        else:
            self.act_out = getattr(nn.modules.activation, act_output)()

        self.wavelet_fn = getattr(wf, wavelet_fn) if wavelet_fn in self.SUPPORTED_WAVELETS else None

    def forward(self, x):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def set_weights(self, solution):
        """
        Set network weights based on a given solution vector.

        Parameters:
            - solution (np.ndarray): A flat array of weights to set in the model.
        """
        with torch.no_grad():
            idx = 0
            for param in self.parameters():
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
        return np.concatenate([param.data.cpu().numpy().flatten() for param in self.parameters()])

    def get_weights_size(self):
        """
        Calculate the total number of trainable parameters in the model.

        Returns:
            - int: Total number of parameters.
        """
        return sum(param.numel() for param in self.parameters() if param.requires_grad)


class CustomWaveletWeightedLinearNetwork(BaseCustomWNN):
    """
    In this version, we calculate the sum of all inputs to each neuron (wx)
    The we calculate z = (wx - b) / a, where wx is the weighted sum of inputs.
    Then we apply the wavelet function to the z value.
    The output layer is a standard linear layer. (weights and bias)

    The number of parameters for WNN(3, 5, 1) is:
    3 * 5 (weights) + 5 (centers) + 5 (scales) + 5 (weights at output layer) + 1 (bias ouput) = 3 * 5 + 5 + 5 + 1 = 26
    """
    def __init__(self, input_dim, hidden_dim, output_dim, wavelet_fn="morlet", act_output=None, seed=None):
        super().__init__(input_dim, hidden_dim, output_dim, wavelet_fn, act_output, seed)
        self.wavelet_layer = WaveletWeightedLinearLayer(input_dim, hidden_dim, self.wavelet_fn)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = self.wavelet_layer(x)
        out = self.output_layer(h)
        out = self.act_out(out)
        return out


class CustomWaveletProductNetwork (BaseCustomWNN):
    """
    In this version, we calculate the z value for each input dimension, then we apply the wavelet function to each z value.
    The output of each hidden neuron is the product of all wavelet functions from input to that hidden neuron.
    The output layer is a standard linear layer. (weights and bias).

    The number of parameters for WNN(3, 5, 1) is:
    5*3 (centers) + 5*3 (scales) + 5 (weights at output layer) + 1 (bias ouput) = 5*3 + 5*3 + 5 + 1 = 36
    """
    def __init__(self, input_dim, hidden_dim, output_dim, wavelet_fn="morlet", act_output=None, seed=None):
        super().__init__(input_dim, hidden_dim, output_dim, wavelet_fn, act_output, seed)
        self.wavelet_layer = WaveletProductLayer(input_dim, hidden_dim, self.wavelet_fn)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = self.wavelet_layer(x)
        out = self.output_layer(h)
        out = self.act_out(out)
        return out


class CustomWaveletSummationNetwork (BaseCustomWNN):
    """
    In this version, we calculate the z value for each input dimension, then we apply the wavelet function to each z value.
    The output of each hidden neuron is the sum of all wavelet functions from input to that hidden neuron.
    The output layer is a standard linear layer. (weights and bias).

    The number of parameters for WNN(3, 5, 1) is:
    5*3 (centers) + 5*3 (scales) + 5 (weights at output layer) + 1 (bias ouput) = 5*3 + 5*3 + 5 + 1 = 36
    """
    def __init__(self, input_dim, hidden_dim, output_dim, wavelet_fn="morlet", act_output=None, seed=None):
        super().__init__(input_dim, hidden_dim, output_dim, wavelet_fn, act_output, seed)
        self.wavelet_layer = WaveletSummationLayer(input_dim, hidden_dim, self.wavelet_fn)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = self.wavelet_layer(x)
        out = self.output_layer(h)
        out = self.act_out(out)
        return out


class CustomWaveletExpansionNetwork (BaseCustomWNN):
    """
    In this version, we calculate the z value for each input dimension, then we apply the wavelet function to each z value.
    The output of each hidden neuron is forming a new feature space. Wavelet-based feature expansion.
    The output layer is a standard linear layer. (weights and bias).

    The number of parameters for WNN(3, 5, 1) is:
    5*3 (centers) + 5*3 (scales) + 3*5 (weights at output layer) + 1 (bias ouput) = 5*3 + 5*3 + 3*5 + 1 = 46
    """
    def __init__(self, input_dim, hidden_dim, output_dim, wavelet_fn="morlet", act_output=None, seed=None):
        super().__init__(input_dim, hidden_dim, output_dim, wavelet_fn, act_output, seed)
        self.wavelet_layer = WaveletExpansionLayer(input_dim, hidden_dim, self.wavelet_fn)
        self.output_layer = nn.Linear(input_dim * hidden_dim, output_dim)

    def forward(self, x):
        h = self.wavelet_layer(x)
        out = self.output_layer(h)
        out = self.act_out(out)
        return out
