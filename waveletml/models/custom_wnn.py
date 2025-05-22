#!/usr/bin/env python
# Created by "Thieu" at 15:45, 22/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import torch
import torch.nn as nn
import torch.nn.functional as F
from waveletml.helpers import wavelet_funcs as wf


# --- Wavelet Neuron Layer ---
class WaveletLayer01(nn.Module):
    """
    In this custom version, each hidden neuron has 1 center and 1 scale.
    The weights are learnable parameters that connect the input to the hidden layer.
    """
    def __init__(self, input_dim, num_neurons, wavelet_fn):
        super(WaveletLayer01, self).__init__()
        self.input_dim = input_dim
        self.num_neurons = num_neurons
        self.wavelet_fn = wavelet_fn  # a callable function

        # Learnable weights for each neuron (input -> hidden)
        self.weights = nn.Parameter(torch.randn(num_neurons, input_dim))
        self.centers = nn.Parameter(torch.randn(num_neurons))      # translation (b)
        self.scales = nn.Parameter(torch.ones(num_neurons))      # dilation (a)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        wx = F.linear(x, self.weights)  # shape: (batch_size, num_neurons)
        z = (wx - self.center) / self.scale  # Apply dilation and translation
        return self.wavelet_fn(z)  # Apply wavelet activation


# --- Wavelet Neural Network ---
class CustomWNN01(nn.Module):
    """
    In this version, we calculate the sum of all inputs to each neuron (wx)
    The we calculate z = (wx - b) / a, where wx is the weighted sum of inputs.
    Then we apply the wavelet function to the z value.
    The output layer is a standard linear layer. (weights and bias)

    The number of parameters for WNN(3, 5, 1) is:
    3 * 5 (weights) + 5 (centers) + 5 (scales) + 5 (weights at output layer) + 1 (bias ouput) = 3 * 5 + 5 + 5 + 1 = 26
    """
    SUPPORTED_WAVELETS = [
        "morlet", "mexican_hat", "haar", "db1", "db2", "sym2", "coif1",
        "bior1.3", "bior1.5", "rbio1.3", "rbio1.5", "dmey", "cmor",
    ]
    def __init__(self, input_dim, hidden_dim, output_dim, wavelet_fn):
        super(CustomWNN01, self).__init__()
        self.wavelet_fn = getattr(wf, wavelet_fn) if wavelet_fn in self.SUPPORTED_WAVELETS else None
        self.wavelet_layer = WaveletLayer01(input_dim, hidden_dim, self.wavelet_fn)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.wavelet_layer(x)
        output = self.output_layer(x)
        return output


# Wavelet Layer (each neuron is product of d wavelets)
class WaveletLayer02(nn.Module):
    """
    In this custom version, each hidden neuron has d centers and d scales (d = input_dim).
    """
    def __init__(self, input_dim, num_neurons, wavelet_fn):
        super(WaveletLayer02, self).__init__()
        self.input_dim = input_dim
        self.num_neurons = num_neurons
        self.wavelet_fn = wavelet_fn  # a callable function

        # Learnable parameters per (neuron, input)
        self.bias = nn.Parameter(torch.randn(num_neurons, input_dim))   # b_ji
        self.scale = nn.Parameter(torch.ones(num_neurons, input_dim))   # a_ji

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        batch_size = x.size(0)
        x_expanded = x.unsqueeze(1)  # (batch_size, 1, input_dim)

        # broadcast: (batch_size, num_neurons, input_dim)
        z = (x_expanded - self.bias) / self.scale

        # Apply wavelet function: shape (batch_size, num_neurons, input_dim)
        psi = self.wavelet_fn(z)

        # Product over input dimensions to get hidden unit output
        h = psi.prod(dim=2)  # shape: (batch_size, num_neurons)

        return h


# Full Wavelet Neural Network
class CustomWNN02(nn.Module):
    """
    In this version, we calculate the z value for each input dimension, then we apply the wavelet function to each z value.
    The output of each hidden neuron is the product of all wavelet functions from input to that hidden neuron.
    The output layer is a standard linear layer. (weights and bias).

    The number of parameters for WNN(3, 5, 1) is:
    5*3 (centers) + 5*3 (scales) + 5 (weights at output layer) + 1 (bias ouput) = 5*3 + 5*3 + 5 + 1 = 36
    """
    SUPPORTED_WAVELETS = [
        "morlet", "mexican_hat", "haar", "db1", "db2", "sym2", "coif1",
        "bior1.3", "bior1.5", "rbio1.3", "rbio1.5", "dmey", "cmor",
    ]
    def __init__(self, input_dim, hidden_dim, output_dim, wavelet_fn):
        super(CustomWNN02, self).__init__()
        self.wavelet_fn = getattr(wf, wavelet_fn) if wavelet_fn in self.SUPPORTED_WAVELETS else None
        self.wavelet_layer = WaveletLayer02(input_dim, hidden_dim, self.wavelet_fn)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = self.wavelet_layer(x)
        y = self.output_layer(h)
        return y



