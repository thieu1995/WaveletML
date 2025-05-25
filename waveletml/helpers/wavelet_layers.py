#!/usr/bin/env python
# Created by "Thieu" at 09:14, 23/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveletWeightedLinearLayer(nn.Module):
    """
    A custom linear layer where each hidden neuron has a center and scale.
    The weights are learnable parameters connecting the input to the hidden layer.
    Each hidden neuron has an input (wx), then transform using center and scale before applying the wavelet function.

    Attributes:
        input_dim (int): The dimensionality of the input.
        num_neurons (int): The number of neurons in the hidden layer.
        wavelet_fn (callable): The wavelet function to apply.
        weights (torch.nn.Parameter): Learnable weights for the input to hidden connections.
        centers (torch.nn.Parameter): Learnable centers for each neuron.
        scales (torch.nn.Parameter): Learnable scales for each neuron.

    Methods:
        forward(x): Computes the forward pass of the layer.
    """

    def __init__(self, input_dim, num_neurons, wavelet_fn):
        super(WaveletWeightedLinearLayer, self).__init__()
        self.input_dim = input_dim
        self.num_neurons = num_neurons
        self.wavelet_fn = wavelet_fn  # a callable function

        # Learnable weights for each neuron (input -> hidden)
        self.weights = nn.Parameter(torch.randn(num_neurons, input_dim))
        self.centers = nn.Parameter(torch.randn(num_neurons))      # translation (b)
        self.scales = nn.Parameter(torch.ones(num_neurons))      # dilation (a)

    def forward(self, x):
        """
        Performs the forward pass of the layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor after applying the wavelet function.
        """
        wx = F.linear(x, self.weights)  # shape: (batch_size, num_neurons)
        z = (wx - self.centers) / self.scales  # Apply dilation and translation
        return self.wavelet_fn(z)  # Apply wavelet activation


class WaveletProductLayer(nn.Module):
    """
    A custom layer where each hidden neuron is the product of wavelets applied to each input dimension.
    Each hidden neuron has d centers and d scales (d = input_dim).
    And the output of each hidden neuron is the product of d wavelets.

    Attributes:
        input_dim (int): The dimensionality of the input.
        num_neurons (int): The number of neurons in the hidden layer.
        wavelet_fn (callable): The wavelet function to apply.
        centers (torch.nn.Parameter): Learnable centers for each neuron and input dimension.
        scales (torch.nn.Parameter): Learnable scales for each neuron and input dimension.

    Methods:
        forward(x): Computes the forward pass of the layer.
    """

    def __init__(self, input_dim, num_neurons, wavelet_fn):
        super(WaveletProductLayer, self).__init__()
        self.input_dim = input_dim
        self.num_neurons = num_neurons
        self.wavelet_fn = wavelet_fn  # a callable function

        # Learnable parameters per (neuron, input)
        self.centers = nn.Parameter(torch.randn(num_neurons, input_dim))   # b_ji
        self.scales = nn.Parameter(torch.ones(num_neurons, input_dim))   # a_ji

    def forward(self, x):
        """
        Performs the forward pass of the layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor after applying the wavelet function and taking the product.
        """
        x_expanded = x.unsqueeze(1)  # (batch_size, 1, input_dim)

        # broadcast: (batch_size, num_neurons, input_dim)
        z = (x_expanded - self.centers) / self.scales

        # Apply wavelet function: shape (batch_size, num_neurons, input_dim)
        psi = self.wavelet_fn(z)

        # Product over input dimensions to get hidden unit output
        h = psi.prod(dim=2)  # shape: (batch_size, num_neurons)
        return h


class WaveletSummationLayer(nn.Module):
    """
    A custom layer where each hidden neuron is the sum of wavelets applied to each input dimension.
    Each hidden neuron has d centers and d scales (d = input_dim).
    And the output of each hidden neuron is the sum of d wavelets.

    Attributes:
        input_dim (int): The dimensionality of the input.
        num_neurons (int): The number of neurons in the hidden layer.
        wavelet_fn (callable): The wavelet function to apply.
        centers (torch.nn.Parameter): Learnable centers for each neuron and input dimension.
        scales (torch.nn.Parameter): Learnable scales for each neuron and input dimension.

    Methods:
        forward(x): Computes the forward pass of the layer.
    """

    def __init__(self, input_dim, num_neurons, wavelet_fn):
        super(WaveletSummationLayer, self).__init__()
        self.input_dim = input_dim
        self.num_neurons = num_neurons
        self.wavelet_fn = wavelet_fn  # a callable function

        # Learnable parameters per (neuron, input)
        self.centers = nn.Parameter(torch.randn(num_neurons, input_dim))   # b_ji
        self.scales = nn.Parameter(torch.ones(num_neurons, input_dim))   # a_ji

    def forward(self, x):
        """
        Performs the forward pass of the layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor after applying the wavelet function and taking the sum.
        """
        batch_size = x.size(0)
        x_expanded = x.unsqueeze(1)  # (batch_size, 1, input_dim)

        # broadcast: (batch_size, num_neurons, input_dim)
        z = (x_expanded - self.centers) / self.scales

        # Apply wavelet function: shape (batch_size, num_neurons, input_dim)
        psi = self.wavelet_fn(z)

        # Product over input dimensions to get hidden unit output
        h = psi.sum(dim=2)  # shape: (batch_size, num_neurons)
        return h


class WaveletExpansionLayer(nn.Module):
    """
    A custom layer for wavelet-based feature expansion, where each hidden neuron outputs a new feature space.
    The output of each hidden neuron is forming a new feature space. Wavelet-based feature expansion.

    Attributes:
        in_features (int): The dimensionality of the input.
        out_features (int): The number of output features.
        wavelet_fn (callable): The wavelet function to apply.
        centers (torch.nn.Parameter): Learnable centers for each output feature and input dimension.
        scales (torch.nn.Parameter): Learnable scales for each output feature and input dimension.

    Methods:
        forward(x): Computes the forward pass of the layer.
    """

    def __init__(self, in_features, out_features, wavelet_fn):
        super(WaveletExpansionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_fn = wavelet_fn  # a callable function

        self.centers = nn.Parameter(torch.randn(out_features, in_features))
        self.scales = nn.Parameter(torch.ones(out_features, in_features))

    def forward(self, x):
        """
        Performs the forward pass of the layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor after applying the wavelet function, expanded to a new feature space.
        """
        x = x.unsqueeze(1)  # [batch_size, 1, in_features]
        centers = self.centers.unsqueeze(0)  # [1, out_features, in_features]
        scales = self.scales.unsqueeze(0)  # [1, out_features, in_features]

        u = (x - centers) / scales  # broadcasting
        # apply wavelet function element-wise
        wave = self.wavelet_fn(u)
        return wave.view(x.shape[0], -1)  # Flatten to [batch_size, out_features * in_features]
