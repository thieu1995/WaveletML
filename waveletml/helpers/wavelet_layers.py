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
    In this custom version, each hidden neuron has 1 center and 1 scale.
    The weights are learnable parameters that connect the input to the hidden layer.

    Mỗi neuron tính tổng có trọng số đầu vào (wx), sau đó chuẩn hóa qua center/scale rồi mới áp dụng wavelet.
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
        # x shape: (batch_size, input_dim)
        wx = F.linear(x, self.weights)  # shape: (batch_size, num_neurons)
        z = (wx - self.centers) / self.scales  # Apply dilation and translation
        return self.wavelet_fn(z)  # Apply wavelet activation


class WaveletProductLayer(nn.Module):
    """
    In this custom version, each hidden neuron has d centers and d scales (d = input_dim).
    Each hidden neuron is product of d wavelets

    Mỗi neuron xử lý từng chiều độc lập bằng wavelet rồi nhân lại → sản phẩm của wavelet theo từng chiều.
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
        # x shape: (batch_size, input_dim)
        batch_size = x.size(0)
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
    In this custom version, each hidden neuron has d centers and d scales (d = input_dim).
    Each hidden neuron is summed of d wavelets

    Mỗi neuron xử lý từng chiều độc lập bằng wavelet rồi cộng lại → sản phẩm của wavelet theo từng chiều.
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
        # x shape: (batch_size, input_dim)
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
    In this custom version, each hidden neuron has d centers and d scales (d = input_dim).
    The output of each hidden neuron is forming a new feature space. Wavelet-based feature expansion.

    Lớp này không gom các giá trị lại, mà trả về toàn bộ các biến wavelet đã chuẩn hóa → mở rộng không gian đặc trưng.
    """
    def __init__(self, in_features, out_features, wavelet_fn):
        super(WaveletExpansionLayer, self).__init__()
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
