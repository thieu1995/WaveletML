#!/usr/bin/env python
# Created by "Thieu" at 03:57, 19/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        return self.wavelet_fn(u)  # apply wavelet function element-wise


class WNNBase(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, wavelet_fn):
        super(WNNBase, self).__init__()
        self.wavelet = WaveletLayer(in_features, hidden_features, wavelet_fn)
        self.linear = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        z = self.wavelet(x)  # apply wavelet transformation
        z = torch.relu(z)    # apply nonlinearity
        return self.linear(z)  # linear output layer

