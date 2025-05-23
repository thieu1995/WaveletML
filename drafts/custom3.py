#!/usr/bin/env python
# Created by "Thieu" at 10:13, 23/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

class WeightedProductWaveletLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, wavelet_fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_fn = wavelet_fn

        # Learnable parameters for each neuron
        self.a = nn.Parameter(torch.randn(out_features, in_features))  # scale
        self.b = nn.Parameter(torch.randn(out_features, in_features))  # shift
        self.w = nn.Parameter(torch.ones(out_features, in_features))   # weight

        # Output linear layer (optional)
        self.linear = nn.Linear(out_features, out_features)

    def forward(self, x):
        # x: shape (batch_size, in_features)
        x = x.unsqueeze(1)  # shape (batch_size, 1, in_features)
        z = (x - self.b) / (self.a.abs() + 1e-6)  # (batch_size, out_features, in_features)
        psi_z = self.wavelet_fn(z)  # Apply wavelet function: same shape

        log_psi = torch.log(psi_z.abs() + 1e-6)  # for numerical stability
        weighted_log = self.w * log_psi
        log_output = weighted_log.sum(dim=-1)  # sum across features

        out = torch.exp(log_output)  # convert back from log-space
        return self.linear(out)

# Example Morlet wavelet
def morlet_wavelet(x, w=5.0):
    return torch.exp(-0.5 * x**2) * torch.cos(w * x)

# Usage example
if __name__ == "__main__":
    batch_size, in_dim, out_dim = 3, 10, 1
    layer = WeightedProductWaveletLayer(in_features=in_dim, out_features=out_dim, wavelet_fn=morlet_wavelet)
    x = torch.randn(batch_size, in_dim)
    out = layer(x)
    print(out.shape)  # should be (batch_size, out_dim)

    for k, v in layer.named_parameters():
        print(f"{k}: {v.shape}, {v.data}")