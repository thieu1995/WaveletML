#!/usr/bin/env python
# Created by "Thieu" at 10:19, 23/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

class WaveletAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, wavelet_fn: Callable[[torch.Tensor], torch.Tensor], hidden_dim: int = 64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_fn = wavelet_fn
        self.hidden_dim = hidden_dim

        # Learnable parameters for wavelet transform
        self.a = nn.Parameter(torch.randn(out_features, in_features))
        self.b = nn.Parameter(torch.randn(out_features, in_features))

        # Attention projections: Q, K, V
        self.q_proj = nn.Linear(in_features, hidden_dim)
        self.k_proj = nn.Linear(in_features, hidden_dim)
        self.v_proj = nn.Linear(in_features, hidden_dim)

        # Final output projection
        self.out_proj = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        # x: (batch_size, in_features)
        x_exp = x.unsqueeze(1)  # (batch_size, 1, in_features)
        z = (x_exp - self.b) / (self.a.abs() + 1e-6)  # (batch_size, out_features, in_features)
        psi = self.wavelet_fn(z)  # Apply wavelet: (batch_size, out_features, in_features)

        # Average over out_features to get context input to Q/K/V
        wavelet_summary = psi.mean(dim=1)  # (batch_size, in_features)

        # Compute Q, K, V
        Q = self.q_proj(wavelet_summary)  # (batch_size, hidden_dim)
        K = self.k_proj(wavelet_summary)  # (batch_size, hidden_dim)
        V = self.v_proj(wavelet_summary)  # (batch_size, hidden_dim)

        # Scaled Dot-Product Attention (Self-attention per sample)
        attn_scores = (Q * K).sum(dim=-1, keepdim=True) / (self.hidden_dim**0.5)  # (batch_size, 1)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, 1)

        # Weighted sum over V
        attended = attn_weights * V  # (batch_size, hidden_dim)

        # Output projection
        out = self.out_proj(attended)  # (batch_size, out_features)
        return out


# Example wavelet function
def morlet_wavelet(x, w=5.0):
    return torch.exp(-0.5 * x**2) * torch.cos(w * x)

# Example usage
if __name__ == "__main__":
    batch_size, in_dim, out_dim = 2, 5, 3
    layer = WaveletAttentionLayer(in_features=in_dim, out_features=out_dim, wavelet_fn=morlet_wavelet)
    x = torch.randn(batch_size, in_dim)
    out = layer(x)
    print(out.shape)  # should be (batch_size, out_dim)

    for k, v in layer.named_parameters():
        print(f"{k}: {v.shape}, {v.data}")