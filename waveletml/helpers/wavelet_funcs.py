#!/usr/bin/env python
# Created by "Thieu" at 03:52, 19/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import torch
import torch.nn.functional as F


def morlet(x):
    """
    Computes the Morlet wavelet function.

    Parameters:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor after applying the Morlet wavelet function.
    """
    return torch.cos(1.75 * x) * torch.exp(-0.5 * x**2)


def mexican_hat(x):
    """
    Computes the Mexican Hat wavelet function.

    Parameters:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor after applying the Mexican Hat wavelet function.
    """
    return (1 - x**2) * torch.exp(-0.5 * x**2)


def haar(x):
    """
    Computes the Haar wavelet function.

    Parameters:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor after applying the Haar wavelet function.
    """
    return torch.where((x >= -0.5) & (x < 0.5), torch.ones_like(x), torch.zeros_like(x))
