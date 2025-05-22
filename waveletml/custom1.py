#!/usr/bin/env python
# Created by "Thieu" at 15:41, 22/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Wavelet Activation Function: Mexican Hat ---
def mexican_hat(x):
    return (1 - x**2) * torch.exp(-x**2 / 2)


# --- Wavelet Neuron Layer ---
class WaveletNeuronLayer(nn.Module):
    def __init__(self, input_dim, num_neurons):
        super(WaveletNeuronLayer, self).__init__()
        self.input_dim = input_dim
        self.num_neurons = num_neurons

        # Learnable weights for each neuron (input -> hidden)
        self.weights = nn.Parameter(torch.randn(num_neurons, input_dim))
        self.bias = nn.Parameter(torch.randn(num_neurons))      # translation (b)
        self.scale = nn.Parameter(torch.ones(num_neurons))      # dilation (a)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        wx = F.linear(x, self.weights)  # shape: (batch_size, num_neurons)
        z = (wx - self.bias) / self.scale  # Apply dilation and translation
        return mexican_hat(z)  # Apply wavelet activation


# --- Wavelet Neural Network ---
class WaveletNeuralNetwork(nn.Module):
    def __init__(self, input_dim, num_wavelet_neurons, output_dim):
        super(WaveletNeuralNetwork, self).__init__()
        self.wavelet_layer = WaveletNeuronLayer(input_dim, num_wavelet_neurons)
        self.output_layer = nn.Linear(num_wavelet_neurons, output_dim)

    def forward(self, x):
        x = self.wavelet_layer(x)
        output = self.output_layer(x)
        return output

# Example: Using the WNN on simple data
if __name__ == "__main__":
    model = WaveletNeuralNetwork(input_dim=1, num_wavelet_neurons=10, output_dim=1)

    # Dummy input and target
    x = torch.linspace(-2, 2, 100).unsqueeze(1)
    y = torch.sin(3 * x)  # target function

    # Forward pass
    y_pred = model(x)

    # Loss and optimization
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop (example, 500 epochs)
    for epoch in range(500):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    for k, v in model.named_parameters():
        print(f"{k}: {v.shape}, {v.data}")