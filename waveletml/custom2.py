#!/usr/bin/env python
# Created by "Thieu" at 16:03, 22/05/2025 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import torch
import torch.nn as nn

# Mexican Hat Wavelet Function
def mexican_hat(x):
    return (1 - x**2) * torch.exp(-x**2 / 2)

# Wavelet Layer (each neuron is product of d wavelets)
class WaveletNeuronLayer(nn.Module):
    def __init__(self, input_dim, num_neurons):
        super(WaveletNeuronLayer, self).__init__()
        self.input_dim = input_dim
        self.num_neurons = num_neurons

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
        psi = mexican_hat(z)

        # Product over input dimensions to get hidden unit output
        h = psi.prod(dim=2)  # shape: (batch_size, num_neurons)

        return h

# Full Wavelet Neural Network
class WaveletNeuralNetwork(nn.Module):
    def __init__(self, input_dim, num_neurons, output_dim):
        super(WaveletNeuralNetwork, self).__init__()
        self.wavelet_layer = WaveletNeuronLayer(input_dim, num_neurons)
        self.output_layer = nn.Linear(num_neurons, output_dim)

    def forward(self, x):
        h = self.wavelet_layer(x)
        y = self.output_layer(h)
        return y

# Example: Using the WNN on simple data
if __name__ == "__main__":
    model = WaveletNeuralNetwork(input_dim=2, num_neurons=10, output_dim=1)

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

# if __name__ == "__main__":
#     # Sample 1D regression
#     model = WaveletNeuralNetwork(input_dim=2, num_neurons=10, output_dim=1)
#     x = torch.randn(32, 2)  # Batch of 32 samples, each with 2 features
#     y = model(x)
#     print(y.shape)  # Should print: torch.Size([32, 1])
