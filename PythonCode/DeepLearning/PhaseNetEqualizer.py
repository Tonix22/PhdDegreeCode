
import torch.nn as nn
import torch
# Define the ResidualBlock class used in the neural network
class ResidualBlock(nn.Module):
    def __init__(self, size):
        super(ResidualBlock, self).__init__()
        # Linear layer that maps from size to size
        self.linear = nn.Linear(size, size)
        # Activation function
        self.activation = nn.GELU()
        # Layer normalization to stabilize learning
        self.Norm = nn.LayerNorm(size)

    def forward(self, x):
        residual = x            # Store the input for the residual connection
        out = self.linear(x)    # Apply linear transformation
        out = self.Norm(out)    # Apply layer normalization
        out = self.activation(out)  # Apply activation function
        out -= residual         # Add the input (residual connection)
        return out

# Define the PhaseEqualizer network
class PhaseEqualizer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=5):
        super(PhaseEqualizer, self).__init__()
        # Initialize a list to hold the layers
        layers = [nn.Linear(input_size, hidden_size), nn.LeakyReLU()]
        # Add multiple ResidualBlocks to the network
        for _ in range(num_layers):
            layers.append(ResidualBlock(hidden_size))
        # Add the final linear layer to map back to the input size
        layers.append(nn.Linear(hidden_size, input_size))
        # Combine all layers into a Sequential model
        self.phase_noise_estimate = nn.Sequential(*layers)

    def forward(self, phase):
        # Subtract the network's output from the input to model phase correction
        return  self.phase_noise_estimate(phase)
    
#"""
# Crear un modelo de ejemplo
input_size = 128  # Puedes cambiarlo según tu caso
hidden_size = 256
num_layers = 5
model = PhaseEqualizer(input_size, hidden_size, num_layers)

# Crear una entrada de prueba
dummy_input = torch.randn(1, input_size)

# Guardar el modelo en formato ONNX
onnx_path = "phase_equalizer.onnx"
torch.onnx.export(model, dummy_input, onnx_path, input_names=["input"], output_names=["output"], opset_version=11)

print(f"Modelo guardado en {onnx_path}")
#"""