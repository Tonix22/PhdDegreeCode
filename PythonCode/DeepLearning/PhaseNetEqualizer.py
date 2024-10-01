
import torch.nn as nn
# Define the ResidualBlock class used in the neural network
class ResidualBlock(nn.Module):
    def __init__(self, size):
        super(ResidualBlock, self).__init__()
        # Linear layer that maps from size to size
        self.linear = nn.Linear(size, size)
        # Activation function
        self.activation = nn.LeakyReLU()
        # Layer normalization to stabilize learning
        self.Norm = nn.LayerNorm(size)

    def forward(self, x):
        residual = x            # Store the input for the residual connection
        out = self.linear(x)    # Apply linear transformation
        out = self.Norm(out)    # Apply layer normalization
        out = self.activation(out)  # Apply activation function
        out += residual         # Add the input (residual connection)
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
        self.noise_estimate = nn.Sequential(*layers)

    def forward(self, x):
        # Subtract the network's output from the input to model phase correction
        return x-self.noise_estimate(x)