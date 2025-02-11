import torch.nn as nn

class ComplexActivation(nn.Module):
    def __init__(self, activation=nn.SELU()):
        super(ComplexActivation, self).__init__()
        self.activation = activation

    def forward(self, x):
        x_real, x_imag = x
        return self.activation(x_real), self.activation(x_imag)
