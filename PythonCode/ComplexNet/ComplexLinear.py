import torch
import torch.nn as nn

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ComplexLinear, self).__init__()
        # Pesos para parte real e imaginaria
        self.fc_real = nn.Linear(in_features, out_features, bias=bias)
        self.fc_imag = nn.Linear(in_features, out_features, bias=bias)
    def forward(self, input_real, input_imag):
        # Salida real = W_real*x_real - W_imag*x_imag 
        # Salida imag = W_real*x_imag + W_imag*x_real 
        # (donde W_real e W_imag son los pesos de cada parte)
        real = self.fc_real(input_real) - self.fc_imag(input_imag)
        imag = self.fc_real(input_imag) + self.fc_imag(input_real)
        return real, imag