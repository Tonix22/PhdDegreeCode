import torch.nn as nn

class ComplexAdaptiveAvgPool1d(nn.Module):
    def __init__(self, output_size):
        super(ComplexAdaptiveAvgPool1d, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(output_size)
    
    def forward(self, x):
        x_real, x_imag = x
        return self.pool(x_real), self.pool(x_imag)
