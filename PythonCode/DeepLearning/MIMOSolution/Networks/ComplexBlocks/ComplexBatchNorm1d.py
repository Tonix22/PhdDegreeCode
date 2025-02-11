import torch.nn as nn
class ComplexBatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super(ComplexBatchNorm1d, self).__init__()
        self.bn_real = nn.BatchNorm1d(num_features)
        self.bn_imag = nn.BatchNorm1d(num_features)
    
    def forward(self, x):
        x_real, x_imag = x
        return self.bn_real(x_real), self.bn_imag(x_imag)
