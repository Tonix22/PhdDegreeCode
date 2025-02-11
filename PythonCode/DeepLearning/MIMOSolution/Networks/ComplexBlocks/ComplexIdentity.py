import torch.nn as nn

class ComplexIdentity(nn.Module):
    def forward(self, x):
        return x
