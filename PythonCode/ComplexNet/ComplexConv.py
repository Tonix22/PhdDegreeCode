import torch
import torch.nn as nn
class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        # Convoluciones reales paralelas para real e imaginario
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, 
                                   stride, padding, dilation, groups, bias)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, 
                                   stride, padding, dilation, groups, bias)
    def forward(self, input_real, input_imag):
        # Salida real = conv_real(x_real) - conv_imag(x_imag)
        # Salida imag = conv_real(x_imag) + conv_imag(x_real)
        real = self.conv_real(input_real) - self.conv_imag(input_imag)
        imag = self.conv_real(input_imag) + self.conv_imag(input_real)
        return real, imag