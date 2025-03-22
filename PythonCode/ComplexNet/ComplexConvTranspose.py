import torch
import torch.nn as nn

class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        super(ComplexConvTranspose2d, self).__init__()
        # Convoluciones traspuestas reales para cada parte
        self.deconv_real = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
                                              stride, padding, output_padding, groups, bias, dilation)
        self.deconv_imag = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
                                              stride, padding, output_padding, groups, bias, dilation)
    def forward(self, input_real, input_imag):
        # Salida real = deconv_real(x_real) - deconv_imag(x_imag)
        # Salida imag = deconv_real(x_imag) + deconv_imag(x_real)
        real = self.deconv_real(input_real) - self.deconv_imag(input_imag)
        imag = self.deconv_real(input_imag) + self.deconv_imag(input_real)
        return real, imag