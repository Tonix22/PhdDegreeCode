import torch.nn as nn

class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        """
        in_channels: número de canales complejos de entrada (cada uno representado por su parte real e imaginaria).
        out_channels: número de canales complejos de salida.
        """
        super(ComplexConv1d, self).__init__()
        # Las convoluciones operan sobre la parte real de cada entrada
        self.real_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.imag_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
    
    def forward(self, x):
        # Se asume que x es una tupla: (x_real, x_imag)
        x_real, x_imag = x
        real_out = self.real_conv(x_real) - self.imag_conv(x_imag)
        imag_out = self.real_conv(x_imag) + self.imag_conv(x_real)
        return real_out, imag_out
