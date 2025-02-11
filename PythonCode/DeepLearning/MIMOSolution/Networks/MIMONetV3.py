import torch.nn as nn
import torch
from ComplexBlocks.ComplexConv1d import ComplexConv1d
from ComplexBlocks.ComplexBatchNorm1d import ComplexBatchNorm1d
from ComplexBlocks.ComplexActivation import ComplexActivation
from ComplexBlocks.ComplexIdentity import ComplexIdentity
from ComplexBlocks.ComplexAdaptiveAvgPool1d import ComplexAdaptiveAvgPool1d

class MIMONetV3(nn.Module):
    def __init__(self, input_length=4, num_classes=4, in_channels=1, 
                 conv1_channels=32, conv2_channels=64, conv3_channels=128, fc_hidden=128, dropout_rate=0.3):
        """
        in_channels: número de canales complejos de entrada (por ejemplo, 1 canal complejo = 2 canales reales: real e imaginaria)
        input_length: longitud del vector de entrada (por canal)
        """
        super(MIMONetV3, self).__init__()
        
        # Bloque 1
        self.conv1 = ComplexConv1d(in_channels, conv1_channels, kernel_size=3, padding=1)
        self.bn1 = ComplexBatchNorm1d(conv1_channels)
        self.act1 = ComplexActivation(nn.SELU())
        
        # Bloque 2
        self.conv2 = ComplexConv1d(conv1_channels, conv2_channels, kernel_size=3, padding=1)
        self.bn2 = ComplexBatchNorm1d(conv2_channels)
        self.act2 = ComplexActivation(nn.SELU())
        
        # Bloque 3
        self.conv3 = ComplexConv1d(conv2_channels, conv3_channels, kernel_size=3, padding=1)
        self.bn3 = ComplexBatchNorm1d(conv3_channels)
        self.act3 = ComplexActivation(nn.SELU())
        
        # Conexión residual: en este ejemplo suponemos que la salida de conv3 ya tiene conv3_channels;
        # si fuera necesaria una adaptación de dimensiones se podría usar una ComplexConv1d con kernel_size=1.
        self.residual = ComplexIdentity()
        
        # Global Average Pooling
        self.global_avg_pool = ComplexAdaptiveAvgPool1d(1)
        
        # Fully-connected:
        # Después del pooling cada parte (real e imaginaria) tendrá forma (N, conv3_channels, 1).
        # Se eliminan las dimensiones finales y se concatenan para formar un vector real de dimensión 2*conv3_channels.
        self.fc1 = nn.Linear(conv3_channels * 2, fc_hidden)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def forward(self, x):
        """
        Se asume que la entrada x es una tupla: (x_real, x_imag),
        cada uno con forma (N, in_channels, input_length).
        """
        # Bloque 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        # Bloque 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        # Bloque 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        
        # Conexión residual
        res = self.residual(x)
        # Suma de números complejos: se suman las partes reales y las imaginarias por separado.
        x = (x[0] + res[0], x[1] + res[1])
        
        # Global Average Pooling: cada parte pasa a tener forma (N, conv3_channels, 1)
        x = self.global_avg_pool(x)
        # Eliminar la dimensión final
        x = (x[0].squeeze(-1), x[1].squeeze(-1))  # ahora cada tensor es de forma (N, conv3_channels)
        
        # Convertir a representación real concatenando las partes real e imaginaria.
        x_cat = torch.cat(x, dim=1)  # forma (N, 2 * conv3_channels)
        
        # Capas Fully-connected con dropout
        x_fc = self.fc1(x_cat)
        x_fc = self.dropout(x_fc)
        x_fc = self.fc2(x_fc)
        
        return x_fc
    
# Ejemplo:  batch de 10 muestras, 1 canal complejo (es decir, cada parte es 1 canal) y longitud 4
x_real = torch.randn(10, 1, 4)
x_imag = torch.randn(10, 1, 4)
x_complex = (x_real, x_imag)

model = MIMONetV3(input_length=4, num_classes=4, in_channels=1,
                   conv1_channels=32, conv2_channels=64, conv3_channels=128, fc_hidden=128, dropout_rate=0.3)

output = model(x_complex)
print(output.shape)  # Debería ser (10, num_classes)