import torch
import torch.nn as nn
import torch.nn.functional as F

# Definir un Bloque Residual con LayerNorm
# Definir un Bloque Residual con LayerNorm y Fully Connected Layers
class ResidualBlockLayerNormFC(nn.Module):
    def __init__(self, channels, length, kernel_size=3, padding=1, fc_hidden_dim=64):
        """
        Args:
            channels (int): Número de canales en la entrada y salida de las capas Conv1D.
            length (int): Longitud fija de las secuencias.
            kernel_size (int): Tamaño del kernel para las capas Conv1D.
            padding (int): Padding para las capas Conv1D.
            fc_hidden_dim (int): Dimensión oculta para las capas FC.
        """
        super(ResidualBlockLayerNormFC, self).__init__()
        
        # Primera capa Conv1D
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.layer_norm1 = nn.LayerNorm([channels, length])
        self.relu = nn.LeakyReLU()
        
        # Segunda capa Conv1D
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.layer_norm2 = nn.LayerNorm([channels, length])
        
        # Capas completamente conectadas
        self.fc1 = nn.Linear(channels, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, channels)
        self.relu_fc = nn.LeakyReLU()
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Tensor de entrada con forma (batch, channels, length)
        
        Returns:
            torch.Tensor: Salida del bloque residual con forma (batch, channels, length)
        """
        residual = x  # Guardar la entrada para la conexión residual
        
        # Primera Conv1D -> LayerNorm -> ReLU
        out = self.conv1(x)  # (batch, channels, length)
        out = self.layer_norm1(out)
        out = self.relu(out)
        
        # Segunda Conv1D -> LayerNorm
        out = self.conv2(out)  # (batch, channels, length)
        out = self.layer_norm2(out)
        
        # Integración de Capas FC
        # Permutar para tener (batch, length, channels)
        out = out.permute(0, 2, 1)  # (batch, length, channels)
        
        # Aplicar FC1 -> ReLU -> FC2
        out = self.fc1(out)         # (batch, length, fc_hidden_dim)
        out = self.relu_fc(out)
        out = self.fc2(out)         # (batch, length, channels)
        
        # Permutar de nuevo a (batch, channels, length)
        out = out.permute(0, 2, 1)  # (batch, channels, length)
        
        # Añadir la conexión residual
        out += residual              # (batch, channels, length)
        out = self.relu(out)         # Aplicar activación final
        
        return out

# Definir la red PhaseEqualizerConv1D con Skip Connections y LayerNorm
class PhaseEqualizerConv1D(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_residual_blocks=5):
        super(PhaseEqualizerConv1D, self).__init__()
        
        self.length = input_channels  # Longitud fija de las secuencias
        
        # Encoder
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=input_channels, kernel_size=3, padding=1)
        self.layer_norm1 = nn.LayerNorm([input_channels, self.length])
        self.relu1 = nn.LeakyReLU()
        
        self.conv2 = nn.Conv1d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=3, padding=1)
        self.layer_norm2 = nn.LayerNorm([hidden_channels, self.length])
        self.relu2 = nn.LeakyReLU()
        
        # Bloques Residuales
        self.residual_blocks = nn.Sequential(
            *[ResidualBlockLayerNormFC(hidden_channels, self.length) for _ in range(num_residual_blocks)]
        )
        
        # Decoder
        self.deconv1 = nn.Conv1d(in_channels=hidden_channels, out_channels=input_channels, kernel_size=3, padding=1)
        self.layer_norm3 = nn.LayerNorm([input_channels, self.length])
        self.relu3 = nn.LeakyReLU()
        
        self.deconv2 = nn.Conv1d(in_channels=input_channels, out_channels=1, kernel_size=3, padding=1)
        self.layer_norm4 = nn.LayerNorm([1, self.length])
        self.relu4 = nn.LeakyReLU()
    
    def forward(self, phase):
        """
        Args:
            phase (torch.Tensor): Tensor de entrada con forma (batch, N)
        
        Returns:
            torch.Tensor: Compensación de fase con forma (batch, N)
        """
        # Agregar dimensión de canal
        x = phase.unsqueeze(1)  # De (batch, N) a (batch, 1, N)
        
        # Encoder
        enc1 = self.conv1(x)       # (batch, input_channels, N)
        enc1 = self.layer_norm1(enc1)
        enc1 = self.relu1(enc1)
        
        enc2 = self.conv2(enc1)    # (batch, hidden_channels, N)
        enc2 = self.layer_norm2(enc2)
        enc2 = self.relu2(enc2)
        
        # Bloques Residuales
        res = self.residual_blocks(enc2)  # (batch, hidden_channels, N)
        
        # Decoder
        dec1 = self.deconv1(res)    # (batch, input_channels, N)
        dec1 = self.layer_norm3(dec1)
        dec1 = self.relu3(dec1)
        
        dec2 = self.deconv2(dec1)   # (batch, 1, N)
        dec2 = self.layer_norm4(dec2)
        dec2 = self.relu4(dec2)
        
        # Compensar la fase restando el ruido estimado
        phase_noise = dec2.squeeze(1)  # De (batch, 1, N) a (batch, N)
        
        return phase_noise
