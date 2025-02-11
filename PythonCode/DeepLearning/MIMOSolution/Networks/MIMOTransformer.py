import torch
import torch.nn as nn
import torch.nn.functional as F

class MIMOTransformer(nn.Module):
    def __init__(self, input_length=4, num_classes=4, in_channels=1, embed_dim=64, num_heads=4, num_layers=2, dropout=0.3):
        """
        MIMOTransformer - Versión con Transformer en lugar de CNN.
        
        Parámetros:
          - input_length: Longitud de la secuencia de entrada.
          - num_classes: Número de clases a predecir.
          - in_channels: Número de canales de entrada (dimensión de características en cada paso de tiempo).
          - embed_dim: Dimensión del embedding interno del Transformer.
          - num_heads: Número de cabezas de atención.
          - num_layers: Número de bloques Transformer.
          - dropout: Dropout para regularización.
        """
        super(MIMOTransformer, self).__init__()

        # Proyección de la entrada (de in_channels a embed_dim)
        self.input_projection = nn.Linear(in_channels, embed_dim)

        # Definir el Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Capa fully-connected final
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Parámetro:
          - x: Tensor de entrada con forma (batch_size, in_channels, input_length)
        """
        # Reordenar input a (batch_size, input_length, in_channels)
        x = x.permute(0, 2, 1)

        # Proyectar las características de entrada
        x = self.input_projection(x)  # (batch_size, input_length, embed_dim)

        # Pasar por el Transformer Encoder
        x = self.transformer(x)  # (batch_size, input_length, embed_dim)

        # Tomar el primer token como representación global (o usar pooling)
        x = x.mean(dim=1)  # Promediamos sobre la dimensión temporal

        # Fully-connected final
        x = self.fc(x)

        return x
