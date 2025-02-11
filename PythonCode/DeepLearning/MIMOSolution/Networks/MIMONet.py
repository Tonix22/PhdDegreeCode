import torch
import torch.nn as nn

class MIMONet(nn.Module):
    def __init__(self, input_length=3, num_classes=4, in_channels=1, 
                 conv1_channels=32, conv2_channels=64, fc_hidden=64):
        """
        Parámetros:
          - input_length: Longitud de la secuencia de entrada.
          - num_classes: Número de clases a predecir.
          - in_channels: Número de canales de entrada (por defecto 1).
          - conv1_channels: Número de filtros en la primera capa convolucional.
          - conv2_channels: Número de filtros en la segunda capa convolucional.
          - fc_hidden: Número de unidades en la capa oculta fully-connected.
        """
        super(MIMONet, self).__init__()
        
        # Bloque 1: Primera convolución + ReLU + BatchNorm1d
        self.conv1 = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=conv1_channels, 
            kernel_size=2, 
            padding=1  # Padding simétrico para mantener la dimensión temporal.
        )
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(conv1_channels)
        
        # Bloque 2: Segunda convolución + ReLU + BatchNorm1d
        self.conv2 = nn.Conv1d(
            in_channels=conv1_channels, 
            out_channels=conv2_channels, 
            kernel_size=2, 
            padding=2  # Padding simétrico.
        )
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(conv2_channels)
        
        # Global Average Pooling: reduce la dimensión temporal a 1.
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Capas fully-connected
        self.fc1 = nn.Linear(conv2_channels, fc_hidden)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden, num_classes)
        
        # Nota: No incluimos softmax final. Cuando se utilice CrossEntropyLoss,
        # ésta espera logits y aplica log-softmax internamente.
    
    def forward(self, x):
        """
        Parámetro:
          - x: Tensor de entrada con forma (batch_size, in_channels, input_length)
        """
        # Bloque 1: Conv1d -> ReLU -> BatchNorm1d
        x = self.conv1(x)      # (N, conv1_channels, input_length)
        x = self.relu1(x)
        x = self.bn1(x)
        
        # Bloque 2: Conv1d -> ReLU -> BatchNorm1d
        x = self.conv2(x)      # (N, conv2_channels, input_length)
        x = self.relu2(x)
        x = self.bn2(x)
        
        # Global Average Pooling: reduce la dimensión temporal a 1.
        x = self.global_avg_pool(x)  # (N, conv2_channels, 1)
        x = x.squeeze(-1)            # (N, conv2_channels)
        
        # Capas fully-connected
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)              # (N, num_classes)
        
        return x