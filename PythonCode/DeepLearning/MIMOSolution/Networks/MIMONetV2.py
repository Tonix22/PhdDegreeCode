import torch
import torch.nn as nn

class MIMONetV2(nn.Module):
    def __init__(self, num_classes=4, in_channels=1, 
                 conv1_channels=32, conv2_channels=64, conv3_channels=128, fc_hidden=128):
        super(MIMONetV2, self).__init__()
        
        # Bloque 1
        self.conv1 = nn.Conv1d(in_channels, conv1_channels, kernel_size=2, padding=1)
        self.bn1 = nn.BatchNorm1d(conv1_channels)
        self.act1 = nn.ReLU()

        # Bloque 2
        self.conv2 = nn.Conv1d(conv1_channels, conv2_channels, kernel_size=2, padding=1)
        self.bn2 = nn.BatchNorm1d(conv2_channels)
        self.act2 = nn.ReLU()

        # Bloque 3
        self.conv3 = nn.Conv1d(conv2_channels, conv3_channels, kernel_size=2, padding=1)
        self.bn3 = nn.BatchNorm1d(conv3_channels)
        self.act3 = nn.ReLU()

        # Residual Connection (ajuste de dimensiones solo si es necesario)
        self.residual = (
            nn.Conv1d(conv3_channels, conv3_channels, kernel_size=1) 
            if in_channels != conv3_channels else nn.Identity()
        )

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Fully-connected
        self.fc1 = nn.Linear(conv3_channels, fc_hidden)
        self.act4 = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def forward(self, x):
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

        # Residual Connection
        x = x + self.residual(x)  # Suma de residual si aplica

        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)  # (N, conv3_channels)

        # Fully-connected con dropout
        x = self.fc1(x)
        x = self.act4(x)
        x = self.fc2(x)

        return x
