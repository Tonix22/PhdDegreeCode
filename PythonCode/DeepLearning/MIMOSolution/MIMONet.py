import torch
import torch.nn as nn

class MIMONet(nn.Module):
    def __init__(self):
        super(MIMONet, self).__init__()

        # 1) Primera convolución
        #   - in_channels=1 (un solo canal de entrada)
        #   - out_channels=32 (como en "convolution1dLayer(3,32,...)" en MATLAB)
        #   - kernel_size=3
        #   - En PyTorch no existe 'padding="causal"', así que se usa padding=1 (simétrico).
        self.conv1 = nn.Conv1d(
            in_channels=1, 
            out_channels=32, 
            kernel_size=3, 
            padding=1
        )
        
        self.relu1 = nn.ReLU()
        
        # 2) Normalización por capa. 
        #    En PyTorch, nn.LayerNorm espera normalizar sobre la última dimensión 
        #    del tensor si no se indica lo contrario. Para secuencias (N, C, L),
        #    muchas veces se hace un permute (N, L, C) antes y después.  
        #    Alternativamente, se puede usar GroupNorm o InstanceNorm.  
        #    Aquí usamos nn.LayerNorm(32) y haremos un permute en el forward.
        self.ln1 = nn.LayerNorm(32)
        
        # 3) Segunda convolución
        #   - in_channels=32
        #   - out_channels=64
        #   - kernel_size=5
        #   - padding=2 (simétrico)
        self.conv2 = nn.Conv1d(
            in_channels=32, 
            out_channels=64, 
            kernel_size=5, 
            padding=2
        )
        
        self.relu2 = nn.ReLU()
        self.ln2 = nn.LayerNorm(64)
        
        # 4) Global Average Pooling 1D
        #    Reduce la dimensión de la secuencia (L) a 1
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 5) Capa fully-connected con 4 salidas
        self.fc1 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(32, 4)
        
        # 6) Softmax final (dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        x.shape = (batch_size, 1, 64)
        """
        # -- Bloque 1: Conv -> ReLU -> (permute) -> LN -> (permute) --
        x = self.conv1(x)             # (N, 32, 64)
        x = self.relu1(x)
        
        x = x.transpose(1, 2)         # (N, 64, 32)  # Cambiamos ejes para LN
        x = self.ln1(x)               # Normaliza sobre la dimensión "32"
        x = x.transpose(1, 2)         # (N, 32, 64)  # Regresamos a (N, C, L)
        
        # -- Bloque 2: Conv -> ReLU -> (permute) -> LN -> (permute) --
        x = self.conv2(x)             # (N, 64, 64)
        x = self.relu2(x)
        
        x = x.transpose(1, 2)         # (N, 64, 64)
        x = self.ln2(x)               # Normaliza sobre la dimensión "64"
        x = x.transpose(1, 2)         # (N, 64, 64)
        
        # -- Global Average Pooling: reduce L de 64 a 1 --
        x = self.global_avg_pool(x)   # (N, 64, 1)
        x = x.squeeze(-1)            # (N, 64)
        
        # -- Fully-connected de 64 a 4 --
        x = self.fc1(x)               # (N, 4)
        x = self.relu3(x)
        x = self.fc2(x)               # (N, 4)
        
        # -- Softmax final --
        x = self.softmax(x)          # (N, 4)

        return x
