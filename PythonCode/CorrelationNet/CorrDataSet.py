import numpy as np
import torch
from torch.utils.data import Dataset

class CorrDataset(Dataset):
    """
    Loads 4-channel correlation matrices from .npy files
    and applies channel-wise normalization.
    
    Expected shape of `data` is (N, 4, 48, 48).
    Labels are (N, 48) with integer classes in [0..3].
    """
    def __init__(self, data_file, label_file, transform=None):
        super().__init__()
        
        # Load arrays from .npy
        self.data = np.load(data_file)     # Expect shape: (N, 4, 48, 48)
        self.labels = np.load(label_file)  # Expect shape: (N, 48)

        # Optional transform (if you want additional augmentations, etc.)
        self.transform = transform

        # --- Validate Shapes ---
        if len(self.data.shape) != 4:
            raise ValueError(
                f"Data file must have 4D shape (N,4,48,48). Got {self.data.shape} instead."
            )
        # Check that second dimension == 4 (channels), third/fourth == 48
        if not (self.data.shape[1] == 4 and self.data.shape[2] == 48 and self.data.shape[3] == 48):
            raise ValueError(
                f"Data file shape must be (N,4,48,48). Got {self.data.shape} instead."
            )
        # Check that labels match N
        if self.labels.shape[0] != self.data.shape[0]:
            raise ValueError(
                f"Data and Labels must have the same length. "
                f"Data N={self.data.shape[0]}, Labels N={self.labels.shape[0]}"
            )
        # Check that label dimension is 48
        if self.labels.shape[1] != 48:
            raise ValueError(
                f"Labels must be (N,48). Got {self.labels.shape} instead."
            )

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        Returns:
          x: FloatTensor, shape (4, 48, 48)
          y: LongTensor, shape (48,)
        """
        x = self.data[idx]    # numpy array (4, 48, 48)
        y = self.labels[idx]  # numpy array (48,)

        # Convert to PyTorch tensors
        x = torch.from_numpy(x).float()  # shape (4, 48, 48)
        y = torch.from_numpy(y).long()   # shape (48,)

        # --- Channel-wise normalization: mean/std per channel ---
        for c in range(x.shape[0]):  # c in {0,1,2,3} for 4 channels
            channel_min = x[c].min()
            channel_max = x[c].max()
            # Avoid divide-by-zero if channel_min == channel_max
            x[c] = (x[c] - channel_min) / (channel_max - channel_min)
        
        # Optional additional transforms (if any)
        if self.transform:
            x = self.transform(x)

        return x, y
