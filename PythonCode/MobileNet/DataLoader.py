import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ComplexNpyDataset(Dataset):
    """
    Dataset que carga un .npy con matrices complejas de 48x48
    y devuelve un tensor (2, 48, 48) con:
        canal 0 → magnitud   |x|
        canal 1 → fase       ∠x  (rad, en [-π, π])
    """
    def __init__(
        self,
        npy_path: str,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
        transform=None,
        mmap: bool = False,
    ):
        # Carga con o sin memory-mapping (útil para archivos grandes)
        self.arr = np.load(npy_path, mmap_mode="r" if mmap else None)
        if not np.iscomplexobj(self.arr):
            raise ValueError("El archivo debe contener números complejos.")

        # Forma esperada: (N, 48, 48)
        if self.arr.ndim != 3 or self.arr.shape[1:] != (48, 48):
            raise ValueError(f"Se esperaba forma (N,48,48); se obtuvo {self.arr.shape}")

        self.dtype     = dtype
        self.device    = device
        self.transform = transform            # p. ej. normalización adicional

    def __len__(self):
        return self.arr.shape[0]

    def __getitem__(self, idx):
        x = self.arr[idx]                     # (48,48) complejo
        mag   = np.abs(x).astype(np.float32)  # magnitud
        phase = np.angle(x).astype(np.float32)# fase  [-π, π]

        # (2,48,48)  → canal 0 = |x|, canal 1 = ∠x
        sample = np.stack((mag, phase), axis=0)

        tensor = torch.from_numpy(sample).to(self.dtype)
        if self.device is not None:
            tensor = tensor.to(self.device, non_blocking=True)

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor

# Ejemplo de uso --------------------------------------------------------------
if __name__ == "__main__":
    DATA_PATH = "/home/tonix/Documents/PhdDegreeCode/Data/kaggle_dataset/v2v80211p_LOS.npy"

    dataset = ComplexNpyDataset(
        DATA_PATH,
        dtype=torch.float32,
        device=None,          # o torch.device("cuda:0") si prefieres
        mmap=True             # más eficiente para archivos grandes
    )

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    for batch in dataloader:
        # batch → (B, 2, 48, 48)
        magnitudes = batch[:, 0]   # |x|
        phases     = batch[:, 1]   # ∠x
        # ... entrenamiento / inferencia ...
