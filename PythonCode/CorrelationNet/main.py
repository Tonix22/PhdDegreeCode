# train.py
import os
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from CorrDataSet import CorrDataset
from CorrNet import CorrelationNetLightning

# Create a torch.Generator and seed it
g = torch.Generator().manual_seed(42)

def run_training():
    # Single dataset path (adjust to your actual paths)
    data_file  = "/home/tonix/Documents/PhdDegreeCode/MatlabCode/NeuronalNetwork/CorrDataGeneration/Data/CorrData_SNR_35dB_data.npy"
    label_file = "/home/tonix/Documents/PhdDegreeCode/MatlabCode/NeuronalNetwork/CorrDataGeneration/Data/CorrData_SNR_35dB_label.npy"


    # Load full dataset
    full_dataset = CorrDataset(
        data_file=data_file,
        label_file=label_file,
    )

    # Train/val split
    val_ratio = 0.2
    total_size = len(full_dataset)
    val_size = int(val_ratio * total_size)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=g  # or torch.Generator().manual_seed(42)
    )

    print(f"Total samples: {total_size}")
    print(f"Train samples: {train_size}, Val samples: {val_size}")

    # Build DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False,num_workers=16)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False,num_workers=16)
    
    # Create model
    model = CorrelationNetLightning(num_classes=4, output_dim=48, lr=0.001)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=4,
        accelerator="auto",  # automatically use GPU if available
        deterministic=True
    )

    # Fit
    trainer.fit(model, train_loader, val_loader)
    
    # Optional test step, if you want to use the same dataset or a separate test set:
    # trainer.test(model, dataloaders=val_loader)

if __name__ == "__main__":
    run_training()
