import json
import argparse
import torch
import numpy as np
from pytorch_lightning import Trainer
from LightningGym import LitModel
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split
from RowDataset import *

def load_config(json_path):
    """ Load training configuration from JSON file. """
    with open(json_path, 'r') as file:
        config = json.load(file)
    return config

def main(json_path):
    # Load configurations from JSON
    config = load_config(json_path)

    # Extract configurations
    EPOCHS = config.get("EPOCHS", 20)
    BATCHSIZE = config.get("BATCHSIZE", 256)
    LEARNINGRATE = config.get("LEARNINGRATE", 1e-3)
    TRAINPERCENT = config.get("TRAINPERCENT", 0.8)
    SNR_RANGE = config.get("SNR_RANGE", [10, 15, 20, 25, 30, 35])  # Default range if not specified
    DATA_PATH = config.get("DATA_PATH", "data/")
    MODEL_SAVE_PATH = config.get("MODEL_SAVE_PATH", "TrainnedModels/")
    NUM_WORKERS = config.get("NUM_WORKERS", 16)
    USE_GPU = config.get("USE_GPU", True)

    torch.manual_seed(0)

    for SNR in SNR_RANGE:
        print(f"CURRENT SNR TRAINING: {SNR}")

        # Load dataset
        rx_data = np.load(f"{DATA_PATH}Signal_SNR_Rx_{SNR}.npy")
        tx_data = np.load(f"{DATA_PATH}Signal_SNR_Tx_{SNR}.npy")

        # Convert to PyTorch tensors
        data_tensor = torch.tensor(rx_data, dtype=torch.float32)
        target_tensor = torch.tensor(tx_data, dtype=torch.long)

        # Initialize model
        model = LitModel(lr=LEARNINGRATE)

        # Initialize logger
        logger = TensorBoardLogger(
            "lightning_logs", 
            name=f"MIMO_{SNR}_Epochs_{EPOCHS}_BS{BATCHSIZE}_LR{LEARNINGRATE}"
        )

        # Trainer configuration
        trainer = Trainer(
            max_epochs=EPOCHS,
            logger=logger,
            accelerator="gpu" if USE_GPU else "cpu",
            devices=1 if USE_GPU else None,
        )

        # Prepare dataset
        dataset = RowDataset(data_tensor, target_tensor)
        train_size = int(TRAINPERCENT * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, num_workers=NUM_WORKERS, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCHSIZE, num_workers=NUM_WORKERS, shuffle=False)

        # Train model
        trainer.fit(model, train_loader, val_loader)

        # Save trained model
        model_path = f"{MODEL_SAVE_PATH}model_MIMO_DPSK_{SNR}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Saved model: {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MIMO Model with configurable parameters")
    parser.add_argument("config_json", type=str, help="Path to JSON configuration file")
    args = parser.parse_args()

    main(args.config_json)
