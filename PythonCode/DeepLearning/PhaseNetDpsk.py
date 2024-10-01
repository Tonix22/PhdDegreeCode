import torch
import os
import sys
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import TQDMProgressBar
import numpy as np

# Add sys.path to include the DataLoaders directory for custom imports
parent_dir_app = os.path.abspath(os.path.join(os.path.dirname(__file__), '../DataLoaders'))
sys.path.insert(0, parent_dir_app)

# Import your custom data loader
from ImageChunksDataSet import ImageChunksDataset  # Adjust the import path as necessary
from PhaseNetEqualizer import PhaseEqualizer

# Define hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
INPUT_SIZE = 48       # Should match your frame_size in the dataset
HIDDEN_SIZE = 96      # Size of the hidden layers in the network
CONSTELLATION_SIZE = 4  # For example, 4-QAM (Quadrature Amplitude Modulation)
IMAGE_PATH = '/home/tonix/Documents/PhdDegreeCode/Data/Picture/Retsuko.jpeg'  # Replace with your image path

# Define the PyTorch Lightning Module
class PhaseNet(pl.LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size,
        learning_rate,
        image_path,
        constellation_size,
        style='Traditional',
        channel_snr=30,
        los=True
    ):
        super(PhaseNet, self).__init__()
        self.save_hyperparameters()  # Save hyperparameters for easy access
        
        # Initialize the PhaseEqualizer network
        self.angle_net = PhaseEqualizer(input_size, hidden_size)
        
        # Define the loss function (Mean Squared Error)
        self.loss_fn = nn.MSELoss()
        
        # Store dataset parameters
        self.image_path = image_path                # Path to the image used in the dataset
        self.constellation_size = constellation_size  # Modulation scheme used
        self.style = style                          # Transmission style ('Traditional' or 'DPSK')
        self.channel_snr = channel_snr              # Signal-to-noise ratio for the channel
        self.los = los                              # Line-of-sight flag (True or False)
    
    def forward(self, x):
        # Forward pass through the PhaseEqualizer network
        return self.angle_net(x)
    
    def common_step(self, label, batch):
        input_tensor, target_tensor = batch  # Get input and target tensors from the batch

        # Normalize the angle of the input tensor to [0,1]
        # torch.angle returns the angle (phase) of the complex tensor elements
        input_tensor = (torch.angle(input_tensor).float() / torch.pi)
        
        # Forward pass through the network
        output = self(input_tensor)
        
        # Build real and imaginary parts from the output angles
        # Map the output back to the range [-π, π] and compute cosine and sine
        out_real = torch.cos((output) * torch.pi)
        out_imag = torch.sin((output) * torch.pi)
        
        # Compute the loss between the output and target real and imaginary parts
        loss_real = self.loss_fn(out_real, torch.real(target_tensor).float())
        loss_imag = self.loss_fn(out_imag, torch.imag(target_tensor).float())
        loss = (loss_real + loss_imag) / 2  # Average the real and imaginary losses
        
        self.log(label, loss)  # Log the loss with the given label ('train_loss' or 'val_loss')
        return loss  # Return the loss value
        
    def training_step(self, batch, batch_idx):
        # Training step called during training loop
        return self.common_step('train_loss', batch)
    
    def validation_step(self, batch, batch_idx):
        # Validation step called during validation loop
        return self.common_step('val_loss', batch)
    
    def configure_optimizers(self):
        # Configure the optimizer for training
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        return optimizer
    
    def train_dataloader(self):
        # Create the dataset for training
        train_dataset = ImageChunksDataset(
            image_path=self.image_path,
            constelation_size=self.constellation_size,
            style=self.style,
            channel_snr=self.channel_snr,
            los=self.los
        )
        # Create the DataLoader for training
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=16
        )
        return train_loader
            
    def val_dataloader(self):
        # Create the dataset for validation
        val_dataset = ImageChunksDataset(
            image_path=self.image_path,
            constelation_size=self.constellation_size,
            style=self.style,
            channel_snr=self.channel_snr,
            los=self.los
        )
        # Create the DataLoader for validation
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=16
        )
        return val_loader

if __name__ == '__main__':
    # Initialize the model with the specified parameters
    model = PhaseNet(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        learning_rate=LEARNING_RATE,
        image_path=IMAGE_PATH,
        constellation_size=CONSTELLATION_SIZE,
        style='DPSK',  # Or 'DPSK' if you prefer
        channel_snr=30,
        los=True
    )
    
    # Initialize the trainer with the specified configurations
    trainer = Trainer(
        max_epochs=NUM_EPOCHS,                           # Number of training epochs
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',  # Use GPU if available
        callbacks=[TQDMProgressBar(refresh_rate=10)],    # Progress bar callback
        logger=pl.loggers.TensorBoardLogger("tb_logs", name="PhaseNet")  # TensorBoard logger
    )
    
    # Start the training process
    trainer.fit(model)
