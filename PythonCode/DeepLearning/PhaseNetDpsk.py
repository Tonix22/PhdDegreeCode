import torch
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import TQDMProgressBar
import math
import numpy as np

# Add sys.path to include the Drivers and App directories for custom imports
parent_dir_app = os.path.abspath(os.path.join(os.path.dirname(__file__), '../DataLoaders'))
sys.path.insert(0, parent_dir_app)

# Import your custom data loader
from ImageChunksDataSet import ImageChunksDataset  # Adjust the import path as necessary

# Define hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
INPUT_SIZE = 48  # Should match your frame_size
HIDDEN_SIZE = 96
CONSTELLATION_SIZE = 4  # For example, 16-QAM
IMAGE_PATH = '/home/tonix/Documents/PhdDegreeCode/Data/Picture/Retsuko.jpeg'  # Replace with your image path

# Define the PhaseEqualizer network
class PhaseEqualizer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PhaseEqualizer, self).__init__()
        self.angle_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),  # Changed to ReLU
            
            nn.Linear(hidden_size, hidden_size * hidden_size),
            nn.LayerNorm(hidden_size * hidden_size),
            nn.LeakyReLU(),
            
            nn.Linear(hidden_size * hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            
            nn.Linear(hidden_size, input_size),
        )
    
    def forward(self, x):
        return self.angle_net(x)

class PhaseNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, learning_rate, image_path, constellation_size, style='Traditional', channel_snr=30, los=True):
        super(PhaseNet, self).__init__()
        self.save_hyperparameters()
        
        # Initialize the PhaseEqualizer network
        self.angle_net = PhaseEqualizer(input_size, hidden_size)
        
        # Define the loss function
        self.loss_fn = nn.MSELoss()
        
        # Store dataset parameters
        self.image_path = image_path
        self.constellation_size = constellation_size
        self.style = style
        self.channel_snr = channel_snr
        self.los = los
    
    def forward(self, x):
        return self.angle_net(x)
    
    def common_step(self,label,batch):
        input_tensor, target_tensor = batch
        input_tensor  = ((torch.angle(input_tensor).float() / torch.pi)+1)/2
        # Forward pass
        output = self(input_tensor)
        
        # Build real and imaginary part
        out_real = torch.cos((output*2)-1 * torch.pi)
        out_imag = torch.sin((output*2)-1 * torch.pi)
        
        # Compute loss
        loss_real = self.loss_fn(out_real, torch.real(target_tensor).float())
        loss_imag = self.loss_fn(out_imag, torch.imag(target_tensor).float())
        loss = (loss_real+loss_imag)/2
        
        self.log(label, loss)
        return loss
        
    def training_step(self, batch, batch_idx):
        return self.common_step('train_loss',batch)
    
    def validation_step(self, batch, batch_idx):
        return self.common_step('val_loss',batch)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        return optimizer
    
    def train_dataloader(self):
        # Create the dataset
        train_dataset = ImageChunksDataset(
            image_path=self.image_path,
            constelation_size=self.constellation_size,
            style=self.style,
            channel_snr=self.channel_snr,
            los=self.los
        )
        # Create the DataLoader
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle = False, num_workers = 16)
        return train_loader
            
    def val_dataloader(self):
        # Create the dataset
        val_dataset = ImageChunksDataset(
            image_path=self.image_path,
            constelation_size=self.constellation_size,
            style=self.style,
            channel_snr=self.channel_snr,
            los=self.los
        )
        # Create the DataLoader
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)
        return val_loader

if __name__ == '__main__':
    # Initialize the model
    model = PhaseNet(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        learning_rate=LEARNING_RATE,
        image_path=IMAGE_PATH,
        constellation_size=CONSTELLATION_SIZE,
        style='Traditional',  # Or 'DPSK' if you prefer
        channel_snr=30,
        los=True
    )
    
    # Initialize the trainer
    trainer = Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[TQDMProgressBar(refresh_rate=10)],
        logger=pl.loggers.TensorBoardLogger("tb_logs", name="PhaseNet")
    )
    
    # Train the model
    trainer.fit(model)
