import torch
import os
import sys
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
from DPSK import DPSK_OFDM

# Add sys.path to include the DataLoaders directory for custom imports
parent_dir_app = os.path.abspath(os.path.join(os.path.dirname(__file__), '../DataLoaders'))
sys.path.insert(0, parent_dir_app)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Drivers'))
sys.path.insert(0, parent_dir)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../DeepLearning'))
sys.path.insert(0, parent_dir)

from ConstelationCoder import ConstelationCoder, Modulation  # type: ignore

# Import your custom data loader and model definitions
from ImageChunksDataSet import ImageChunksDataset
from PhaseNetDpsk import PhaseNet

# Define hyperparameters (should match those used during training)
INPUT_SIZE = 48
HIDDEN_SIZE = 96
LEARNING_RATE = 1e-3
CONSTELLATION_SIZE = 4
IMAGE_PATH = '/home/tonix/Documents/PhdDegreeCode/Data/Picture/Retsuko.jpeg'  # Replace with your test image path

# Path to the saved checkpoint
CHECKPOINT_PATH = '/home/tonix/Documents/PhdDegreeCode/PythonCode/DeepLearning/tb_logs/PhaseNet/version_41/checkpoints/epoch=14-step=13170.ckpt'  # Adjust as necessary

def main():
    # Initialize the model (parameters must match those used during training)
    model = PhaseNet(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        learning_rate=LEARNING_RATE,
        image_path=IMAGE_PATH,  # Not used during inference but required for initialization
        constellation_size=CONSTELLATION_SIZE,
        style='DPSK',
        channel_snr=30,
        los=True
    )

    # Load the checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))  # Use 'cuda' if using GPU

    # Load the model state_dict
    model.load_state_dict(checkpoint['state_dict'])

    # Optionally, move the model to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Set the model to evaluation mode
    model.eval()

    H = np.load('/home/tonix/Documents/PhdDegreeCode/Data/kaggle_dataset/v2v80211p_LOS.npy')  # Placeholder for channel matrix

    # Initialize and run the simulation
    ofdm_system = DPSK_OFDM(
        snr_dB_range=[5, 10, 15, 25, 30],
        modulation_order=4,
        fft_size=48,
        num_subcarriers=48,
        channel_snr=10,
        los=True
    )
    
    dataset = ImageChunksDataset(
        image_path=IMAGE_PATH,
        constelation_size=4,
        style='DPSK',
        channel_snr=30,
        los=True
    )
    
    ofdm_system.run_simulation(txbits = dataset.databit_encode.frames, network = model)
    

if __name__ == '__main__':
    main()
