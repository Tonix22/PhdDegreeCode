import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

# Custom Dataset to iterate over rows
class RowDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        self.num_realizations, self.num_rows, _ = data.shape  # (10000, 64, 64)

    def __len__(self):
        return self.num_realizations * self.num_rows  # 10000 * 64 total rows

    def __getitem__(self, index):
        realization_idx = index // self.num_rows  # Find the corresponding realization
        row_idx = index % self.num_rows  # Find the row index within the realization
        
        row = self.data[realization_idx, :, row_idx]  # Extract the row (shape: [64])
        label = self.targets[realization_idx, row_idx]  # Extract the corresponding target element
        
        row = row.unsqueeze(0)
        
        return row, label  # row: (64,), label: scalar

"""
basePath = "/home/tonix/Documents/PhdDegreeCode/MatlabCode/PuertoSourceCode/MIMODataSet/"
# Load dataset
data = np.load(basePath+"Signal_SNR_Rx_30.npy")  # Shape: (num_samples, num_rows, num_cols)
targets = np.load(basePath+"Signal_SNR_Tx_30.npy")  # Shape: (num_samples, labels)

# Convert to PyTorch tensors
data_tensor = torch.tensor(data, dtype=torch.float32)  # Shape: (num_samples, num_rows, num_cols)
target_tensor = torch.tensor(targets, dtype=torch.long)  # Shape: (num_samples, labels)
# Create dataset and DataLoader
dataset = RowDataset(data_tensor, target_tensor)
batch_size = 30
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# ✅ Corrected Function to Visualize Batch (Stacked Rows)
def plot_batch(batch_data, batch_labels):
    batch_labels = batch_labels.numpy()
    num_rows = batch_data.shape[0]
    plt.figure()
    plt.imshow(batch_data, cmap='jet')  # Display the phase as an image with colormap
    plt.colorbar()  # Show the color scale
    plt.title('Phase of the Complex Signal')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.gca().tick_params(axis='y', pad=10)
    plt.gca().set_aspect(2.0)
    plt.yticks(ticks=np.arange(num_rows), labels=batch_labels)
    plt.savefig('phase_plot.png')  # Save figure as a file
    plt.show()


# Iterate through DataLoader and visualize first batch
for batch_idx, (inputs, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx+1}:")
    print(f"  Input shape: {inputs.shape}")  # Expected: (batch_size, 64)
    print(f"  Labels shape: {labels.shape}")  # Expected: (batch_size,)

    plot_batch(inputs, labels)
    
    break  # Stop after one batch

"""

