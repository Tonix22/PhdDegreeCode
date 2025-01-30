import matplotlib.pyplot as plt
import numpy as np
import os
from ImageNetDataLoader import ImageNetDataLoader 

def imshow(img, title=None):
    """Helper function to display an image"""
    img = img.numpy().transpose((1, 2, 0))  # Convert from CHW to HWC format
    mean = np.array([0.485, 0.456, 0.406])  # Mean used in normalization
    std = np.array([0.229, 0.224, 0.225])   # Std used in normalization
    img = std * img + mean  # Denormalize the image
    img = np.clip(img, 0, 1)  # Clip values to ensure they are in valid range [0, 1]
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.show()

def test_display_images(data_loader):
    """Test function to display both input and target images"""
    for input_tensor, target_tensor in data_loader:
        # Visualize the input image
        print("Displaying input image...")
        imshow(input_tensor[0], title="Input Image (Original ImageNet Image)")

        # Visualize the target image (after channel processing)
        print("Displaying target image...")
        imshow(target_tensor[0], title="Target Image (After Channel Encoding/Decoding)")

        # Only display the first pair of images, break the loop after one iteration
        break

# Usage example
if __name__ == "__main__":
    root_dir = os.getcwd()  # Use current directory
    constelation_size = 4
    SNR = 30  # Set your desired SNR value
    batch_size = 1  # Load one image at a time for easy visualization
    num_workers = 4

    # Initialize the DataLoader setup class
    data_loader_setup = ImageNetDataLoader(root_dir=root_dir, constelation_size=constelation_size, SNR=SNR, batch_size=batch_size, num_workers=num_workers)

    # Get the DataLoaders
    train_loader, val_loader, test_loader = data_loader_setup.setup_dataloaders()

    # Test with the train_loader or any other loader
    test_display_images(train_loader)
