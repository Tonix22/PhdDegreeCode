from torchvision import transforms
from ImageNetOFDMChannelDataSet import ImageNetOFDMChannelDataSet
from torch.utils.data import DataLoader

class ImageNetDataLoader:
    def __init__(self, root_dir, constelation_size, SNR, batch_size=32, num_workers=4):
        """
        Class to set up DataLoaders for ImageNet training, validation, and testing sets.

        Args:
            root_dir (str): Path to the root directory where ImageNet is stored or downloaded.
            constelation_size (int): Size of the constellation (e.g., 4 for QPSK).
            SNR (int): Signal-to-noise ratio (SNR) for the channel.
            batch_size (int, optional): Batch size for DataLoaders. Default is 32.
            num_workers (int, optional): Number of worker processes for loading data. Default is 4.
        """
        self.root_dir = root_dir
        self.constelation_size = constelation_size
        self.SNR = SNR
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Define transformations for the images
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to 256x256
            transforms.CenterCrop(224),     # Center crop to 224x224
            transforms.ToTensor(),          # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet stats
        ])

    def get_dataset(self, split):
        """
        Load ImageNet dataset for a specific split (train, val, test).

        Args:
            split (str): Dataset split to use ('train' or 'val').

        Returns:
            ImageNetOFDMChannelDataSet: Dataset object for the given split.
        """
        return ImageNetOFDMChannelDataSet(
            root_dir=self.root_dir,
            split=split,
            constelation_size=self.constelation_size,
            SNR=self.SNR,
            transform=self.transform,
            download=True
        )

    def get_dataloader(self, dataset):
        """
        Create a DataLoader for a given dataset.

        Args:
            dataset (Dataset): The dataset for which to create the DataLoader.

        Returns:
            DataLoader: DataLoader object for the provided dataset.
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True if dataset.split == 'train' else False,
            num_workers=self.num_workers
        )

    def setup_dataloaders(self):
        """
        Set up DataLoaders for training, validation, and test sets.

        Returns:
            tuple: A tuple containing DataLoader objects for train, val, and test sets.
        """
        train_dataset = self.get_dataset('train')
        val_dataset = self.get_dataset('val')
        test_dataset = self.get_dataset('val')  # Typically use 'val' split for testing

        train_loader = self.get_dataloader(train_dataset)
        val_loader = self.get_dataloader(val_dataset)
        test_loader = self.get_dataloader(test_dataset)

        return train_loader, val_loader, test_loader
