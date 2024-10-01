import os
import sys
import unittest
import numpy as np
from PIL import Image
import torch
from unittest.mock import MagicMock, patch

parent_dir_app = os.path.abspath(os.path.join(os.path.dirname(__file__), '../DataLoaders'))
sys.path.insert(0, parent_dir_app)

# Import the ImageChunksDataset class
# Adjust the import path based on where your ImageChunksDataset class is defined
from ImageChunksDataSet import ImageChunksDataset

class TestImageChunksDataset(unittest.TestCase):
    """
    Unit test class for the ImageChunksDataset.
    """
    
    def setUp(self):
        """
        Set up the test environment before each test method.
        """
        # Test parameters
        self.image_path = 'test_image.png'  # Path to a test image
        self.constellation_size = 4         # For example, QPSK
        self.style = 'DPSK'          # 'Traditional' or 'DPSK'
        self.channel_snr = 30
        self.los = True

        # Create a small test image if it doesn't exist
        if not os.path.exists(self.image_path):
            width, height = 10, 10
            random_image_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            img = Image.fromarray(random_image_array, 'RGB')
            img.save(self.image_path)

        # Initialize the dataset
        self.dataset = ImageChunksDataset(
            image_path=self.image_path,
            constelation_size=self.constellation_size,
            style=self.style,
            channel_snr=self.channel_snr,
            los=self.los
        )

    def test_dataset_length(self):
        """
        Test that the dataset length matches the expected number of frames.
        """
        num_frames = self.dataset.num_frames
        self.assertEqual(len(self.dataset), num_frames, "Dataset length does not match the number of frames.")

    def test_get_item_returns_tensors(self):
        """
        Test that __getitem__ returns tensors of the correct type.
        """
        idx = 0  # Test the first item
        input_tensor, target_tensor = self.dataset[idx]

        # Check that the returned objects are instances of torch.Tensor
        self.assertIsInstance(input_tensor, torch.Tensor, "Input is not a torch.Tensor.")
        self.assertIsInstance(target_tensor, torch.Tensor, "Target is not a torch.Tensor.")

    def test_get_item_tensor_shapes(self):
        """
        Test that the tensors returned by __getitem__ have the correct shapes.
        """
        idx = 0  # Test the first item
        input_tensor, target_tensor = self.dataset[idx]

        # Check that the tensors are not empty and have the expected shape
        self.assertGreater(input_tensor.numel(), 0, "Input tensor is empty.")
        self.assertGreater(target_tensor.numel(), 0, "Target tensor is empty.")

        # Optionally, check the specific shape if known
        expected_shape = (self.dataset.frame_size//2,)
        self.assertEqual(input_tensor.shape, expected_shape, f"Input tensor shape is not {expected_shape}.")
        self.assertEqual(target_tensor.shape, expected_shape, f"Target tensor shape is not {expected_shape}.")

    def test_all_items(self):
        """
        Optionally, iterate through all items in the dataset to ensure consistency.
        """
        for idx in range(len(self.dataset)):
            input_tensor, target_tensor = self.dataset[idx]
            self.assertIsInstance(input_tensor, torch.Tensor, f"Input at index {idx} is not a torch.Tensor.")
            self.assertIsInstance(target_tensor, torch.Tensor, f"Target at index {idx} is not a torch.Tensor.")
            self.assertEqual(input_tensor.shape, target_tensor.shape, f"Input and target shapes do not match at index {idx}.")

    def tearDown(self):
        """
        Clean up after each test method.
        """
        # Remove the test image file if it was created
        if os.path.exists(self.image_path):
            os.remove(self.image_path)

if __name__ == '__main__':
    unittest.main()
