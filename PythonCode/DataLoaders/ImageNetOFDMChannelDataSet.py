import torch
from torch.utils.data import Dataset
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import datasets
import sys

# Add sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Drivers'))
sys.path.insert(0, parent_dir)

# Import Encoder and Driver
from DataBinEncoder import EncodeDataIntoBits # type: ignore
from DataBinDecoder import DecodeFramesToBits # type: ignore

from PictureEncoder import PictureEncoder # type: ignore
from PictureDecoder import PictureDecoder # type: ignore

from ConstelationCoder import ConstelationCoder, Modulation # type: ignore
from SignalChannelResponse import Channel # type: ignore

# Custom dataset class based on ImageNet with SNR parameter
class ImageNetOFDMChannelDataSet(Dataset):
    def __init__(self, root_dir, split, constelation_size, SNR, transform=None, download=False):
        """
        PyTorch Dataset class for loading ImageNet images and processing them through a channel.

        Args:
            root_dir (str): Path to the root ImageNet dataset directory.
            split (str): Dataset split to use ('train' or 'val').
            constelation_size (int): Size of the constellation (e.g., 4 for QPSK).
            SNR (int): Signal-to-noise ratio (SNR) for the channel.
            transform (callable, optional): Optional transform to be applied on an image.
            download (bool, optional): Whether to download the dataset if not found.
        """
        self.transform = transform
        self.constelation_size = constelation_size
        self.SNR = SNR  # Save SNR value to the class

        # Load the ImageNet dataset
        self.imagenet_data = datasets.ImageNet(root=os.path.join(root_dir, 'imagenet'), split=split, transform=transform, download=download)

        # Preprocess images into chunks or frames
        self.frame_size = 96  # Adjust as needed for frames
        self.bit_slice = int(np.log2(self.constelation_size))  # Number of bits per symbol

        # Initialize the channel with the provided SNR
        self.channel = Channel(self.SNR)

    def __len__(self):
        return len(self.imagenet_data)

    def __getitem__(self, idx):
        """
        Fetch the original image and generate the target image after encoding/decoding through the channel.

        Args:
            idx (int): Index of the image.

        Returns:
            tuple: (input_tensor, target_tensor)
                - input_tensor (torch.Tensor): The original image as a tensor.
                - target_tensor (torch.Tensor): The image after encoding/decoding.
        """
        # Get original image and its label from ImageNet
        input_image, label = self.imagenet_data[idx]

        # Convert image to numpy array for processing
        input_image_np = np.array(input_image)

        # Encoding the image
        picture_encoder = PictureEncoder(input_image_np)
        encoded_bytes = picture_encoder.encode()  # Encode the image into bytes

        # Encode the bytes into frames
        databit_encode = EncodeDataIntoBits(encoded_bytes, self.bit_slice, self.frame_size)
        databit_encode.process_byte_array()

        # Initialize coder
        coder = ConstelationCoder(Modulation.QAM, self.constelation_size)
        rx_frames = np.empty_like(databit_encode.frames)

        # Process the frames
        for i in tqdm(range(databit_encode.frames.shape[0]), desc="Processing Frames"):
            tx_bits = databit_encode.frames[i, :]  # Access the i-th row
            tx = coder.Encode(tx_bits)  # Encode the bits into symbols
            rx = self.channel.response(tx)   # Pass through the channel
            rx_bits = coder.Decode(rx)  # Decode the received signal back to bits

            # Save the received frame
            rx_frames[i, :] = rx_bits

        # Decode the received frames back to bytes
        databit_decode = DecodeFramesToBits(rx_frames, self.frame_size)
        decoded_output = databit_decode.flatten_frames_in_bytes()

        # Decode the picture from the byte sequence
        decoder = PictureDecoder(decoded_output)
        output_image_np = decoder.hint_decode(input_image_np.shape[0], input_image_np.shape[1], input_image_np.shape[2])  # Decode the image back

        # Convert the output image back to a tensor
        target_tensor = torch.from_numpy(np.array(output_image_np)).permute(2, 0, 1)  # Convert from HWC to CHW format
        
        # Apply any additional transformations (e.g., normalization) to the input image if necessary
        if self.transform:
            input_image = self.transform(input_image)

        # Return both input image and the processed target tensor
        return input_image, target_tensor
