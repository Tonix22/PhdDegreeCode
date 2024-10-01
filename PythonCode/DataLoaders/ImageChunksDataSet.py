import torch
from torch.utils.data import Dataset
import os
import sys
from PIL import Image
import math

# Add sys.path to include the Drivers and App directories for custom imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Drivers'))
sys.path.insert(0, parent_dir)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../App'))
sys.path.insert(0, parent_dir)

# Import custom modules
from PictureEncoder import PictureEncoder  # type: ignore
from SignalChannelResponse import Channel  # type: ignore
from DataBinEncoder import EncodeDataIntoBits  # type: ignore
from ConstelationCoder import ConstelationCoder, Modulation  # type: ignore
from DPSK import DPSK_OFDM  # type: ignore

class ImageChunksDataset(Dataset):
    def __init__(self, image_path, constelation_size, style='Traditional', channel_snr=30, los=True):
        """
        PyTorch Dataset class for loading and processing image data into chunks suitable for transmission over a channel.

        Args:
            image_path (str): Path to the image file.
            constelation_size (int): Size of the constellation (e.g., 4 for QPSK).
            style (str, optional): Transmission style, either 'Traditional' or 'DPSK'. Default is 'Traditional'.
            channel_snr (float, optional): Signal-to-noise ratio for the channel in dB. Default is 30 dB.
            los (bool, optional): Line-of-sight flag for the channel. True for LOS, False for NLOS. Default is True.
        """
        self.image_path = image_path
        
        # Open the image to determine its dimensions and number of channels
        self.image = Image.open(self.image_path)
        self.width, self.height = self.image.size
        self.channels = len(self.image.getbands())  # Number of channels (e.g., 1 for grayscale, 3 for RGB)
        self.image.close()  # Close the image file
        
        self.constelation_size = constelation_size  # Constellation size (e.g., 4 for QPSK)
        self.frame_size = 96  # Frame size (e.g., 96 symbols per frame as per V2V standard)
        self.bit_slice = int(math.log2(self.constelation_size))  # Number of bits per symbol
        self.channel_snr = channel_snr  # Channel SNR in dB
        self.los = los  # Line-of-sight flag
        
        # Initialize parameters
        # Encode the image to get encoded_bytes
        picture_encoder = PictureEncoder(self.image_path, self.height, self.width, self.channels)
        self.encoded_bytes = picture_encoder.encode()  # Encoded image as bytes

        # Process the encoded bytes into frames of bits
        self.databit_encode = EncodeDataIntoBits(self.encoded_bytes, self.bit_slice, self.frame_size)
        self.databit_encode.process_byte_array()  # Convert bytes to bits and split into frames

        # Store the total number of frames
        self.num_frames = self.databit_encode.frames.shape[0]
        
        # Initialize the channel and constellation coder
        self.channel = Channel(self.channel_snr, LOS=self.los)
        self.coder = ConstelationCoder(Modulation.QAM, self.constelation_size)
        
        self.style = style  # Transmission style ('Traditional' or 'DPSK')
        if self.style == 'DPSK':
            # Initialize the DPSK-OFDM system if style is 'DPSK'
            self.dpsk = DPSK_OFDM(
                snr_dB_range=[channel_snr],
                modulation_order=self.constelation_size,
                fft_size=self.frame_size // 2,
                num_subcarriers=self.frame_size // 2,
                channel_snr=channel_snr,
                los=los
            )

    def __len__(self):
        """
        Return the total number of frames in the dataset.
        """
        return self.num_frames

    def __getitem__(self, idx):
        """
        Retrieve the input and target data for a given index.

        Args:
            idx (int): Index of the data item.

        Returns:
            tuple: (input_tensor, target_tensor)
                - input_tensor (torch.Tensor): Received signal after processing (model input).
                - target_tensor (torch.Tensor): Original transmitted symbols (ground truth for training).
        """
        # Get the bits for the given frame
        tx_bits = self.databit_encode.frames[idx, :]
        # Modulate the bits to symbols using the constellation coder
        target = self.coder.Encode(tx_bits)

        if self.style == 'DPSK':
            # DPSK encoding
            DPSK_signalTx = self.dpsk.applyDPSKEncoding(target)
            # Get the channel matrix
            G = self.channel.getChannel()
            # Pass the DPSK signal through the channel and add noise
            signalRx = self.dpsk.DPSK_channel_and_Noise(DPSK_signalTx, G, self.channel_snr)
            
            # OFDM demodulation and differential decoding
            OFDM_signalRx = self.dpsk.utils.ofdm_demodulate(signalRx)
            input_signal = self.dpsk.applyDPSKDecoding(OFDM_signalRx)
            
        elif self.style == 'Traditional':
            # Traditional transmission through the channel
            input_signal = self.channel.response(target)
        
        else:
            raise ValueError(f"Invalid style '{self.style}'. Choose 'Traditional' or 'DPSK'.")

        # Convert input and target to torch tensors
        input_tensor = torch.from_numpy(input_signal)
        target_tensor = torch.from_numpy(target)
        
        return input_tensor, target_tensor
