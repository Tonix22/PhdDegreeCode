import numpy as np  # Import NumPy for numerical operations
from Ofdm_utils import OFDMUtils

# Define the path to the dataset
# Uncomment the following line if you want to use the current working directory
# DataPath = os.getcwd() + "../../Data/kaggle_dataset/"
DataPath = "/home/tonix/Documents/PhdDegreeCode/Data/kaggle_dataset/"  # Path to the dataset

class Channel():
    def __init__(self, snr, channel_path = DataPath, LOS=True, stationary_timeout = 1000):
        """
        Initialize the Channel object.

        Parameters:
        - snr: Signal-to-noise ratio
        - channel_path: data channel path
        - LOS: Line of Sight, True for LOS, False for Non-Line of Sight (NLOS)
        """
        self.channel_path = channel_path
        self.channelLOS = LOS  # Set the type of channel (LOS or NLOS)
        if self.channelLOS:
            # Load the Line of Sight (LOS) channel data
            self.channel = np.load(self.channel_path + 'v2v80211p_LOS.npy')
        else:
            # Load the Non-Line of Sight (NLOS) channel data
            self.channel = np.load(self.channel_path + 'v2v80211p_NLOS.npy')

        self.snr = snr  # Set the SNR value
        self.index = 0  # Initialize the channel index
        # Time out for stationary channel
        # The channel may not change during a certain period of time
        
        self.stationary_timeout = stationary_timeout  # Timeout period for stationary channel
        self.stationary_idx = 0  # Initialize the stationary index
        self.ofdm_utils = OFDMUtils()

    def shift_next_channel(self):
        """
        Shift to the next channel by incrementing the index.
        """
        # Increment the channel index
        self.index += 1
        # Reset the index if it reaches 10000
        if self.index == 1000:
            self.index = 0

    def getChannel(self):
        """
        Get current channel and move forward the stationary index.
        When the stationary movement end it will change the channel.
        """
        H = self.channel[:, :, self.index]
        # Increment the stationary index
        self.stationary_idx += 1
        # Check if the stationary index exceeds the timeout
        if self.stationary_idx > self.stationary_timeout:
            # Reset the stationary index and shift to the next channel
            self.stationary_idx = 0
            self.shift_next_channel()
            
        return H

    def response(self, tx):
        """
        Generate the noisy signal with Additive White Gaussian Noise (AWGN).

        Parameters:
        - tx: Transmitted signal

        Returns:
        - Noisy received signal
        """
        G = self.getChannel()
        # Generate the noisy signal with AWGN and channel effect
        self.noisy_signal = tx @ G
        self.noisy_signal = self.ofdm_utils.awgn_python(self.noisy_signal, self.snr, measured = False)
        
        return self.noisy_signal  # Return the noisy signal
