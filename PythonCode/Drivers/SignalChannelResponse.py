import commpy.channels as chan  # Import communication channels module from commpy
import numpy as np  # Import NumPy for numerical operations
import os  # Import os for interacting with the operating system

# Define the path to the dataset
# Uncomment the following line if you want to use the current working directory
# DataPath = os.getcwd() + "../../Data/kaggle_dataset/"
DataPath = "/home/tonix/Documents/PhdDegreeCode/Data/kaggle_dataset/"  # Path to the dataset

class Channel():
    def __init__(self, snr, LOS=True):
        """
        Initialize the Channel object.

        Parameters:
        - snr: Signal-to-noise ratio
        - LOS: Line of Sight, True for LOS, False for Non-Line of Sight (NLOS)
        """
        self.channelLOS = LOS  # Set the type of channel (LOS or NLOS)
        if self.channelLOS:
            # Load the Line of Sight (LOS) channel data
            self.channel = np.load(DataPath + 'v2v80211p_LOS.npy')
        else:
            # Load the Non-Line of Sight (NLOS) channel data
            self.channel = np.load(DataPath + 'v2v80211p_NLOS.npy')

        self.snr = snr  # Set the SNR value
        self.index = 0  # Initialize the channel index
        # Time out for stationary channel
        # The channel may not change during a certain period of time
        
        self.stationary_timeout = 1000  # Timeout period for stationary channel
        self.stationary_idx = 0  # Initialize the stationary index

    def shift_next_channel(self):
        """
        Shift to the next channel by incrementing the index.
        """
        # Increment the channel index
        self.index += 1
        # Reset the index if it reaches 10000
        if self.index == 10000:
            self.index = 0

    def response(self, tx):
        """
        Generate the noisy signal with Additive White Gaussian Noise (AWGN).

        Parameters:
        - tx: Transmitted signal

        Returns:
        - Noisy received signal
        """
        # Generate the noisy signal with AWGN and channel effect
        self.noisy_signal = tx @ self.channel[:, :, self.index] + chan.awgn(tx, self.snr)

        # Increment the stationary index
        self.stationary_idx += 1
        # Check if the stationary index exceeds the timeout
        if self.stationary_idx > self.stationary_timeout:
            # Reset the stationary index and shift to the next channel
            self.stationary_idx = 0
            self.shift_next_channel()

        return self.noisy_signal  # Return the noisy signal
