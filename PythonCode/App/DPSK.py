import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import os
import sys
import matlab.engine
from tqdm import tqdm

# Add sys.path for accessing the required modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Drivers'))
sys.path.insert(0, parent_dir)

from ConstelationCoder import ConstelationCoder, Modulation  # type: ignore
from SignalChannelResponse import Channel  # Import the custom Channel class
from Ofdm_utils import OFDMUtils  # Import the utils class

class DPSK_OFDM:
    def __init__(self, snr_dB_range, modulation_order, fft_size, num_subcarriers, channel_snr, los=True):
        """
        Initialize the OFDM system parameters.
        Args:
            snr_dB_range (list): Range of SNR values in dB.
            modulation_order (int): Modulation order (e.g., 4 for QPSK).
            fft_size (int): Size of the FFT used in OFDM.
            num_subcarriers (int): Number of subcarriers.
            channel_snr (float): The SNR to be used for the OFDM channel.
            los (bool): Set to True for LOS or False for NLOS channel.
        """
        self.SNR_dB = snr_dB_range
        self.M = modulation_order
        self.FFTSize = fft_size
        self.numSC = num_subcarriers
        self.numBitSymbol = self.numSC * np.log2(self.M)
        self.ber = np.zeros(len(self.SNR_dB))
        self.constellation_coder = ConstelationCoder(Modulation.PSK, self.M)
        self.channel = Channel(channel_snr, LOS=los)  # Initialize the channel with SNR and LOS
        self.utils = OFDMUtils()  # Initialize the utils class

    def applyDPSKEncoding(self, psk_signal):
        DPSK_signalTx = np.copy(psk_signal)
        for n in range(1, len(DPSK_signalTx)):
            DPSK_signalTx[n] *= DPSK_signalTx[n - 1]
        return DPSK_signalTx

    def applyDPSKDecoding(self, OFDM_signalRx):
        DPSK_signalRx = np.zeros(self.numSC, dtype=complex)
        DPSK_signalRx[0] = OFDM_signalRx[0]
        DPSK_signalRx[1:] = OFDM_signalRx[1:] * np.conj(OFDM_signalRx[:-1])
        return DPSK_signalRx
    
    def DPSK_encoder(self, signalTx):
        # PSK modulation
        psk_signal = self.constellation_coder.Encode(signalTx)

        # Differential encoding
        DPSK_signalTx = self.applyDPSKEncoding(psk_signal)
        
        return DPSK_signalTx

    
    def DPSK_channel_and_Noise(self, DPSK_signalTx, G, snr_dB):
        RxSignal = G @ DPSK_signalTx

        # Apply OFDM modulation
        OFDM_signalTx = self.utils.ofdm_modulate(RxSignal)

        # Add AWGN noise
        signalRx = self.utils.awgn_python(OFDM_signalTx, snr_dB, measured=False)
        
        return signalRx
    
    def DPSK_decoder(self, signalRx):
        # OFDM demodulation and differential decoding
        OFDM_signalRx = self.utils.ofdm_demodulate(signalRx)

        DPSK_signalRx = self.applyDPSKDecoding(OFDM_signalRx)
    
        # PSK demodulation
        signalEstimate = self.constellation_coder.Decode(DPSK_signalRx)
        
        return signalEstimate
        

    def transmit_and_receive(self, snr_dB):
        """
        Transmit and receive the signal through the channel with modulation and OFDM.
        Args:
            snr_dB (float): Signal-to-noise ratio in dB.
        Returns:
            tuple: Transmitted and estimated signals.
        """
        # Generate random data symbols
        signalTx = np.random.randint(0, 2, self.numSC*2)

        DPSK_signalTx = self.DPSK_encoder(signalTx)
        
        # Transmit through the channel
        G = self.channel.getChannel()
        signalRx = self.DPSK_channel_and_Noise(DPSK_signalTx, G, snr_dB)

        signalEstimate = self.DPSK_decoder(signalRx)

        return signalTx, signalEstimate

    def run_simulation(self):

        for i, snr_dB in enumerate(tqdm(self.SNR_dB, desc='SNR Loop')):
            num_error = 0
            num_bits = 0

            with tqdm(total=100, desc=f'SNR {snr_dB} dB', leave=False) as pbar:
                while num_error < 100:
                    signalTx, signalEstimate = self.transmit_and_receive(snr_dB)
                    errors = int(np.bitwise_xor(signalTx, signalEstimate).sum())
                    num_error += errors
                    num_bits += self.numBitSymbol
                    pbar.update(errors)

            self.ber[i] = num_error / num_bits

        self.utils.plot_ber(self.SNR_dB, self.ber, self.M, self.numBitSymbol)
        self.utils.save_ber_snr_to_csv(self.SNR_dB, self.ber)

"""
# Example usage:
# Assuming the channel dataset is already loaded and available.
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

ofdm_system.run_simulation()
"""