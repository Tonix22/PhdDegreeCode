import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import os
import sys
import matlab.engine
import commpy.channels as chan 
import csv
from tqdm import tqdm
import numpy as np


eng = None

# Add sys.path for accessing the required modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Drivers'))
sys.path.insert(0, parent_dir)

from ConstelationCoder import ConstelationCoder, Modulation  # type: ignore
from SignalChannelResponse import Channel  # Import the custom Channel class

class OFDMSystem:
    def __init__(self, snr_dB_range, modulation_order, fft_size, num_subcarriers, channel_snr, los=True):
        """
        Initialize the OFDM system parameters.
        Args:
            snr_dB_range (list): Range of SNR values in dB.
            modulation_order (int): Modulation order (e.g., 4 for QPSK).
            fft_size (int): Size of the FFT used in OFDM.
            num_subcarriers (int): Number of subcarriers.
            channel_snr (float): The SNR to be used for the channel.
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
    
    def init_matlab_engine(self):
        global eng
        if(eng == None):
            eng = matlab.engine.start_matlab()
            eng.cd(r'/home/tonix/Documents/PhdDegreeCode/MatlabCode/PuertoSourceCode/Libraries', nargout=0)

    def ofdm_modulate(self, symbols):
        """
        Perform OFDM modulation using IFFT.
        Args:
            symbols (numpy.ndarray): Input symbols to modulate.
        Returns:
            numpy.ndarray: OFDM modulated signal.
        """
        return np.fft.ifft(symbols, self.FFTSize)

    def ofdm_demodulate(self, signal):
        """
        Perform OFDM demodulation using FFT.
        Args:
            signal (numpy.ndarray): OFDM signal to demodulate.
        Returns:
            numpy.ndarray: Demodulated symbols.
        """
        return np.fft.fft(signal, self.FFTSize)

    def applyDPSKEncoding(self,psk_signal):
        DPSK_signalTx = np.copy(psk_signal)
        for n in range(1, len(DPSK_signalTx)):
            DPSK_signalTx[n] *= DPSK_signalTx[n - 1]
        return DPSK_signalTx
    
    def applyDPSKDecoding(self,OFDM_signalRx):
        DPSK_signalRx = np.zeros(self.numSC, dtype=complex)
        DPSK_signalRx[0] = OFDM_signalRx[0]
        DPSK_signalRx[1:] = OFDM_signalRx[1:] * np.conj(OFDM_signalRx[:-1])
        return DPSK_signalRx

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

        # PSK modulation
        psk_signal = self.constellation_coder.Encode(signalTx)
        
        # Differential encoding (simple version)
        DPSK_signalTx = self.applyDPSKEncoding(psk_signal)
        
        # Transmit through the channel
        G = self.channel.getChannel()
        RxSignal = G @ DPSK_signalTx
        
        # Apply OFDM modulation
        OFDM_signalTx = self.ofdm_modulate(RxSignal)
        
        signalRx = self.awgn_python(OFDM_signalTx, snr_dB, measured=False)

        # OFDM demodulation and differential decoding
        OFDM_signalRx = self.ofdm_demodulate(signalRx)

        DPSK_signalRx = self.applyDPSKDecoding(OFDM_signalRx)
        
        # PSK demodulation
        signalEstimate = self.constellation_coder.Decode(DPSK_signalRx)

        return signalTx, signalEstimate

    def calculate_theoretical_ber(self):
        """
        Calculate the theoretical BER based on SNR and modulation scheme.
        """
        self.init_matlab_engine()
        # Convert Python variables to MATLAB types
        snr_dB_matlab = matlab.double([float(snr) for snr in self.SNR_dB])  # Convert list to MATLAB double array
        M_matlab = float(self.M)
        numBitSymbol_matlab = float(self.numBitSymbol)

        # Call the MATLAB function
        ber_theoretical = eng.calculateTheoreticalBER(snr_dB_matlab, M_matlab, numBitSymbol_matlab)
        
        # Convert the result back to a NumPy array
        return np.array(ber_theoretical).squeeze()


    def convert_snr_to_ebno(self, SNR_dB, num_bit_symbol):
        """
        Convert SNR to Eb/No.
        Args:
            SNR_dB (float): SNR in dB.
            num_bit_symbol (int): Number of bits per symbol.
        Returns:
            float: Eb/No in dB.
        """
        SNR_linear = 10 ** (np.array(SNR_dB) / 10)
        EbNo_linear = SNR_linear / num_bit_symbol
        EbNo_dB = 10 * np.log10(EbNo_linear)
        return EbNo_dB

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

        self.plot_ber()
        self.save_ber_snr_to_csv(self.SNR_dB, self.ber)


    def plot_ber(self):
        """
        Plot BER results and save to a file.
        """
        theoretical_ber = self.calculate_theoretical_ber()
        plt.semilogy(self.SNR_dB, theoretical_ber, 'k-', linewidth=1.5, label="Theoretical BER")
        plt.semilogy(self.SNR_dB, self.ber, 'b--', linewidth=1.5, label="Estimated BER")
        plt.xlabel("SNR (dB)")
        plt.ylabel("BER")
        plt.legend()
        plt.grid(True)
        plt.savefig("BER_plot_SNR.png")
        plt.show()
        
    def save_ber_snr_to_csv(self, snr_values, ber_values, filename="ber_snr_data.csv"):
        """
        Save SNR and BER values to a CSV file.

        Args:
            snr_values (list or np.ndarray): The SNR values.
            ber_values (list or np.ndarray): The corresponding BER values.
            filename (str): The name of the CSV file to save the data. Default is 'ber_snr_data.csv'.
        """
        # Ensure that snr_values and ber_values are NumPy arrays for compatibility
        snr_values = np.array(snr_values)
        ber_values = np.array(ber_values)

        # Write the SNR and BER data to a CSV file
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["SNR (dB)", "BER"])  # Write the header
            for snr, ber in zip(snr_values, ber_values):
                writer.writerow([snr, ber])  # Write each row of SNR and BER values

        print(f"Data saved to {filename}")


    def awgn_python(self, signal, snr_dB, sig_power_dB=0, measured=False, seed=None):
        """
        Add AWGN noise to a signal, replicating MATLAB's awgn function behavior.

        Args:
            signal (np.ndarray): Input signal (can be real or complex).
            snr_dB (float): Desired Signal-to-Noise Ratio in dB.
            sig_power_dB (float, optional): Signal power in dBW. Default is 0 dBW (1 watt).
            measured (bool, optional): If True, measure the signal power.
            seed (int, optional): Seed for random number generator.

        Returns:
            np.ndarray: Noisy signal.
        """
        if seed is not None:
            np.random.seed(seed)

        # Measure signal power if requested
        if measured:
            sig_power_linear = np.mean(np.abs(signal) ** 2)
        else:
            # Convert sig_power_dB to linear scale
            sig_power_linear = 10 ** (sig_power_dB / 10)

        # Convert SNR from dB to linear scale
        snr_linear = 10 ** (snr_dB / 10)

        # Calculate noise power
        noise_power = sig_power_linear / snr_linear

        # Generate white Gaussian noise
        if np.iscomplexobj(signal):
            # For complex signals, generate noise for real and imaginary parts
            noise_std = np.sqrt(noise_power / 2)
            noise = noise_std * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
        else:
            noise_std = np.sqrt(noise_power)
            noise = noise_std * np.random.randn(*signal.shape)

        # Add noise to the signal
        noisy_signal = signal + noise

        return noisy_signal


    
# Example usage:
# Assuming the channel dataset is already loaded and available.
H = np.load('/home/tonix/Documents/PhdDegreeCode/Data/kaggle_dataset/v2v80211p_LOS.npy')  # Placeholder for channel matrix

# Initialize and run the simulation
ofdm_system = OFDMSystem(snr_dB_range=[5, 10, 15,25,30], modulation_order=4, fft_size=48, num_subcarriers=48, channel_snr=10, los=True)
ofdm_system.run_simulation()
