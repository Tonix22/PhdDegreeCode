import numpy as np
import matplotlib.pyplot as plt
import matlab.engine
import csv

class OFDMUtils:
    def __init__(self):
        self.eng = None

    def init_matlab_engine(self):
        if self.eng is None:
            self.eng = matlab.engine.start_matlab()
            self.eng.cd(r'/home/tonix/Documents/PhdDegreeCode/MatlabCode/PuertoSourceCode/Libraries', nargout=0)

    def ofdm_modulate(self, symbols):
        """
        Perform OFDM modulation using IFFT.
        Args:
            symbols (numpy.ndarray): Input symbols to modulate.
        Returns:
            numpy.ndarray: OFDM modulated signal.
        """
        return np.fft.ifft(symbols)

    def ofdm_demodulate(self, signal):
        """
        Perform OFDM demodulation using FFT.
        Args:
            signal (numpy.ndarray): OFDM signal to demodulate.
            Returns:
            numpy.ndarray: Demodulated symbols.
        """
        return np.fft.fft(signal)

    def calculate_theoretical_ber(self, SNR_dB, M, numBitSymbol):
        """
        Calculate the theoretical BER based on SNR and modulation scheme.
        """
        self.init_matlab_engine()
        # Convert Python variables to MATLAB types
        snr_dB_matlab = matlab.double([float(snr) for snr in SNR_dB])  # Convert list to MATLAB double array
        M_matlab = float(M)
        numBitSymbol_matlab = float(numBitSymbol)

        # Call the MATLAB function
        ber_theoretical = self.eng.calculateTheoreticalBER(snr_dB_matlab, M_matlab, numBitSymbol_matlab)

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

    def plot_ber(self, SNR_dB, ber_estimated, M, numBitSymbol, filename="BER_plot_SNR.png"):
        """
        Plot BER results and save to a file.
        """
        theoretical_ber = self.calculate_theoretical_ber(SNR_dB, M, numBitSymbol)
        plt.semilogy(SNR_dB, theoretical_ber, 'k-', linewidth=1.5, label="Theoretical BER")
        plt.semilogy(SNR_dB, ber_estimated, 'b--', linewidth=1.5, label="Estimated BER")
        plt.xlabel("SNR (dB)")
        plt.ylabel("BER")
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
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
