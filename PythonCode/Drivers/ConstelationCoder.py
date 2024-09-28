from enum import Enum
import numpy as np
from commpy.modulation import PSKModem, QAMModem  # Assuming these classes are defined in commpy.modulation

# Define an enumeration for different modulation types.
class Modulation(Enum):
    PSK = 0  # Phase Shift Keying
    QAM = 1  # Quadrature Amplitude Modulation

# Class for encoding and decoding signals based on modulation schemes.
class ConstelationCoder:
    def __init__(self, modulation, constelationSize):
        """
        Initialize the constellation coder with the specified modulation type and constellation size.
        Args:
            modulation (Modulation): The modulation type (PSK or QAM) defined in the Modulation Enum.
            constelationSize (int): The size of the constellation, e.g., 2 for BPSK, 4 for QPSK, etc.
        """
        # Select the appropriate modem based on the modulation type.
        if modulation == Modulation.PSK:
            self.modem = PSKModem(constelationSize)  # Initialize PSK modem with the given constellation size.
        
        if modulation == Modulation.QAM:
            self.modem = QAMModem(constelationSize)  # Initialize QAM modem with the given constellation size.

    def Encode(self, tx):
        """
        Encode the input signal using the selected modulation scheme.
        Args:
            tx (array-like): The input binary data to be modulated.
        Returns:
            np.ndarray: The modulated signal symbols normalized by average power.
        """
        # Modulate the input binary data into signal symbols.
        self.signal_symbols = self.modem.modulate(tx)
        
        # Calculate the average power of the modulated symbols.
        self.average_power = np.sqrt(np.mean(np.abs(self.signal_symbols)**2))
        
        # Normalize the signal symbols to ensure unit average power.
        self.signal_symbols /= self.average_power
        
        # Return the normalized signal symbols.
        return self.signal_symbols

    def Decode(self, rx):
        """
        Decode the received signal using the selected demodulation scheme.
        Args:
            rx (array-like): The received signal symbols to be demodulated.
        Returns:
            array-like: The demodulated binary data.
        """
        # Scale the received symbols back to the original power level.
        rx = rx * self.average_power
        
        # Demodulate the received symbols to recover the transmitted data.
        self.signal_symbols = self.modem.demodulate(rx,'hard')
        
        # Return the received symbols after scaling.
        return self.signal_symbols
