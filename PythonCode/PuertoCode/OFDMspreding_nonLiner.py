import numpy as np
import scipy.io
from commpy.modulation import QAMModem
from scipy.fftpack import fft, ifft
from commpy.utilities import bit_err_rate
from scipy.signal import awgn
import matplotlib.pyplot as plt

# Add additional dependencies for handling equalizers if needed

# %% Parameter 
# QAM modulation order
M = 4 
frame_size = 48

# Log2(M) bits per symbol
k = np.log2(M)
modem = QAMModem(M)  # QAM modulator
constellation = modem.constellation

# %% Load V2V channel data
channelLOS = scipy.io.loadmat('../../Data/kaggle_dataset/v2v80211p_LOS.mat')['vectReal32b']
channelNLOS = scipy.io.loadmat('../../Data/kaggle_dataset/v2v80211p_NLOS.mat')['vectReal32b']

# Create H tensor with interleaved matrices
H = np.zeros((48, 48, 20000))
H[:, :, 0::2] = channelLOS  # Assign LOS channels to odd indices
H[:, :, 1::2] = channelNLOS  # Assign NLOS channels to even indices

channel_cont = 0

# %% System model
SNR = np.arange(5, 50, 5)  # Range of SNR values, in dB
EffEsp = (np.log2(M) * 48 / 64)

# Define equalizers (You need to implement LMMSE, OSIC, NearML as separate classes or functions)
class Equalizer:
    def __init__(self, name, is_linear, handler):
        self.name = name
        self.is_linear = is_linear
        self.handler = handler

# Implement your equalizer handlers as functions like LMMSE, OSIC_Det, and QRM_Det4b

LMMSE = Equalizer("LMMSE", True, lambda H, noise_var, rx, N: lmmse_equalizer(H, noise_var, rx, N))
OSIC = Equalizer("OSIC", False, lambda H, rx: osic_equalizer(H, rx))
NearML = Equalizer("NearML", False, lambda yp, R, constellation, orden: qrm_det_equalizer(yp, R, constellation, orden))

eq_collection = [LMMSE, OSIC, NearML]

# Placeholder functions for equalizers
def lmmse_equalizer(H, noise_var, rx, N):
    # Implement the LMMSE equalizer algorithm here
    pass

def osic_equalizer(H, rx):
    # Implement the OSIC equalizer algorithm here
    pass

def qrm_det_equalizer(yp, R, constellation, orden):
    # Implement the NearML equalizer algorithm here
    pass

# %% Main loop for each equalizer
for equalizer in eq_collection:
    print(f"Processing Equalizer: {equalizer.name}")
    ber_est = []

    for snr_value in SNR:
        num_errors = 0
        num_bits = 0
        while num_errors < 1e3 and num_bits < 1e6:
            # Generate binary data and convert to symbols
            tx = np.random.randint(0, 2, int(frame_size * np.log2(M)))
            qpsk_sig = modem.modulate(tx)  # QAM modulation

            # Process channel
            G = H[:, :, channel_cont]
            channel_cont += 1
            if channel_cont == 10000:
                channel_cont = 0

            # Transmit signal processing
            TxSig = fft(qpsk_sig, frame_size)

            # Multiply by the channel in frequency domain
            RxSignal = np.dot(G, TxSig)
            H1 = ifft(G, frame_size)
            H1 = fft(H1.T).T  # Re-transforming channel matrix

            # Add AWGN to the received signal
            noise_var = 10 ** (-snr_value / 10)
            RxSignal_noisy = awgn(RxSignal, snr_value, 'measured')
            rx_sig = ifft(RxSignal_noisy, frame_size)

            # Handling equalizers
            if equalizer.is_linear:
                if equalizer.name == "LMMSE":
                    rx_sig = equalizer.handler(H1, noise_var, rx_sig, 48)
            else:
                if equalizer.name == "OSIC":
                    rx_sig = equalizer.handler(H1, rx_sig)
                elif equalizer.name == "NearML":
                    yp, R, orden = MMSESortedQRC(H1, noise_var, rx_sig, 48, 0)
                    rx_sig = equalizer.handler(yp, R, constellation, orden)

            # Demodulate received signal
            rx_bits = modem.demodulate(rx_sig, 'hard')

            # Calculate the number of bit errors
            n_errors = bit_err_rate(tx, rx_bits)

            # Update error and bit counters
            num_errors += n_errors
            num_bits += frame_size * np.log2(M)

        # Calculate BER for this SNR
        ber_est.append(num_errors / num_bits)
        print(f"SNR: {snr_value}, BER: {ber_est[-1]}")

    # Plot results for the current equalizer
    plt.semilogy(SNR, ber_est, label=equalizer.name)

# Show the BER plots
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('BER vs SNR for different equalizers')
plt.legend()
plt.grid(True)
plt.show()
