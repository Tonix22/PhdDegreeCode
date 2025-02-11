close all;
clc;
addpath('Libraries');
%% Parameter System
SNR_dB = 5:5:30;         % Range of SNR values in dB
M = 4;
k = log2(M);             % Bits per symbol (log base 2 of modulation order)
numSC = 48;              % Number of subcarriers
numBitSymbol = numSC * k; % Total number of bits per OFDM symbol

berTheoretical = calculateTheoreticalBER(SNR_dB, M, numBitSymbol);
saveBERToCSV(SNR_dB, berTheoretical, 'DPSK_SNR_Theorical_BER_AWGN.csv');