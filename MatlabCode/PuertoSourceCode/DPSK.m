close all;
clc;
addpath('Libraries');
%% Parameter System
SNR_dB = 5:5:30;         % Range of SNR values in dB
M = 4;                   % Modulation order (QPSK)
FFTSize = 48;            % FFT size for OFDM
k = log2(M);             % Bits per symbol (log base 2 of modulation order)
numSC = 48;              % Number of subcarriers
numBitSymbol = numSC * k; % Total number of bits per OFDM symbol
%H = load('../../Data/kaggle_dataset/v2v80211p_LOS.mat').vectReal32b;

ber = zeros(1, length(SNR_dB)); % Preallocate BER results

%% Main Loop for each SNR
for i = 1:length(SNR_dB)
    numError = 0;  % Counter for bit errors
    numBits = 0;   % Counter for total bits

    % Loop until we have enough bit errors for accurate BER calculation
    while numError < 1000
        % Generate random data symbols
        signalTx = generateRandomData(M, numSC);
        % Transmit and receive the signal through the channel
        [signalEstimate] = processChannelAndTransmit(signalTx, M, FFTSize, SNR_dB(i), numSC);
        
        % Calculate bit errors
        numErrorCalculate = biterr(signalTx, signalEstimate);  
        numError = numError + numErrorCalculate;  
        numBits = numBits + numBitSymbol;  
    end
    
    % Calculate BER for the current SNR value
    ber(i) = numError / numBits;
end

%% Plot Results
berTheoretical = calculateTheoreticalBER(SNR_dB, M, numBitSymbol); % Optional calculation of theoretical BER
plotBER(SNR_dB, ber, M, numBitSymbol, 'DPSK_SNR', berTheoretical);
saveBERToCSV(SNR_dB, ber, 'DPSK_SNR.csv');
