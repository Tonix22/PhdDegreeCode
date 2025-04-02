close all;
clc;
addpath('../Libraries');

%% Parameter System
SNR_dB = 0:5:15;         % Range of SNR values in dB
M = 4;                   % Modulation order (QPSK)
FFTSize = 48;            % FFT size for OFDM
k = log2(M);             % Bits per symbol (log base 2 of modulation order)
numSC = 48;              % Number of subcarriers
numBitSymbol = numSC * k; % Total number of bits per OFDM symbol
H = load('../../Data/kaggle_dataset/v2v80211p_LOS.mat').vectReal32b;

ber = zeros(1, length(SNR_dB)); % Preallocate BER results

%% Main Loop for each SNR
fprintf('Processing SNR values:\n');
for i = 1:length(SNR_dB)
    numError = 0;  % Counter for bit errors
    numBits = 0;   % Counter for total bits

    % Loop until we have enough bit errors for accurate BER calculation
    while numError < 10000 && numBits < 1e7
        % Generate random data symbols
        signalTxBits = repmat([0; 1], numBitSymbol / 2, 1); % Alterna entre 0 y 1
        signalTx = bit2int(signalTxBits, k);          % Convert bits to symbols
        
        % Transmit and receive the signal through the channel
        [signalEstimate, ~] = processChannelAndTransmit(signalTx, M, FFTSize, SNR_dB(i), numSC);
        
        % Convert received symbols to bits
        signalEstimateBits = int2bit(signalEstimate, k);
        
        % Calculate bit errors
        numErrorCalculate = biterr(signalTxBits, signalEstimateBits);  
        numError = numError + numErrorCalculate;  
        numBits = numBits + numBitSymbol;  

        % Imprimir signalTxBits si numBits es múltiplo de 1000
        if mod(numBits, 1e6) == 0
            fprintf('numBits: %d, numerror: %d\n', numBits,numError);
        end
    end
    
    % Calculate BER for the current SNR value
    ber(i) = numError / numBits;

    % Actualizar barra de progreso en CLI
    fprintf('SNR %d/%d: %.2f%% complete\n', i, length(SNR_dB), (i / length(SNR_dB)) * 100);
end

%% Plot Results
berTheoretical = calculateTheoreticalBER(SNR_dB, M); % Optional calculation of theoretical BER
plotBER(SNR_dB, ber, M, numBitSymbol, 'DPSK_SNR', berTheoretical);
saveBERToCSV(SNR_dB, ber, 'DPSK_SNR.csv');