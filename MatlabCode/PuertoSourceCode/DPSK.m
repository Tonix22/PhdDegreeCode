close all;
clc;

%% Parameter System
SNR_dB = 5:5:30;         % Range of SNR values in dB
M = 4;                   % Modulation order (QPSK)
FFTSize = 48;            % FFT size for OFDM
k = log2(M);             % Bits per symbol (log base 2 of modulation order)
numSC = 48;              % Number of subcarriers
numBitSymbol = numSC * k; % Total number of bits per OFDM symbol
H = load('../../Data/kaggle_dataset/v2v80211p_LOS.mat').vectReal32b;

ber = zeros(1, length(SNR_dB)); % Preallocate BER results

%% Main Loop for each SNR
for i = 1:length(SNR_dB)
    numError = 0;  % Counter for bit errors
    numBits = 0;   % Counter for total bits

    % Loop until we have enough bit errors for accurate BER calculation
    while numError < 1000
        % Transmit and receive the signal through the channel
        [signalTx, signalEstimate] = processChannelAndTransmit(H, M, FFTSize, SNR_dB(i), numSC);
        
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

%% Functions

% Generate random data symbols
function signalTx = generateRandomData(M, numSC)
    signalTx = randi([0 M-1], numSC, 1);
end

% Modulate data with PSK
function pskSignal = applyPSKModulation(signalTx, M)
    pskSignal = pskmod(signalTx, M, pi/2);
end

% Demodulate data with PSK
function signalEstimate = applyPSKDemodulation(DPSKsignalRx, M)
    signalEstimate = pskdemod(DPSKsignalRx, M, pi/2);
end

% Differentially encode signal for DPSK
function DPSKsignalTx = applyDPSKEncoding(pskSignal)
    DPSKsignalTx = pskSignal;
    for n = 2:length(DPSKsignalTx)
        DPSKsignalTx(n) = DPSKsignalTx(n) * DPSKsignalTx(n-1);
    end
end

% Differentially decode DPSK signal
function DPSKsignalRx = applyDPSKDecoding(OFDMsignalRx, numSC)
    DPSKsignalRx = complex(zeros(numSC, 1));
    DPSKsignalRx(1) = OFDMsignalRx(1); % First symbol remains unchanged
    DPSKsignalRx(2:end) = OFDMsignalRx(2:end) .* conj(OFDMsignalRx(1:end-1));
end

% OFDM modulation (IFFT)
function OFDMsignalTx = ofdmModulate(DPSKsignalTx, FFTSize)
    OFDMsignalTx = ifft(DPSKsignalTx, FFTSize);
end

% OFDM demodulation (FFT)
function OFDMsignalRx = ofdmDemodulate(signalRx, FFTSize)
    OFDMsignalRx = fft(signalRx, FFTSize);
end

% Encapsulates the process of transmitting and receiving the signal
function [signalTx, signalEstimate] = processChannelAndTransmit(H, M, FFTSize, SNR_dB, numSC)
    % Generate random data symbols
    signalTx = generateRandomData(M, numSC);
    
    % Modulate data with PSK
    pskSignal = applyPSKModulation(signalTx, M);

    % Differential encoding for DPSK
    DPSKsignalTx = applyDPSKEncoding(pskSignal);

    % Process channel
    G = processChannel(H);
    RxSignal = G * DPSKsignalTx;

    % OFDM modulation
    OFDMsignalTx = ofdmModulate(RxSignal, FFTSize);

    % Pass through AWGN channel using SNR
    signalRx = awgn(OFDMsignalTx, SNR_dB);

    % OFDM demodulation
    OFDMsignalRx = ofdmDemodulate(signalRx, FFTSize);

    % Differential decoding for DPSK
    DPSKsignalRx = applyDPSKDecoding(OFDMsignalRx, numSC);

    % PSK demodulation
    signalEstimate = applyPSKDemodulation(DPSKsignalRx, M);
end

% Process the channel and maintain an internal counter
function G = processChannel(H)
    persistent channelCont;
    if isempty(channelCont)
        channelCont = 1; % Start with the first channel
    end
    
    % Process the current channel
    G = H(:,:,channelCont);
    
    % Update the channel counter
    channelCont = channelCont + 1;
    
    % Reset the counter if it reaches 10000
    if channelCont == 10000
        channelCont = 1;
    end
end

% Calculate theoretical BER
function berTheoretical = calculateTheoreticalBER(SNR_dB, M, numBitSymbol)
    EbNo_dB = convertSNRtoEbNo(SNR_dB, numBitSymbol); % Convert SNR to Eb/No
    berTheoretical = berawgn(EbNo_dB, 'dpsk', M, 'nondiff'); % Theoretical BER
end

% Plots the BER graph and saves the figure as a file
function plotBER(SNR_dB, ber, M, numBitSymbol, plotName, berTheoretical)
    % Create the plot
    figure;
    
    % Plot theoretical BER if provided
    if nargin >= 6 && ~isempty(berTheoretical)
        semilogy(SNR_dB, berTheoretical, 'k-', 'LineWidth', 1.5); % Theoretical BER
        hold on;
    end
    
    % Plot estimated BER
    semilogy(SNR_dB, ber, 'b--', 'LineWidth', 1.5); % Estimated BER
    xlabel('SNR (dB)');
    ylabel('BER');
    legend('Theoretical BER', 'Estimated BER');
    grid on;
    
    % Save the plot as a file in the current directory
    if nargin < 4
        plotName = 'BER_plot_SNR'; % Default name if not provided
    end
    saveas(gcf, [plotName, '.png']); % Save as PNG file

    % Close the figure to prevent unnecessary display
    close(gcf);
end

% Save the BER data to a CSV file
function saveBERToCSV(SNR_dB, ber, csvFileName)
    % Combine SNR and BER data into a single matrix
    dataToSave = [SNR_dB(:), ber(:)];
    
    % If no file name is provided, use a default name
    if nargin < 3
        csvFileName = 'ber_data_SNR.csv';
    end
    
    % Save the data to a CSV file
    writematrix(dataToSave, csvFileName);

    % Display a message indicating the data was saved
    fprintf('Data saved to %s\n', csvFileName);
end

% Convert SNR to Eb/No
function EbNo_dB = convertSNRtoEbNo(SNR_dB, numBitSymbol)
    % Convert SNR (in dB) to Eb/No (in dB)
    SNR_linear = 10.^(SNR_dB / 10);    % Convert SNR from dB to linear scale
    EbNo_linear = SNR_linear / numBitSymbol; % Calculate Eb/No in linear scale
    EbNo_dB = 10 * log10(EbNo_linear); % Convert Eb/No to dB
end
