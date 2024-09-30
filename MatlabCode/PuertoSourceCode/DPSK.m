close all;
clc;

%% Parameter System
EbNo = (0:2:10);         % Range of Eb/No values
M = 4;                   % Modulation order (QPSK)
FFTSize = 48;            % FFT size for OFDM
k = log2(M);             % Bits per symbol (log base 2 of modulation order)
numSC = 48;              % Number of subcarriers
numBitSymbol = numSC * k; % Total number of bits per OFDM symbol
H  = load('../../Data/kaggle_dataset/v2v80211p_LOS.mat').vectReal32b;

ber = zeros(1, length(EbNo)); % Preallocate BER results

%% Main Loop for each Eb/No
for i = 1:length(EbNo)
    numError = 0;  % Counter for bit errors
    numBits = 0;   % Counter for total bits

    % Loop until we have enough bit errors for accurate BER calculation
    while numError < 1000
        
        % Generate transmitted signal
        signalTx = generateRandomData(M, numSC);
        
        % Modulate data with PSK
        pskSignal = applyPSKModulation(signalTx, M);

        % Differential encoding for DPSK
        DPSKsignalTx = applyDPSKEncoding(pskSignal);

        % OFDM modulation
        OFDMsignalTx = ofdmModulate(DPSKsignalTx, FFTSize);

        % Pass through AWGN channel
        SNR = convertSNR(EbNo(i), 'ebno', 'BitsPerSymbol', numBitSymbol); 
        signalRx = awgn(OFDMsignalTx, SNR); % Add noise
        
        % OFDM demodulation
        OFDMsignalRx = ofdmDemodulate(signalRx, FFTSize);

        % Differential decoding for DPSK
        DPSKsignalRx = applyDPSKDecoding(OFDMsignalRx, numSC);

        % Demodulate data with PSK
        signalEstimate = applyPSKDemodulation(DPSKsignalRx, M);
        
        % Calculate bit errors
        numErrorCalculate = biterr(signalTx, signalEstimate);  
        numError = numError + numErrorCalculate;  
        numBits = numBits + numBitSymbol;  

    end
    
    % Calculate BER for current Eb/No value
    ber(i) = numError / numBits;

end

%% Plot Results
plotBER(EbNo, ber, M,'DPSK');
saveBERToCSV(EbNo, ber, 'DPSk.csv')

%% Functions

% Generates random data symbols
function signalTx = generateRandomData(M, numSC)
    signalTx = randi([0 M-1], numSC, 1);
end

% Applies PSK modulation
function pskSignal = applyPSKModulation(signalTx, M)
    pskSignal = pskmod(signalTx, M, pi/2);
end

% Applies PSK demodulation
function signalEstimate = applyPSKDemodulation(DPSKsignalRx, M)
    signalEstimate = pskdemod(DPSKsignalRx, M, pi/2);
end

% Differentially encodes signal for DPSK
function DPSKsignalTx = applyDPSKEncoding(pskSignal)
    DPSKsignalTx = pskSignal;
    for n = 2:length(DPSKsignalTx)
        DPSKsignalTx(n) = DPSKsignalTx(n) * DPSKsignalTx(n-1);
    end
end

% Differentially decodes DPSK signal
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

% Plots the BER graph and saves the figure as a file
function plotBER(EbNo, ber, M, plotName)
    % Plot the theoretical BER
    berTheoretical = berawgn(EbNo, 'dpsk', M, 'nondiff'); % Theoretical BER
    
    % Create the plot
    figure;
    semilogy(EbNo, berTheoretical, 'k-', 'LineWidth', 1.5); % Theoretical BER
    hold on;
    semilogy(EbNo, ber, 'b--', 'LineWidth', 1.5); % Estimated BER
    xlabel('Eb/No (dB)');
    ylabel('BER');
    legend('Theoretical BER', 'Estimated BER');
    grid on;
    
    % Save the plot as a file in the current directory
    if nargin < 4
        plotName = 'BER_plot'; % Default name if not provided
    end
    saveas(gcf, [plotName, '.png']); % Save as PNG file

    % Close the figure to prevent unnecessary display
    close(gcf);
end

% Save the BER data to a CSV file
function saveBERToCSV(EbNo, ber, csvFileName)
    % Combine EbNo and BER data into a single matrix
    dataToSave = [EbNo(:), ber(:)];
    
    % If no file name is provided, use a default name
    if nargin < 3
        csvFileName = 'ber_data.csv';
    end
    
    % Save the data to a CSV file
    writematrix(dataToSave, csvFileName);

    % Display a message indicating the data was saved
    fprintf('Data saved to %s\n', csvFileName);
end

function G = processChannel(H)
    % A function that processes a channel from H and maintains an internal channel counter
    
    % Declare the persistent variable channelCont
    persistent channelCont;
    
    % Initialize channelCont the first time the function is called
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