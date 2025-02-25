close all;
clc;
addpath('../PuertoSourceCode/Libraries');
addpath('BinaryClassifier');

%% Parameter System
SNR_dB = 5:5:30;         % Range of SNR values in dB
M = 4;                   % Modulation order (QPSK)
FFTSize = 48;            % FFT size for OFDM
k = log2(M);             % Bits per symbol (log base 2 of modulation order)
numSC = 48;              % Number of subcarriers
numBitSymbol = numSC * k; % Total number of bits per OFDM symbol
H = load('../../Data/kaggle_dataset/v2v80211p_LOS.mat').vectReal32b;

MaxNumOfSymbols = 1000;
%% Neuronal Network
templateNet = load("BinaryClassifier/BinaryNet.mat").net;
% hyperameters
numEpochs = 10;
miniBatchSize = 10;
learningRate = 0.001;

%% Application
ber = zeros(1, length(SNR_dB)); % Preallocate BER results
networks = cell(length(SNR_dB),M);

for i = 1:length(SNR_dB)
    for j = 1:M
        % Create and assign your dlnetwork object.
        % Replace "layerGraph" with your actual layer graph or construction method.
        networks{i, j} = templateNet;
    end
end

%% Main Loop for each SNR
for i = 1:length(SNR_dB)

    % 3D array for binary masks
    signalTx = zeros(numSC, M, MaxNumOfSymbols);  
    signalToFix = zeros(numSC, MaxNumOfSymbols);
    
    for symbolsNumber = 1 : MaxNumOfSymbols
        % Generate random data symbols
        rawSymbols = generateRandomData(M, numSC);

        % Create binary masks for each symbol value
        for symbol = 0:M-1
            signalTx(:, symbol + 1, symbolsNumber) = (rawSymbols == symbol);
        end

        % Transmit and receive the signal through the channel
        [signalEstimate, ~] = processChannelAndTransmit(rawSymbols, M, FFTSize, SNR_dB(i), numSC,H);
        signalToFix(:,symbolsNumber) = signalEstimate;
    end

    %% Train Net with binary Net
    % Preallocate containers to store metrics and corresponding symbol indices
    allMetrics = cell(M-1, 1);
    symbolVec = zeros(M-1, 1);

    for constelationPoint = 1 : M
        % Reshape data to ensure the batch dimension is along columns.
        Y = reshape(signalTx(:, constelationPoint, :), [numSC, MaxNumOfSymbols]);
        X_dl = dlarray(signalToFix, 'CB');
        Y_dl = dlarray(round(Y), 'CB');
        
        % Split data into training and test sets.
        [XTrain, XTest, YTrain, YTest] = splitDlData(X_dl, Y_dl);
        
        % Train the network using the training split.
        tempNet = networks{i, constelationPoint};
        tempNet = trainNetworkCustom(tempNet, XTrain, YTrain, ...
                                        numEpochs, miniBatchSize, learningRate, @modelLoss);
        
        % Evaluate the trained network on the test split.
        metrics = testNetwork(XTest, YTest, tempNet);
        networks{i, constelationPoint} = tempNet;

        % Store the metrics and symbol index
        allMetrics{constelationPoint} = metrics;
        symbolVec(constelationPoint) = constelationPoint;
    end
    % Ensure that symbolVec has the correct number of entries
    assert(length(symbolVec) == length(allMetrics), "Mismatch in symbolVec and allMetrics size!");
    % Accumulate all metrics into a table
    metricsTable = accumulateMetrics(allMetrics, symbolVec);
    disp(metricsTable);
    return;
end

