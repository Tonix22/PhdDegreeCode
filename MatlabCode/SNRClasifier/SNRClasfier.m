% filepath: /home/tonix/Documents/PhdDegreeCode/MatlabCode/SNRClasifier/SNRClasfier.m
close all;
clear all;
clc;
addpath('../Libraries');

%% Parameters
channelAWGN = 1; % Select if AWGN channel is used
FFTSize = 48;    % FFT size for OFDM
M = 4;           % Modulation order (QPSK)
k = log2(M);     % Bits per symbol
numSC = FFTSize; % Number of subcarriers
numBitSymbol = numSC * k; % Total bits per OFDM symbol
numFrames = 5000; % Number of frames for dataset generation
EbNo = 0:2:12;   % Range of EbNo values

if channelAWGN
    snrValues = EbNo + 10*log10(k);
else
    snrValues = EbNo + 10*log10(numBitSymbol);
    H = load('../../Data/kaggle_dataset/v2v80211p_LOS.mat').vectReal32b;
end

%% Dataset Generation
fprintf('Generating dataset...\n');
XData = zeros(numFrames * length(snrValues), 2, numSC); % Features: [real, imag] for each subcarrier
YData = zeros(numFrames * length(snrValues), 1);        % Labels: SNR values (normalized)

index = 1;
for i = 1:length(snrValues)
    currentSNR = snrValues(i);
    fprintf('Processing SNR = %d dB\n', currentSNR);
    
    for j = 1:numFrames
        % Generate random bits and convert to symbols
        signalTxBits = randi([0 1], numBitSymbol, 1);
        signalTx = bit2int(signalTxBits, k);
        
        % Transmit through the channel
        if channelAWGN
            [~, DPSKSignalEstimate] = processChannelAndTransmit(signalTx, M, FFTSize, currentSNR, numSC);
        else
            [~, DPSKSignalEstimate] = processChannelAndTransmit(signalTx, M, FFTSize, currentSNR, numSC, H);
        end
        
        % Stack real and imaginary parts into separate channels
        XData(index, 1, :) = real(DPSKSignalEstimate(:));
        XData(index, 2, :) = imag(DPSKSignalEstimate(:));
        YData(index) = (currentSNR - min(snrValues)) / (max(snrValues) - min(snrValues)); % Normalize SNR to [0, 1]
        index = index + 1;
    end
end

%% Split Dataset into Training and Testing
numTrain = round(0.8 * size(XData, 1));
indices = randperm(size(XData, 1));

% Reshape XData into [numSamples, 2 * numSC] for feature input
XDataReshaped = reshape(XData, size(XData, 1), []); % Flatten [2, numSC] into [2 * numSC]

XTrain = XDataReshaped(indices(1:numTrain), :);
YTrain = YData(indices(1:numTrain));
XTest = XDataReshaped(indices(numTrain+1:end), :);
YTest = YData(indices(numTrain+1:end));

% Verify dimensions
fprintf('XTrain size: [%d, %d]\n', size(XTrain));
fprintf('YTrain size: [%d]\n', numel(YTrain));
fprintf('XTest size: [%d, %d]\n', size(XTest));
fprintf('YTest size: [%d]\n', numel(YTest));

%% Define Neural Network Architecture
layers = [
    featureInputLayer(2 * numSC, 'Normalization', 'zscore', 'Name', 'input') % Input is [2 * numSC]
    fullyConnectedLayer(128, 'Name', 'fc1') % Fully connected layer
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(64, 'Name', 'fc2') % Fully connected layer
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(1, 'Name', 'fc3') % Output layer size is 1 for regression
    regressionLayer('Name', 'output')]; % Regression layer for MSE loss

%% Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress');

%% Train the Neural Network
fprintf('Training the neural network...\n');
net = trainNetwork(XTrain, YTrain, layers, options);

%% Evaluate the Network
fprintf('Evaluating the network...\n');
YPred = predict(net, XTest);

% Denormalize predictions and ground truth for comparison
YPredDenorm = YPred * (max(snrValues) - min(snrValues)) + min(snrValues);
YTestDenorm = YTest * (max(snrValues) - min(snrValues)) + min(snrValues);

% Assign the closest SNR value to each prediction
[~, YPredClosestIdx] = min(abs(YPredDenorm - snrValues), [], 2);
YPredClosest = snrValues(YPredClosestIdx);

% Assign the closest SNR value to the ground truth
[~, YTestClosestIdx] = min(abs(YTestDenorm - snrValues), [], 2);
YTestClosest = snrValues(YTestClosestIdx);

% Compute Mean Squared Error
mseError = mean((YPredDenorm - YTestDenorm).^2);
fprintf('Mean Squared Error: %.4f\n', mseError);

% Generate Confusion Matrix
fprintf('Generating confusion matrix...\n');
confMat = confusionmat(YTestClosest, YPredClosest);

% Plot Confusion Matrix
figure;
confusionchart(YTestClosest, YPredClosest, ...
    'Title', 'Confusion Matrix', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');

% Save Confusion Matrix as PNG
resultsFolder = 'SNRRegressionResults';
if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end
confusionMatrixFile = fullfile(resultsFolder, 'ConfusionMatrix.png');
saveas(gcf, confusionMatrixFile);
fprintf('Confusion matrix saved as: %s\n', confusionMatrixFile);

% Save predictions and ground truth
save(fullfile(resultsFolder, 'Predictions.mat'), 'YPredDenorm', 'YTestDenorm', 'YPredClosest', 'YTestClosest');
fprintf('Results saved in: %s\n', resultsFolder);