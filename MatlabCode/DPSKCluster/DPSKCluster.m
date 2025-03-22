close all;
clc;
addpath('../Libraries');
addpath('BinaryClassifier');

%% Parameter System
SNR_dB = 35:5:35;         % Range of SNR values in dB
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
numEpochs = 30;
miniBatchSize = 1;
learningRate = 0.001;

%% Application
ber = zeros(1, length(SNR_dB)); % Preallocate BER results
networks = cell(length(SNR_dB),M);

% Preallocate containers to store metrics and corresponding symbol indices
allMetrics = cell(length(SNR_dB), M-1, 1);
symbolVec = zeros(length(SNR_dB), M-1, 1);

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
    signalTx    = zeros(numSC, M, MaxNumOfSymbols);  
    signalToFix = zeros(numSC, MaxNumOfSymbols);
    labels      = zeros(numSC, MaxNumOfSymbols);
    
    for symbolsNumber = 1 : MaxNumOfSymbols
        % Generate random data symbols
        rawSymbols = generateRandomData(M, numSC);
        labels(:,symbolsNumber) = rawSymbols;

        % Create binary masks for each symbol value
        for symbol = 0:M-1
            signalTx(:, symbol + 1, symbolsNumber) = (rawSymbols ~= symbol);
        end

        % Transmit and receive the signal through the channel
        [label, signalEstimate] = processChannelAndTransmit(rawSymbols, M, FFTSize, SNR_dB(i), numSC,H);
        signalToFix(:,symbolsNumber) = (angle(signalEstimate)+pi)/(2*pi);

    end

    % Ensure labels(:,1) is a column vector (48x1)
    labelVector = labels(:,1); 

    % Extract the 48x4 matrix from signalTx
    signalFeatures = signalTx(:,:,1); % (48x4)

    % Stack them together to form a 48x5 matrix
    tabularData = [labelVector,signalToFix(:,1), signalFeatures]; % (48x5)

    % Convert to table format
    tabularTable = array2table(tabularData, 'VariableNames', ...
        {'Label','Phase','Feature1', 'Feature2', 'Feature3', 'Feature4'});

    % Display the table
    %disp(tabularTable);

    %return;

    %% Train Net with binary Net

    Predictions = zeros(M, numSC);
    GT = zeros(M, numSC);
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
        Predict = forward(tempNet, XTest(:,1));

        Predictions(constelationPoint,:) = Predict > mean(Predict);
        GT(constelationPoint,:) = YTrain(:,1);

        metrics = testNetwork(XTest, YTest, tempNet);
        networks{i, constelationPoint} = tempNet;

        % Store the metrics and symbol index
        allMetrics{i,constelationPoint} = metrics;
        symbolVec(i,constelationPoint) = constelationPoint;
    end


    figure;

    % 🔹 First plot (Original Predictions)
    subplot(1,2,1); % 1 row, 2 columns, first plot
    imagesc(Predictions); % Display as an image
    colormap('hot'); % Colormap for better visualization
    colorbar; % Show the color scale
    title('Predictions');
    xlabel('Columns');
    ylabel('Rows');
    
    % Second plot (Magnitude of the Predictions)
    subplot(1,2,2); % 1 row, 2 columns, second plot
    imagesc(GT); % Display magnitude
    colormap('hot'); % Different colormap for better contrast
    colorbar;
    title('GT');
    xlabel('Columns');
    ylabel('Rows');
    
    % Save the figure
    saveas(gcf, 'PredictionsPlot.png');
    return;
    
end

for i = 1:length(SNR_dB)
    % Ensure that symbolVec has the correct number of entries
    assert(length(symbolVec(i,:)) == length(allMetrics(i,:)), "Mismatch in symbolVec and allMetrics size!");
    % Accumulate all metrics into a table
    metricsTable = accumulateMetrics(allMetrics(i,:), symbolVec(i,:));
    disp(i);
    disp(metricsTable);
end

