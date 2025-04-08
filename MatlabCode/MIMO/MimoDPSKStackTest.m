%% DPSKStack_NN_BER.m
% This script generates DPSK stacked signals over a range of SNRs,
% feeds each row of the stack into a trained neural network, and computes
% the resulting Bit Error Rate (BER).

close all;
clc;
addpath('../Libraries');  % Ensure your helper functions are accessible

% Leer JSON desde argumento de ejecución
p = inputParser;
addRequired(p, 'jsonPath', @ischar);
parse(p, getenv('jsonPath'));

jsonPath = p.Results.jsonPath;

if exist(jsonPath, 'file') ~= 2
    error('El archivo JSON no existe: %s', jsonPath);
end

% Leer el archivo JSON
jsonText = fileread(jsonPath);
config = jsondecode(jsonText);

% Asignar parámetros desde JSON
EbNo = config.EbNo(:);  % Asegurar que sea un vector columna
M = config.M;                        % Modulation order (QPSK)
FFTSize = config.FFTSize;            % FFT size for OFDM
Retransmitions = config.Retransmissions;  % Number of retransmissions
k = log2(M);                         % Bits per symbol (log base 2 of modulation order)
numSC = config.numSC;                 % Number of subcarriers
numBitSymbol = numSC * k;             % Total number of bits per OFDM symbol
samplesPerSNR = config.samplesPerSNR; % Samples per SNR value
V2VChannel = config.V2VChannel;
PTHBasePath = config.PTHBasePath;
errorThreshold = config.errorThreshold;
bitsThreshold = config.bitsThreshold;

% Cargar el canal desde el archivo MAT si está habilitado
if islogical(V2VChannel) && V2VChannel
    disp('USING CHANNEL')
    H = load('../../Data/kaggle_dataset/v2v80211p_LOS.mat').vectReal32b;
    SNR_dB_Range = EbNo + 10*log10(numBitSymbol);
else
    SNR_dB_Range = EbNo + 10*log10(k);
    disp('NO CHANNEL')
end

%% Add the Python file directory to Python's search path
mypythonpath = '../../PythonCode/DeepLearning/MIMOSolution';
insert(py.sys.path, int32(0), mypythonpath);

%% 3. Initialize BER Storage
ber = zeros(1, length(SNR_dB_Range));

%% 4. Main Loop: Process each SNR value
%% 4. Main Loop: Process each SNR value
for idx = 1:length(SNR_dB_Range)
    currentSNR = SNR_dB_Range(idx);
    currentEbno = EbNo(idx);
    % Create an instance of the ModelLoader class from Python
    pthFile = PTHBasePath + "model_MIMO_DPSK_" + currentEbno + ".pth";
    model_loader = py.ModelLoader.ModelLoader(pthFile, py.float(1e-3), "cuda");

    numError = 0;  % Accumulated bit errors
    numBits = 0;   % Accumulated total bits processed

    % Initialize arrays to store the true and predicted symbol labels
    allTrueSymbols = [];
    allPredSymbols = [];
    
    fprintf('Processing EbNo = %d dB...\n', currentEbno);
    
    % Continue processing samples until errorThreshold is reached
    while (numError < errorThreshold && numBits < bitsThreshold)
        % Generate random transmitted symbols (1 x numSC)
        signalTx = generateRandomData(M, numSC);  
        % signalTx is assumed to contain symbol labels (e.g., 0,1,2,3) for each subcarrier
        
        % Generate DPSK stack sample (dimensions: Retransmissions+1 x FFTSize)
        sampleStack = zeros(Retransmitions+1, FFTSize);
        for row = 1:Retransmitions
            if islogical(V2VChannel) && V2VChannel
                [~, DPSKsignalRx] = processChannelAndTransmit(signalTx, M, FFTSize, currentSNR, numSC, H);
            else
                [~, DPSKsignalRx] = processChannelAndTransmit(signalTx, M, FFTSize, currentSNR, numSC);
            end
            sampleStack(row, :) = DPSKsignalRx;
        end

        for i = 1:FFTSize
            symbols = sampleStack(1:Retransmitions,i);  % Complex numbers
            mean_angle = angle(mean(exp(1j * angle(symbols))));
            sampleStack(1:Retransmitions,i) = angle(symbols);
            sampleStack(end,i) = mean_angle;
            sampleStack(:,i) = (sampleStack(:,i) + pi) / (2*pi);
        end

        % Use the neural network to predict the transmitted symbols from the stack.
        predictedSymbols = zeros(1, FFTSize);
        outputArray = zeros(4, FFTSize);
        for col = 1:FFTSize
            colData = sampleStack(:, col);
            sampleInputMat = reshape(colData, [1, 1, Retransmitions+1]);
            
            % Convert MATLAB array to a Python tensor
            sampleInputPy = py.torch.tensor(...
                py.numpy.array(sampleInputMat, pyargs('dtype', py.numpy.float32)), ...
                pyargs('device', 'cuda') );
            
            outputPy = model_loader.predict(sampleInputPy);
            outputArray(:,col) = double(outputPy.cpu().numpy());
            [~, pred] = max(outputArray(:,col));
            predictedSymbols(col) = pred-1;  % MATLAB indexes from 1, so subtract one
        end
        
        % Store the labels for confusion matrix (ensure column vectors)
        allTrueSymbols = [allTrueSymbols; signalTx'];  
        allPredSymbols = [allPredSymbols; predictedSymbols'];
        
        % Count symbol errors and update BER computation
        symbolErrors = sum(predictedSymbols ~= signalTx');
        numError = numError + symbolErrors * log2(M);
        numBits = numBits + numSC * log2(M);
        if(mod(numBits,10000) == 0)
            fprintf('numError = %d, numBits = %.4e\n', numError, numBits);
        end
    end
    
    % Compute the BER for the current SNR value
    ber(idx) = numError / numBits;
    fprintf('SNR = %d dB, BER = %.4e\n', currentSNR, ber(idx));
    
    %% Plot and Save the Confusion Matrix for the current SNR
    % Ensure true and predicted symbols are column vectors
    trueVec = allTrueSymbols(:); % Flatten to a column vector
    predVec = allPredSymbols(:); % Flatten to a column vector

    % Check if trueVec and predVec are non-empty and have the same length
    if isempty(trueVec) || isempty(predVec)
        warning('True or predicted symbols are empty for SNR = %d dB. Skipping confusion matrix.', currentSNR);
        continue;
    elseif length(trueVec) ~= length(predVec)
        error('Mismatch in dimensions: trueVec (%d) and predVec (%d) for SNR = %d dB.', ...
            length(trueVec), length(predVec), currentSNR);
    end

    % Compute the confusion matrix
    confMat = confusionmat(trueVec, predVec);

    % Create a confusion chart
    hFig = figure;
    cmChart = confusionchart(confMat, 0:(M-1)); % Specify the label range [0, 1, ..., M-1]
    cmChart.Title = sprintf('%d dB Confusion Matrix', currentSNR);
    cmChart.RowSummary = 'row-normalized';
    cmChart.ColumnSummary = 'column-normalized';

    % Save the confusion matrix as a PNG file
    filename = sprintf('%d_confusionmatrix.png', currentSNR);
    saveas(hFig, filename);
    disp(['Confusion matrix saved as: ' filename]);
    close(hFig);
end

%% 5. Plot BER vs. SNR
%% Plot Results
berTheoretical = calculateTheoreticalBER(SNR_dB_Range, M); % Optional calculation of theoretical BER
plotBER(SNR_dB_Range, ber, M, numBitSymbol, 'DPSK_SNR_MIMONET', berTheoretical);
saveBERToCSV(SNR_dB_Range, ber, 'DPSK_SNR_MIMONET.csv');