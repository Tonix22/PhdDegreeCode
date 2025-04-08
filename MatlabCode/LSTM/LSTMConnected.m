close all;
clear all;
clc;
addpath('../Libraries');

channelAWGN = 1; % Selecciona si es awgn o canal

%% 1. Crear carpeta de resultados "LSTMResults" si no existe
if channelAWGN
    resultsFolder = 'LSTMResultsAwgn';
else
    resultsFolder = 'LSTMResultsV2VChannel';
end

if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

%% 2. Parámetros comunes del sistema
EbNo = 0:2:12;           % Rango de EbNo

FFTSize = 48;            % Tamaño de la FFT para OFDM
M = 4;                   % Orden de modulación (QPSK)
k = log2(M);             % Bits por símbolo (para QPSK: 2)
numSC = FFTSize;              % Número de subportadoras
numBitSymbol = numSC * k; % Bits totales por símbolo OFDM
numFramesTrain = 5000;   % Número de tramas para generar datos de entrenamiento

if channelAWGN
    snrValues = EbNo + 10*log10(k);
else
    snrValues = EbNo + 10*log10(numBitSymbol); %#ok<UNRCH>
    H = load('../../Data/kaggle_dataset/v2v80211p_LOS.mat').vectReal32b;
end

% Prealocar vectores para almacenar BER (método raw y con red)
berRaw_vals = zeros(1, length(snrValues));
berNN_vals = zeros(1, length(snrValues));

%% 3. Bucle principal para cada SNR
for i = 1:length(snrValues)
    currentSNR = snrValues(i);
    fprintf('Procesando SNR = %d dB\n', currentSNR);
    
    %% 3.1 Generar datos de entrenamiento (secuencias)
    XTrainCell = cell(numFramesTrain,1);
    YTrainCell = cell(numFramesTrain,1);
    for j = 1:numFramesTrain
        % Generar bits fijos alternando entre 0 y 1
        signalTxBits = randi([0 1], numBitSymbol,1);
        signalTx = bit2int(signalTxBits, k);               % Convertir bits a símbolos
        
        % Transmitir por el canal al SNR actual
        if channelAWGN
            [Labels, DPSKSignalEstimate] = processChannelAndTransmit(signalTx, M, FFTSize, currentSNR, numSC);
        else
            % Uses Channel
            [Labels, DPSKSignalEstimate] = processChannelAndTransmit(signalTx, M, FFTSize, currentSNR, numSC, H); %#ok<UNRCH>
        end
        
        % Convertir DPSKSignalEstimate a secuencia: matriz de 2xnumSC (2 features: real e imaginario)
        featuresSeq = [ real(DPSKSignalEstimate(:)).' ; imag(DPSKSignalEstimate(:)).' ];  % [2 x numSC]
        
        %Convertir signalTx a etiquetas categóricas (ground truth)
        labelsSeq = categorical(signalTx(:).', 0:M-1);  % fila de 1 x numSC con clases {0, 1, 2, 3}
        
        XTrainCell{j} = featuresSeq;
        YTrainCell{j} = labelsSeq;
    end
    
    %% 3.2 Dividir en entrenamiento y prueba (secuencias)
    indices = randperm(numFramesTrain);
    numTrain = round(0.8 * numFramesTrain);
    Xtrain_seq = XTrainCell(indices(1:numTrain));
    Ytrain_seq = YTrainCell(indices(1:numTrain));
    Xtest_seq = XTrainCell(indices(numTrain+1:end));
    Ytest_seq = YTrainCell(indices(numTrain+1:end));
    
    %% 3.3 Definir la arquitectura de la red neuronal LSTM para secuencias
    layers = [
        sequenceInputLayer(2, 'Normalization','zscore','Name','input')
        lstmLayer(128, 'OutputMode','sequence','Name','lstm')
        fullyConnectedLayer(128, 'Name','fc1')
        reluLayer('Name','relu1')
        fullyConnectedLayer(32, 'Name','fc2')
        reluLayer('Name','relu2')
        fullyConnectedLayer(4, 'Name','fc3')
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classOutput')];

    %% 3.4 Opciones de entrenamiento
    options = trainingOptions('adam', ...
        'MaxEpochs', 9, ...
        'MiniBatchSize', 128, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', true, ...
        'Plots', 'none');
    
    %% 3.5 Entrenar la red neuronal con los datos de entrenamiento (secuencias)
    fprintf('Entrenando la red neuronal para SNR = %d dB...\n', currentSNR);
    net = trainNetwork(Xtrain_seq, Ytrain_seq, layers, options);

    % Guardar la red entrenada en un archivo .mat
    networkFileName = fullfile(resultsFolder, sprintf('LSTM_Network_SNR_%ddB.mat', currentSNR));
    save(networkFileName, 'net');
    fprintf('Red neuronal guardada en: %s\n', networkFileName);
    
    %% 3.6 Evaluación iterativa para calcular el BER y el Accuracy
    numError_raw = 0;    % Errores (método raw)
    numError_NN = 0;     % Errores (método con red)
    numBits = 0;         % Bits totales procesados

    % Contadores para accuracy (a nivel de símbolo)
    numSymbols_NN = 0;
    numCorrect_NN = 0;
    
    while (numError_raw < 10000 && numBits < 1e7)
        % Generar bits fijos alternando entre 0 y 1
        signalTxBits = repmat([0; 1], numBitSymbol / 2, 1); % Alterna entre 0 y 1
        signalTx = bit2int(signalTxBits, k);               % Convertir bits a símbolos
        
        % Ground truth para la comparación
        Ytrue = categorical(signalTx(:).');  % fila de 1 x numSC
        
        if channelAWGN
            [Labels, DPSKSignalEstimate] = processChannelAndTransmit(signalTx, M, FFTSize, currentSNR, numSC);
        else
            % Uses Channel
            [Labels, DPSKSignalEstimate] = processChannelAndTransmit(signalTx, M, FFTSize, currentSNR, numSC,H); %#ok<UNRCH>
        end
        
        % --- Método "raw" ---
        numErrorCalculate_raw = biterr(signalTx, Labels);
        
        % --- Método "con red" ---
        featuresSeq_eval = [ real(DPSKSignalEstimate(:)).' ; imag(DPSKSignalEstimate(:)).' ];
        YPredCell = classify(net, {featuresSeq_eval});
        YPred = YPredCell{1};  % vector de 1 x numSC
        
        % Contar errores
        numErrorCalculate_NN = sum(YPred ~= Ytrue);
        
        % Acumular errores y bits procesados
        numError_raw = numError_raw + numErrorCalculate_raw;
        numError_NN = numError_NN + numErrorCalculate_NN;
        numBits = numBits + numBitSymbol;  % Cada trama aporta numSC*k bits

        % Imprimir signalTxBits si numBits es múltiplo de 1000
        if mod(numBits, 1e6) == 0
            fprintf('numBits: %d, numerror: %d\n', numBits,numError_raw);
        end
        
        % Acumular para accuracy
        numCorrect_NN = numCorrect_NN + sum(YPred == Ytrue);
        numSymbols_NN = numSymbols_NN + numel(Ytrue);
    end
    
    % Calcular BER y accuracy
    BER_raw = numError_raw / numBits;
    BER_NN  = numError_NN / numBits;
    berRaw_vals(i) = BER_raw;
    berNN_vals(i) = BER_NN;
    
    accuracy_NN = numCorrect_NN / numSymbols_NN;
    
    fprintf('SNR = %d dB  --->  BER (Raw) = %e,  BER (NN) = %e, Accuracy (NN) = %.2f%%\n', ...
        currentSNR, BER_raw, BER_NN, accuracy_NN*100);
end

%% Guardar resultados en un archivo CSV
results = [snrValues(:), berRaw_vals(:), berNN_vals(:)];
csvFileName = fullfile(resultsFolder, 'LSTM_DPSK_Network.csv');
writematrix(results, csvFileName);

fprintf('Resultados guardados en: %s\n', csvFileName);