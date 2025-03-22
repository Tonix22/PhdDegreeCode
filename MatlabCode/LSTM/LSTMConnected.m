close all;
clc;
addpath('../Libraries');

%% 1. Crear carpeta de resultados "FCResults" si no existe
resultsFolder = 'LSTMResults';
if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

%% 2. Parámetros comunes del sistema
snrValues = 0:5:25;      % Rango de SNR en dB
M = 4;                   % Orden de modulación (QPSK)
FFTSize = 48;            % Tamaño de la FFT para OFDM
k = log2(M);             % Bits por símbolo (para QPSK: 2)
numSC = 48;              % Número de subportadoras
numBitSymbol = numSC * k; % Bits totales por símbolo OFDM
numFramesTrain = 5000;   % Número de tramas para generar datos de entrenamiento
H = load('../../Data/kaggle_dataset/v2v80211p_LOS.mat').vectReal32b;

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
        % Generar símbolos aleatorios para una trama OFDM de numSC subportadoras
        signalTx = generateRandomData(M, numSC);
        % Transmitir por el canal al SNR actual
        [Labels, DPSKSignalEstimate] = processChannelAndTransmit(signalTx, M, FFTSize, currentSNR, numSC);
        
        % Convertir DPSKSignalEstimate a secuencia: matriz de 2xnumSC (2 features: real e imaginario)
        featuresSeq = [ real(DPSKSignalEstimate(:)).' ; imag(DPSKSignalEstimate(:)).' ];  % [2 x numSC]
        % Usar las etiquetas demoduladas (Labels) para la secuencia
        labelsSeq = categorical(Labels(:).');  % fila de 1 x numSC
        
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
        bilstmLayer(128, 'OutputMode','sequence','Name','bilstm')
        dropoutLayer(0.2, 'Name','dropout1')
        fullyConnectedLayer(128, 'Name','fc1')
        batchNormalizationLayer('Name','bn1')
        reluLayer('Name','relu1')
        dropoutLayer(0.2, 'Name','dropout2')
        fullyConnectedLayer(32, 'Name','fc2')
        batchNormalizationLayer('Name','bn3')
        reluLayer('Name','relu2')
        dropoutLayer(0.2, 'Name','dropout3')
        fullyConnectedLayer(4, 'Name','fc3')
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classOutput')];
    
    
    
    
    %% 3.4 Opciones de entrenamiento
    options = trainingOptions('adam', ...
        'MaxEpochs', 10, ...
        'MiniBatchSize', 128, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', false, ...
        'Plots', 'none');
    
    %% 3.5 Entrenar la red neuronal con los datos de entrenamiento (secuencias)
    net = trainNetwork(Xtrain_seq, Ytrain_seq, layers, options);
    
    %% 3.6 Evaluación iterativa para calcular el BER y el Accuracy
    % Se evaluarán:
    %   - BER "raw": usando DPSKSignalEstimate directamente y la función biterr,
    %   - BER "NN": usando la salida clasificada por la red.
    numError_raw = 0;    % Errores (método raw)
    numError_NN = 0;     % Errores (método con red)
    numBits = 0;         % Bits totales procesados

    % Contadores para accuracy (a nivel de símbolo)
    numSymbols_NN = 0;
    numCorrect_NN = 0;
    
    while (numError_raw < 10000 && numBits < 1e6)
        % Generar una nueva trama de prueba
        signalTx = generateRandomData(M, numSC);
        % Ground truth para la comparación (usamos signalTx o Labels según convenga)
        Ytrue = categorical(signalTx(:).');  % fila de 1 x numSC
        
        [Labels, DPSKSignalEstimate] = processChannelAndTransmit(signalTx, M, FFTSize, currentSNR, numSC);
        
        % --- Método "raw" ---
        numErrorCalculate_raw = biterr(signalTx, Labels);
        
        % --- Método "con red" ---
        % Convertir DPSKSignalEstimate a secuencia: [2 x numSC]
        featuresSeq_eval = [ real(DPSKSignalEstimate(:)).' ; imag(DPSKSignalEstimate(:)).' ];
        % Clasificar usando la red (se pasa la secuencia dentro de una celda)
        YPredCell = classify(net, {featuresSeq_eval});
        YPred = YPredCell{1};  % vector de 1 x numSC
        
        % Para fines de comparación, se cuentan los errores a nivel de símbolo
        numErrorCalculate_NN = sum(YPred ~= Ytrue);
        
        % Acumular errores y bits procesados
        numError_raw = numError_raw + numErrorCalculate_raw;
        numError_NN = numError_NN + numErrorCalculate_NN;
        numBits = numBits + numBitSymbol;  % Cada trama aporta numSC*k bits
        
        % Acumular para accuracy
        numCorrect_NN = numCorrect_NN + sum(YPred == Ytrue);
        numSymbols_NN = numSymbols_NN + numel(Ytrue);
    end
    
    % Calcular BER (a nivel de símbolo, no bit a bit) y accuracy
    BER_raw = numError_raw / numBits;
    BER_NN  = numError_NN / numBits;
    berRaw_vals(i) = BER_raw;
    berNN_vals(i) = BER_NN;
    
    accuracy_NN = numCorrect_NN / numSymbols_NN;
    
    fprintf('SNR = %d dB  --->  BER (Raw) = %e,  BER (NN) = %e, Accuracy (NN) = %.2f%%\n', ...
        currentSNR, BER_raw, BER_NN, accuracy_NN*100);
    
    %% 3.7 Generar y guardar la matriz de confusión para este SNR (usando la red)
    numFramesTest = 1000;
    XTestConf = cell(numFramesTest,1);
    YTestConf = cell(numFramesTest,1);
    for j = 1:numFramesTest
        signalTx = generateRandomData(M, numSC);
        [Labels, DPSKSignalEstimate] = processChannelAndTransmit(signalTx, M, FFTSize, currentSNR, numSC);
        featuresSeq_conf = [ real(DPSKSignalEstimate(:)).' ; imag(DPSKSignalEstimate(:)).' ];
        labelsSeq_conf = categorical(Labels(:).');
        XTestConf{j} = featuresSeq_conf;
        YTestConf{j} = labelsSeq_conf;
    end
    YPredConfCell = classify(net, XTestConf);
    % Concatenar resultados para la matriz de confusión
    YConf_all = [];
    YPred_all = [];
    for j = 1:numFramesTest
        YConf_all = [YConf_all, YTestConf{j}];
        YPred_all = [YPred_all, YPredConfCell{j}];
    end
    
    figCM = figure('visible','off');
    confusionchart(YConf_all, YPred_all, 'Title', sprintf('Matriz de Confusión - NN a %d dB', currentSNR), ...
        'RowSummary','row-normalized', 'ColumnSummary','column-normalized');
    saveas(figCM, fullfile(resultsFolder, sprintf('NN_ConfusionMatrix_SNR%d.png', currentSNR)));
    
end

%% 4. Graficar la relación SNR vs BER y guardar la figura
figBER = figure('visible','off');
plot(snrValues, berRaw_vals, '-o', 'LineWidth', 2);
hold on;
plot(snrValues, berNN_vals, '-s', 'LineWidth', 2);
grid on;
xlabel('SNR (dB)');
ylabel('BER');
title('Relación SNR vs BER');
legend('DPSK Directo (Raw)', 'Con Red Neuronal', 'Location', 'southwest');
saveas(figBER, fullfile(resultsFolder, 'SNR_vs_BER.png'));

%% Opcional: Guardar los datos de BER en un archivo CSV
csvwrite(fullfile(resultsFolder, 'SNR_vs_BER.csv'), [snrValues' berRaw_vals' berNN_vals']);
