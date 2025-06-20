clear all
close all
clc
addpath('Helper');
%% Parámetros del sistema
EbNo_dB = 0:2:16;           % Rango de Eb/No (energía por bit a ruido en dB)
modulationOrder = 4;       % Orden de la modulación DPSK (QPSK = 4)
fftSize = 48;              % Tamaño de FFT en OFDM (número de subportadoras)
bitsPerSymbol = log2(modulationOrder); % Bits transmitidos por símbolo DPSK
numSubcarriers = fftSize;  % Número de subportadoras
bitsPerOFDMSymbol = numSubcarriers * bitsPerSymbol;
numFramesTrain = 3000;   % Número de tramas para generar datos de entrenamiento

useOnlyAWGNChannel = false; % Indica uso de canal AWGN (solo ruido)

% Conversión Eb/No a SNR
if useOnlyAWGNChannel
    SNR_dB = EbNo_dB + 10*log10(bitsPerSymbol);
else
    SNR_dB = EbNo_dB + 10*log10(bitsPerSymbol) + 10*log10(117/128);
end

if useOnlyAWGNChannel
    resultsFolder = 'LSTMResultsAwgn';
else
    resultsFolder = 'LSTMResultsV2VChannel';
end

if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

bitErrorRate = zeros(1, length(EbNo_dB)); % Inicialización de vector BER

%% Carga de datos del canal V2V (LOS y NLOS)
load('../../Data/kaggle_dataset/v2v80211p_LOS.mat', 'vectReal32b')
channelLOS = vectReal32b;

load('../../Data/kaggle_dataset/v2v80211p_NLOS.mat', 'vectReal32b')
channelNLOS = vectReal32b;

%% Simulación principal sobre valores de SNR
for snrIdx = 1:length(SNR_dB)
    totalBitsTransmitted = 0;  
    channelIndex = 1;           
    
    numSeq      = numFramesTrain * fftSize;      % 5000 × 48
    XTrainCell  = cell(numSeq,1);
    YTrainCell  = cell(numSeq,1);
    seqIdx = 1;

    for j = 1:numFramesTrain
        
        % 1. Genera un frame completo de símbolos OFDM solo una vez por cada iteración del while
        [~,symbolFrame, transmittedFrame] = generateTransmittedFrame(fftSize, bitsPerOFDMSymbol, bitsPerSymbol, modulationOrder);

        % 2. Transmisión símbolo por símbolo a través del canal y recepción
        receivedFrame = transmitThroughChannel(transmittedFrame, fftSize, SNR_dB(snrIdx), channelLOS, channelNLOS, channelIndex);

        for sc = 1:fftSize
            XTrainCell{seqIdx} = [ real(receivedFrame(sc,:)) ; imag(receivedFrame(sc,:)) ];
            YTrainCell{seqIdx} = categorical(symbolFrame(sc,:), 0:modulationOrder-1);
            seqIdx = seqIdx + 1;
        end
    end

    currentSNR = SNR_dB(snrIdx);
    
    %% 3.0 Dividir en entrenamiento y prueba (secuencias)
    % Mezcla las secuencias completas (numSeq = numFramesTrain*fftSize)
    indices   = randperm(numSeq);

    pctTrain  = 0.7;                       % 90 % para entrenamiento
    numTrain  = round(pctTrain*numSeq);

    Xtrain_seq = XTrainCell(indices(1:numTrain));
    Ytrain_seq = YTrainCell(indices(1:numTrain));

    Xval_seq   = XTrainCell(indices(numTrain+1:end));
    Yval_seq   = YTrainCell(indices(numTrain+1:end));


    %% 3.1 Definir la arquitectura de la red neuronal LSTM para secuencias
    layers = [
        sequenceInputLayer(2, 'Normalization','zscore','Name','input')
        lstmLayer(288, 'OutputMode','sequence','Name','lstm')

        fullyConnectedLayer(288, 'Name','fc1')
        reluLayer('Name','relu1')
        dropoutLayer(0.4, 'Name','dropout1')   % Dropout after first ReLU

        fullyConnectedLayer(96, 'Name','fc2')
        reluLayer('Name','relu2')
        dropoutLayer(0.4, 'Name','dropout2')   % Dropout after second ReLU

        fullyConnectedLayer(4, 'Name','fc3')
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classOutput')
    ];
    %% 3.2 Opciones de entrenamiento
    miniBatch = 128;
    valFreq   = floor(numTrain/miniBatch);   % ≈ una validación cada época
    ckptDir   = fullfile(resultsFolder,'checkpoints');   % crea dir si no existe
    if ~exist(ckptDir,'dir'), mkdir(ckptDir); end

    options = trainingOptions('adam', ...
        'MaxEpochs', 50, ...
        'MiniBatchSize', miniBatch, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', {Xval_seq, Yval_seq}, ...
        'ValidationFrequency', valFreq, ...
        'ValidationPatience', 5, ...             % early-stopping si no mejora
        'GradientThreshold', 1, ...              % evita explosión de gradiente
        'Verbose', true, ...
        'VerboseFrequency', 750, ...
        'Plots', 'none', ...
        'CheckpointPath', ckptDir);              % guarda red cada mejora

    %% 3.3 Entrenar la red neuronal con los datos de entrenamiento (secuencias)
    fprintf('Entrenando la red neuronal para SNR = %.1f dB…\n', currentSNR);
    net = trainNetwork(Xtrain_seq, Ytrain_seq, layers, options);

    % Guardar la red entrenada en un archivo .mat
    networkFileName = fullfile(resultsFolder, sprintf('LSTM_Network_SNR_%ddB.mat', currentSNR));
    save(networkFileName, 'net');
    fprintf('Red neuronal guardada en: %s\n', networkFileName);

    %% ───────────────── 4.  Evaluación con datos NUEVOS y aleatorios ─────────────────
    targetErr   = 1e3;              % para terminar cuando se acumule este nº de errores
    targetBits  = 1e7;              % o cuando se alcance este nº de bits procesados
    bitsPerSym  = bitsPerSymbol;    % =2 para π/4-DQPSK
    numErrors   = 0;
    totalBits   = 0;
    channelIndexEval = 1;           % no mezclar con el de entrenamiento

    while (numErrors < targetErr) && (totalBits < targetBits)

        % 4.1  Generar nueva trama aleatoria (bits – símbolos – señal TX)
        [bitsFrame, symbolFrame, txFrame] = ...
            generateTransmittedFrame(fftSize, bitsPerOFDMSymbol, bitsPerSym, modulationOrder);

        % 4.2  Pasar la trama por el canal
        rxFrame = transmitThroughChannel(txFrame, fftSize, ...
                    SNR_dB(snrIdx), channelLOS, channelNLOS, channelIndexEval);

        % 4.3  Clasificar cada subportadora con la red
        for sc = 1:fftSize

            % ► Entradas 2×48  (Real/Imag)
            seqIn  = [ real(rxFrame(sc,:)) ; imag(rxFrame(sc,:)) ];

            % ► Predicción   (classify devuelve 1×1 cell)
            symPredCat  = classify(net, {seqIn});
            symPred     = double(symPredCat{1}) - 1;    % 1×48   (0…3)
            bitsPred    = sym2bitsRow(symPred, bitsPerSym);

            % ► Ground-truth
            bitsTrue    = bitsFrame(sc,:);              % 1×96   (ya fila)

            % ► Contar errores
            numErrors   = numErrors + sum(bitsPred ~= bitsTrue);
            totalBits   = totalBits  + numel(bitsTrue); % +96

            % ► Avanzar snapshot del canal (igual que en tu decodeAndCountErrors)
            channelIndexEval = mod(channelIndexEval, 9999) + 1;
        end
    end

    berTest = numErrors / totalBits;
    fprintf('\nBER con datos aleatorios (SNR = %.1f dB): %.3e  (errores=%u, bits=%u)\n', ...
            SNR_dB(snrIdx), berTest, numErrors, totalBits);

    bitErrorRate(snrIdx) = berTest;

end

save('BERNN.mat','bitErrorRate');
%% Gráfica de resultados
plotBERResults(EbNo_dB,bitErrorRate,modulationOrder);

function bitsRow = sym2bitsRow(symRow, bitsPerSymbol)
% symRow : 1×48  enteros 0…M-1
% bitsRow: 1×96  (orden MSB→LSB; mismo que bit2int/int2bit)

    bitsMat = int2bit(symRow', bitsPerSymbol);  % 48×2
    bitsRow = reshape(bitsMat.', 1, []);        % 1×96
end
