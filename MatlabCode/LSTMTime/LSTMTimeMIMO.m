% =========================================================================
%  DPSK‑OFDM 48×48  ·  CNN‑BiLSTM con Retransmisiones (T = 3)
%  Entrena una red profunda para equalizar DPSK‑OFDM sobre canal V2V/‑AWGN
%  y calcula la BER con bits completamente nuevos.
%
%  · Cada muestra de entrenamiento = tensor 2×3:
%       – Fila 1  → parte real  de 3 retransmisiones
%       – Fila 2  → parte imaginaria de 3 retransmisiones
%  · Etiqueta     = secuencia 3×1 (el mismo símbolo repetido)
%  · Front‑end    = 2 × (Conv1d → ReLU → BN) → FC(2) → BiLSTM → FC
% =========================================================================
clear; close all; clc;
addpath('Helper');

%% ───────── 0. Parámetros generales ──────────────────────────────────────
EbNo_dB          = 0:2:16;        % Energía/bit a ruido, dB
Retransmissions  = 3;             % ← NUEVO · pasos temporales T
modulationOrder  = 4;             % π/4‑DQPSK  (M = 4)
fftSize          = 48;            % n.º subportadoras
bitsPerSymbol    = log2(modulationOrder);
numSubcarriers   = fftSize;
bitsPerOFDMSym   = numSubcarriers * bitsPerSymbol;
numFramesTrain   = 1000;          % para generación de datos
useOnlyAWGNChan  = false;         % true = solo AWGN, false = V2V+AWGN

% Front‑end CNN
conv1Channels    = 32;
conv2Channels    = 64;
fcHidden         = 64;

%% ───────── 1. SNR ← Eb/No ───────────────────────────────────────────────
if useOnlyAWGNChan
    SNR_dB = EbNo_dB + 10*log10(bitsPerSymbol);
    resultsFolder = 'CNN_BiLSTM_AWGN';
else
    SNR_dB = EbNo_dB + 10*log10(bitsPerSymbol) + 10*log10(117/128);
    resultsFolder = 'CNN_BiLSTM_V2V';
end
if ~exist(resultsFolder,'dir'); mkdir(resultsFolder); end

%% ───────── 2. Cargar canal V2V (si aplica) ──────────────────────────────
if ~useOnlyAWGNChan
    load('../../Data/kaggle_dataset/v2v80211p_LOS.mat','vectReal32b');
    channelLOS  = vectReal32b;
    load('../../Data/kaggle_dataset/v2v80211p_NLOS.mat','vectReal32b');
    channelNLOS = vectReal32b;
else
    channelLOS  = []; channelNLOS = [];
end

bitErrorRate = zeros(1,numel(EbNo_dB));   % BER por Eb/No

%% ───────── 3. Bucle por cada SNR  ───────────────────────────────────────
for snrIdx = 1:numel(SNR_dB)

    % ── 3.1 Generar dataset de entrenamiento (cell arrays) ──────────────
    numSeq     = numFramesTrain * fftSize;     % 5000 × 48 * 3
    XTrainCell = cell(numSeq,1); % (feature = 48,channel = 2,samples = XTrainCell)
    XTrainGT   = cell(numSeq,1); % (feature = 48,channel = 2,samples = XTrainCell)
    YTrainCell = cell(numSeq,1); % (feature = 48,samples = XTrainCell)
    seqIdx     = 1;
    chanPtr    = 1;                            % índice de snapshot

    for frm = 1:numFramesTrain

        % (a) Un frame completo (48 símbolos)
        [~, symbolFrame, txFrame] = generateTransmittedFrame(fftSize, bitsPerOFDMSym, bitsPerSymbol, modulationOrder);
        % (b) Obtener pila de T = 3 retransmisiones
        rxStack = zeros(fftSize,fftSize, Retransmissions);   % COMPLEX (48×3)

        for r = 1:Retransmissions
            rxStack(:,:,r) = transmitThroughChannel(txFrame, fftSize, SNR_dB(snrIdx), channelLOS, channelNLOS, chanPtr);           
        end
        chanPtr = mod(chanPtr,9999)+1; 

        % (c) Construir muestras por subportadora
        for sc = 1:fftSize
            %% Conv2d Input
            % Suponiendo rxStack tiene tamaño [48, 48, 3]
            realPart = real(rxStack(sc,:,:)); % 1x48x3
            imagPart = imag(rxStack(sc,:,:)); % 1x48x3

            % Quitar la dimensión singleton para obtener 48x3
            realPart = squeeze(realPart); % 48x3
            imagPart = squeeze(imagPart); % 48x3

            % Calcula la fase de la matriz compleja (por ejemplo, originalFrame)
            %phasePart = angle(realPart + 1j*imagPart); % 48x3
            
            % Concatenar como canales en la tercera dimensión: [48, 3, 2]
            seqIn = cat(3, realPart, imagPart);
            XTrainCell{seqIdx} = seqIn;
            %% Conv2d Output
            realPart = real(txFrame(sc,:)); % 1x48
            imagPart = imag(txFrame(sc,:)); % 1x48
            % Convertir a columna (48x1)
            realPart = realPart(:); % 48x1
            imagPart = imagPart(:); % 48x1
            XTrainGT{seqIdx} = cat(3, realPart, imagPart); % [48, 1, 2]
            %% For Clasificiation
            YTrainCell{seqIdx} = categorical(symbolFrame(sc,:), 0:modulationOrder-1); % For classification
            seqIdx = seqIdx + 1;
        end
    end

    currentSNR = SNR_dB(snrIdx);
    % ── 3.2 Train / Validation split ───────────────────────────────────
    pctTrain = 0.7;
    idx      = randperm(numSeq);

    nTrain        = floor(pctTrain * numSeq);       % sincronía garantizada
    trainIdx      = idx(1:nTrain);
    valIdx        = idx(nTrain+1:end);

    % ---------- 1) Predictores ----------
    XtrainCell    = XTrainCell(trainIdx);           % celda – se convertirá a 4‑D
    XvalCell      = XTrainCell(valIdx);

    % ---------- 2) GT para regresión ----------
    GTtrainCell   = XTrainGT(trainIdx);             % celda – se convertirá a 4‑D
    GTvalCell     = XTrainGT(valIdx);

    % ---------- 3) Etiquetas para clasificación (se conservan) ----------
    Ytrain        = YTrainCell(trainIdx);           % SIN tocar
    Yval          = YTrainCell(valIdx);

    % ---------- 4) Conversión a 4‑D (sólo lo necesario) ----------
    Xtrain4D      = cat(4, XtrainCell{:});          % [48 3 2 Ntrain]
    GTtrain4D     = cat(4, GTtrainCell{:});         % [48 1 2 Ntrain]

    Xval4D        = cat(4, XvalCell{:});            % [48 3 2 Nval]
    GTval4D       = cat(4, GTvalCell{:});           % [48 1 2 Nval]

    assert(size(Xtrain4D,4) == size(GTtrain4D,4), 'Train mismatch');
    assert(size(Xval4D,4)   == size(GTval4D,4)  , 'Val mismatch');

   

    % ── 3.3 Arquitectura CNN+BiLSTM ─────────────────────────────────────
    % CNN
    layers = [
        % Entrada: 48 filas × 3 columnas × 2 canales (Re/Im)
        imageInputLayer([fftSize Retransmissions 2], 'Normalization','rescale-symmetric', 'Name','input')   % [48×3×2]

        % Conv 1×2, 16 filtros – reduce ancho 3→2
        convolution2dLayer([1 2], 8, 'Name','conv1')                         % [48×2×16]
        tanhLayer('Name','sig1')

        % Conv 1×2, 8 filtros  – reduce ancho 2→1
        convolution2dLayer([1 2], 4,  'Name','conv2')                         % [48×1×8]
        tanhLayer('Name','tanh2')

        % Conv 1×1, 2 filtros – mantiene tamaño, mezcla canales
        convolution2dLayer([1 1], 2,  'Name','convOut')                       % [48×1×2]
    ];


    % ── 3.4 Opciones de entrenamiento ──────────────────────────────────
    miniBatch     = 128;
    valFreq       = floor(numel(Ytrain)/miniBatch);
    ckptDir       = fullfile(resultsFolder,'checkpoints');
    if ~exist(ckptDir,'dir')
        mkdir(ckptDir); 
    end

    options = trainingOptions('adam', ...
        'MaxEpochs',              20, ...
        'MiniBatchSize',          miniBatch, ...
        'Shuffle',               'every-epoch', ...
        'ValidationData',         {Xval4D,GTval4D}, ...
        'ValidationFrequency',    valFreq, ...
        'ValidationPatience',     5, ...
        'GradientThreshold',      1, ...
        'Verbose',                true, ...
        'VerboseFrequency',       500, ...
        'Plots',                  'none', ...
        'CheckpointPath',         ckptDir);

    % ── 3.5 Entrenamiento ──────────────────────────────────────────────
    fprintf('\n Entrenando CNN BiLSTM (SNR = %.1f dB)…\n',SNR_dB(snrIdx));
    netEq = trainnet(Xtrain4D, GTtrain4D, layers,"mse",options);
    save(fullfile(resultsFolder, sprintf('CNN_%ddB.mat',SNR_dB(snrIdx))), 'netEq');

    % === A) PREDICCIONES SOBRE CONJUNTO DE ENTRENAMIENTO ==========
    % Xtrain4D  : [48 3 2 Ntrain]   (ya lo tienes)
    eqTrain4D   = predict(netEq, Xtrain4D);   % → [48 1 2 Ntrain]

    % === B) PREDICCIONES SOBRE CONJUNTO DE VALIDACIÓN ============
    eqVal4D     = predict(netEq, Xval4D);     % → [48 1 2 Nval]

    % ------------ Entrenamiento ----------------
    eqTrain2x48 = permute(eqTrain4D, [3 1 2 4]);   % [2 48 1 Ntrain]
    eqTrain2x48 = squeeze(eqTrain2x48);            % [2 48 Ntrain]

    XeqTrain = squeeze(num2cell(eqTrain2x48,[1 2]));  % celda de Ntrain

    % ------------ Validación --------------------
    eqVal2x48  = permute(eqVal4D,  [3 1 2 4]);    % [2 48 1 Nval]
    eqVal2x48  = squeeze(eqVal2x48);

    XeqVal   = squeeze(num2cell(eqVal2x48,[1 2]));    % celda de Nval

    YclsTrain = Ytrain;   % ya en el mismo orden
    YclsVal   = Yval;

    layersCls = [
        sequenceInputLayer(2,'Normalization','rescale-zero-one','Name','input')

        bilstmLayer(192,'OutputMode','sequence','Name','lstm')

        fullyConnectedLayer(192,'Name','fc1')
        reluLayer('Name','relu1')
        dropoutLayer(0.2,'Name','dropout1')

        fullyConnectedLayer(96,'Name','fc2')
        reluLayer('Name','relu2')
        dropoutLayer(0.2,'Name','dropout2')

        fullyConnectedLayer(4,'Name','fc3')     % 4 clases
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classOutput')
    ];

    optsCls = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', miniBatch, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XeqVal, YclsVal}, ...
    'ValidationFrequency', valFreq, ...
    'ValidationPatience', 5, ...             % early-stopping si no mejora
    'GradientThreshold', 1, ...              % evita explosión de gradiente
    'Verbose', true, ...
    'VerboseFrequency', 750, ...
    'Plots', 'none', ...
    'CheckpointPath', ckptDir);

    netCls = trainNetwork(XeqTrain, YclsTrain, layersCls, optsCls);
    % Guardar la red entrenada en un archivo .mat
    networkFileName = fullfile(resultsFolder, sprintf('CNN_LSTM_Network_SNR_%ddB.mat', currentSNR));
    save(networkFileName, 'netCls');
    fprintf('Red neuronal guardada en: %s\n', networkFileName);

    %% --- 4. Evaluación BER ------------------------------------------------
    targetErr  = 1e3;
    targetBits = 1e7;
    numErr     = 0;
    totBits    = 0;
    chanPtr    = 1;

    while (numErr < targetErr) && (totBits < targetBits)

        % 4.1  Frame nuevo (48 símbolos × 48 subportadoras)
        [bitFrame, symFrame, txFrame] = generateTransmittedFrame(fftSize, bitsPerOFDMSym,...
                                    bitsPerSymbol, modulationOrder);

        % 4.2  Pila de T retransmisiones COMPLETA
        rxStack = zeros(fftSize, fftSize, Retransmissions);   % 48×48×T
        for r = 1:Retransmissions
            rxStack(:,:,r) = transmitThroughChannel(txFrame, fftSize, ...
                    SNR_dB(snrIdx), channelLOS, channelNLOS, chanPtr);
        end
        chanPtr = mod(chanPtr,9999)+1;

        % 4.3  Recorre las 48 subportadoras (una “muestra” para la CNN)
        for sc = 1:fftSize

            % --------- 4.3.1  PREPARAR entrada CNN  -----------------------
            % realPart / imagPart  → 48×T
            realPart = squeeze(real(rxStack(sc,:,:)));   % 48×T
            imagPart = squeeze(imag(rxStack(sc,:,:)));   % 48×T
            cnnInput = cat(3, realPart, imagPart);       % [48  T  2]

            % --------- 4.3.2  INFERENCIA CNN (= equalizador) -------------
            eqOut = predict(netEq, cnnInput);            % [48 1 2]

            % --------- 4.3.3  Preparar entrada LSTM  ---------------------
            %   De [48 1 2]  →  [2 48]   (features × time‐steps)
            eqSeq = squeeze(eqOut)';                     % 2×48

            % --------- 4.3.4  INFERENCIA LSTM (= clasificador) ----------
            predCat = classify(netCls, {eqSeq});         % 1×48 cell
            predSym = double(predCat{1}) - 1;            % vector 1×48

            % --------- 4.3.5  Bits verdaderos y bits predichos ----------
            bitsPred = sym2bitsRow(predSym , bitsPerSymbol); % idem
            
            % ► Ground-truth
            bitsTrue    = bitFrame(sc,:);

            % --------- 4.3.6  Acumular métricas --------------------------
            numErr   = numErr + sum(bitsPred ~= bitsTrue);
            totBits   = totBits  + numel(bitsTrue); % +96
        end
    end

    ber = numErr / totBits;
fprintf('>> BER  %.1f dB : %.3e   (err=%u  bits=%u)\n', ...
        SNR_dB(snrIdx), ber, numErr, totBits);

end

save(fullfile(resultsFolder,'BER_CNN_BiLSTM.mat'),'bitErrorRate');
plotBERResults(EbNo_dB,bitErrorRate,modulationOrder);

%% ───────────────────────────────────────────────────────────────────────
function bitsRow = sym2bitsRow(symRow, bitsPerSymbol)
% symRow : 1×48  enteros 0…M-1
% bitsRow: 1×96  (orden MSB→LSB; mismo que bit2int/int2bit)

    bitsMat = int2bit(symRow', bitsPerSymbol);  % 48×2
    bitsRow = reshape(bitsMat.', 1, []);        % 1×96
end
