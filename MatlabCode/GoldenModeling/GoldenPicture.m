clear; clc;
% Carga canales
loadmatlab;
addpath('.');

%% ========= Imagen → bits → OFDM(QPSK) → Canal(H)+Ruido → Reconstrucción =========
% Parámetros de constelación/OFDM
modorder   = 4;                        % QPSK (4-QAM)
bitsPerSym = log2(modorder);

% Nº de tonos (subportadoras usadas como payload)
numTones = size(LOS,1);                % asume H de tamaño [numTones x numTones]
payloadBitsPerFrame = numTones * bitsPerSym;

% Imagen a transmitir (forzar 720x960x3)
imgPath = '/home/tonix/Documents/PhdDegreeCode/Data/Picture/Cascade.jpeg';
Itx = imread(imgPath);
Itx = imresize(Itx,[720 960]);         % por si acaso
if size(Itx,3) ~= 3, Itx = repmat(Itx,1,1,3); end
Itx_u8 = uint8(Itx);

% Imagen → stream de bits (Left-MSB por byte)
imgBytes = Itx_u8(:);                                      % vector columna uint8
imgBits  = reshape(de2bi(imgBytes,8,'left-msb').',[],1);   % [Nbits x 1]
Nbits    = numel(imgBits);

% Padding para múltiplo de payload
padBits  = mod(-Nbits, payloadBitsPerFrame);
if padBits > 0
    imgBitsPadded = [imgBits; zeros(padBits,1,'uint8')];
else
    imgBitsPadded = imgBits;
end
numFrames = numel(imgBitsPadded)/payloadBitsPerFrame;

% SNRs para visualizar (puedes endurecer: [20 10 5 0 -5])
SNRVECT_IMG = [20 10];

% Constellation para NearML
conste = qammod(0:modorder-1, modorder, 'gray', 'UnitAveragePower', true);
index  = 1:size(LOS, 1);

% Estado de canal
NLOS_cnt = 1;

% Prealocar receptores (uno por método)
rxBits_ZF_all     = zeros(numel(imgBitsPadded),1,'uint8');
rxBits_LMMSE_all  = zeros(numel(imgBitsPadded),1,'uint8');
rxBits_NML_all    = zeros(numel(imgBitsPadded),1,'uint8');
rxBits_Diag_all   = zeros(numel(imgBitsPadded),1,'uint8');
rxBits_NoEq_all   = zeros(numel(imgBitsPadded),1,'uint8');   % NUEVO: sin equalizar

% Helper conversión bits→imagen
toImg = @(b, sz) uint8(reshape( bi2de(reshape(b,8,[]).','left-msb'), sz ));

for s = 1:numel(SNRVECT_IMG)
    SNR = SNRVECT_IMG(s);

    % Buffers de salida por SNR
    rxBits_ZF     = rxBits_ZF_all;     % se sobreescriben
    rxBits_LMMSE  = rxBits_LMMSE_all;
    rxBits_NML    = rxBits_NML_all;
    rxBits_Diag   = rxBits_Diag_all;
    rxBits_NoEq   = rxBits_NoEq_all;

    bitPtr = 1;
    step = max(1, floor(numFrames/100));   % ~1% de los frames

    for f = 1:numFrames
        % progreso ~1%
        if mod(f, step) == 0 || f == numFrames
            pct = floor(100 * f / numFrames);
            fprintf('%3d%% (%d/%d) | SNR=%ddB\n', pct, f, numFrames, SNR);
        end

        % Canal: rota NLOS(:,:,k) con wrap correcto
        H = NLOS(:,:,99);
        NLOS_cnt = NLOS_cnt + 1;
        if NLOS_cnt > size(NLOS,3), NLOS_cnt = 1; end

        % Payload de bits para este frame
        bits_f = imgBitsPadded(bitPtr:bitPtr+payloadBitsPerFrame-1);

        % Modulación QPSK (potencia unitaria)
        X = qammod(bits_f, modorder, 'gray', 'InputType','bit', 'UnitAveragePower', true);

        % Paso por canal
        Y = H * X;

        % Ruido AWGN
        Ps = sum(abs(Y).^2) / numel(Y);
        Pn = Ps / (10^(SNR/10));
        n  = sqrt(Pn/2) * complex(randn(size(Y)), randn(size(Y)));
        y_n = Y + n;

        % --- Detectores ---
        % 1) SIN EQUALIZAR: demodular directo lo recibido
        rb_NoEq = qamdemod(y_n, modorder,'gray','OutputType','bit','UnitAveragePower',true);

        % 2) ZF/LS
        Xhat_ZF  = (H' * H) \ (H' * y_n);
        rb_ZF    = qamdemod(Xhat_ZF, modorder,'gray','OutputType','bit','UnitAveragePower',true);

        % 3) LMMSE
        Xhat_LMMSE = (H' * H + eye(numTones)*Pn) \ (H' * y_n);
        rb_LMMSE   = qamdemod(Xhat_LMMSE, modorder,'gray','OutputType','bit','UnitAveragePower',true);

        % 4) Near-ML
        R = H;
        [Xhat_NML, ~] = Near_ML(y_n, R, conste, index);
        rb_NML    = qamdemod(Xhat_NML, modorder,'gray','OutputType','bit','UnitAveragePower',true);

        % 5) One-tap por diagonal (si H es (casi) diagonal)
        Hdiag = diag(H);
        Hdiag_safe = Hdiag; Hdiag_safe(abs(Hdiag_safe) < 1e-12) = 1e-12;
        Xhat_Diag = y_n ./ Hdiag_safe;
        rb_Diag  = qamdemod(Xhat_Diag, modorder,'gray','OutputType','bit','UnitAveragePower',true);

        % Guardar streams
        rxBits_NoEq(bitPtr:bitPtr+payloadBitsPerFrame-1)  = rb_NoEq;
        rxBits_ZF(bitPtr:bitPtr+payloadBitsPerFrame-1)    = rb_ZF;
        rxBits_LMMSE(bitPtr:bitPtr+payloadBitsPerFrame-1) = rb_LMMSE;
        rxBits_NML(bitPtr:bitPtr+payloadBitsPerFrame-1)   = rb_NML;
        rxBits_Diag(bitPtr:bitPtr+payloadBitsPerFrame-1)  = rb_Diag;

        bitPtr = bitPtr + payloadBitsPerFrame;
    end

    % Recortar padding y reconstruir imágenes
    rxBits_NoEq = rxBits_NoEq(1:Nbits);
    rxBits_ZF    = rxBits_ZF(1:Nbits);
    rxBits_LMMSE = rxBits_LMMSE(1:Nbits);
    rxBits_NML   = rxBits_NML(1:Nbits);
    rxBits_Diag  = rxBits_Diag(1:Nbits);

    I_NoEq  = toImg(rxBits_NoEq, size(Itx_u8));
    I_ZF    = toImg(rxBits_ZF,    size(Itx_u8));
    I_LMMSE = toImg(rxBits_LMMSE, size(Itx_u8));
    I_NML   = toImg(rxBits_NML,   size(Itx_u8));
    I_Diag  = toImg(rxBits_Diag,  size(Itx_u8));

    % PSNR robusto (evita NaN/Inf)
    psnrNoEq  = psnr255(I_NoEq,  Itx_u8);
    psnrZF    = psnr255(I_ZF,    Itx_u8);
    psnrLMMSE = psnr255(I_LMMSE, Itx_u8);
    psnrNML   = psnr255(I_NML,   Itx_u8);
    psnrDiag  = psnr255(I_Diag,  Itx_u8);

    % BER rápido para sanity-check
    berNoEq  = mean(xor(imgBits, rxBits_NoEq));
    berZF    = mean(xor(imgBits, rxBits_ZF));
    berLMMSE = mean(xor(imgBits, rxBits_LMMSE));
    berNML   = mean(xor(imgBits, rxBits_NML));
    berDiag  = mean(xor(imgBits, rxBits_Diag));

    fprintf('SNR=%2d dB | PSNR(dB): NoEq %.1f | ZF %.1f | LMMSE %.1f | NML %.1f | Diag %.1f\n', ...
        SNR, psnrNoEq, psnrZF, psnrLMMSE, psnrNML, psnrDiag);
    fprintf('SNR=%2d dB | BER:      NoEq %.4f | ZF %.4f | LMMSE %.4f | NML %.4f | Diag %.4f\n', ...
        SNR, berNoEq, berZF, berLMMSE, berNML, berDiag);

    % Guardar resultados
    outDirImg = 'recon_images';
    if ~exist(outDirImg,'dir'), mkdir(outDirImg); end
    imwrite(I_NoEq,  fullfile(outDirImg, sprintf('RX_NoEq_SNR%ddB.png',SNR)));
    imwrite(I_ZF,    fullfile(outDirImg, sprintf('RX_ZF_SNR%ddB.png',SNR)));
    imwrite(I_LMMSE, fullfile(outDirImg, sprintf('RX_LMMSE_SNR%ddB.png',SNR)));
    imwrite(I_NML,   fullfile(outDirImg, sprintf('RX_NearML_SNR%ddB.png',SNR)));
    imwrite(I_Diag,  fullfile(outDirImg, sprintf('RX_Diag_SNR%ddB.png',SNR)));
end

%% ===== Función local: PSNR robusto para uint8/double/single =====
function p = psnr255(A,B)
    A = double(A); B = double(B);
    if ~isequal(size(A), size(B)), error('PSNR: tamaños distintos'); end
    d = A - B;
    if any(isnan(d(:))) || any(isinf(d(:)))
        p = NaN; return;
    end
    mse = mean(d(:).^2);
    if mse == 0
        p = Inf;  % idénticas
    else
        p = 10*log10(255^2 / mse);
    end
end
