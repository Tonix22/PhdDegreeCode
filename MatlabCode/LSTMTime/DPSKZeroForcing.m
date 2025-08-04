%% -------------------------------------------------------------------
%   Barrido Eb/No 0:2:16  –  DPSK‑OFDM con ecualizador Zero‑Forcing    
%  * Para cada Eb/No calcula BER antes y después de ZF                
%  * Guarda los vectores en BERZeroForcemat.mat                       
% -------------------------------------------------------------------
clear; close all; clc;
addpath('Helper');                              % por si se necesitan funciones externas

%% -------------------- Parámetros generales --------------------------
EbNo_dB_vec   = 0:2:16;         % [dB]
M             = 4;              % DPSK‑QPSK
N             = 48;             % subportadoras (= símbolos por trama)
bitsPerSymbol = log2(M);

nFramesPerEbNo = 40;            % nº de tramas para promediar (ajusta si deseas más precisión)

%% ------------- Carga del canal V2V (una sola vez) -------------------
load('../../Data/kaggle_dataset/v2v80211p_LOS.mat',  'vectReal32b'); H1 = vectReal32b;
load('../../Data/kaggle_dataset/v2v80211p_NLOS.mat', 'vectReal32b'); H2 = vectReal32b;

%% -------------------- Pre‑alocación resultados ----------------------
bitErrorRate_rx = zeros(size(EbNo_dB_vec));   % BER antes del EQ
bitErrorRate_eq = zeros(size(EbNo_dB_vec));   % BER después del ZF

%% ======================== Bucle principal ===========================
for idxEb = 1:length(EbNo_dB_vec)
    EbNo_dB = EbNo_dB_vec(idxEb);
    SNR_dB  = EbNo_dB + 10*log10(bitsPerSymbol);  % sólo AWGN

    err_rx = 0;  err_eq = 0;  totBits = 0;
    channelIdx = 1;                               % reiniciamos índice canal

    for frame = 1:nFramesPerEbNo
        %% --- Generación de trama TX ---------------------------------
        bits_tx = randi([0 1], N, N*bitsPerSymbol);  % 48×(48*2)
        X = zeros(N,N);
        for k = 1:N
            ints  = bit2int(bits_tx(k,:).', bitsPerSymbol);
            X(k,:) = dpskmod(ints, M, pi/4);
        end

        %% --- Paso por canal símbolo a símbolo -----------------------
        Y  = zeros(N,N);         % recibido sin EQ
        HdiagMat = zeros(N,N);   % diagonales del canal por símbolo

        for nSym = 1:N
            H = H1(:,:,channelIdx) + H2(:,:,channelIdx);  % 48×48
            hdiag = diag(H).';
            HdiagMat(nSym,:) = hdiag;

            tx_time = ifft(X(:,nSym), N);
            sigPwr_dB = 10*log10(var(tx_time));
            noiseVar  = 10^(0.1*(sigPwr_dB - SNR_dB));
            w         = sqrt(noiseVar/2)*(randn(size(tx_time))+1j*randn(size(tx_time)));
            rx_time   = tx_time + w;
            rx_freq   = fft(rx_time,N);
            Y(:,nSym) = H * rx_freq;

            channelIdx = channelIdx + 1;
        end

        %% --- Equalización ZF (diagonal) -----------------------------
        Z = zeros(N,N);
        for nSym = 1:N
            Z(:,nSym) = Y(:,nSym) ./ (HdiagMat(nSym,:).' + eps);
        end

        %% --- Demod & acumulación de errores -------------------------
        for k = 1:N
            % Sin EQ
            sym_rx = dpskdemod(Y(k,:), M, pi/4);
            bits_rx = int2bit(sym_rx.', bitsPerSymbol);
            % Con EQ
            sym_eq = dpskdemod(Z(k,:), M, pi/4);
            bits_eq = int2bit(sym_eq.', bitsPerSymbol);
            % Original
            bits_orig = bits_tx(k,:).';

            err_rx = err_rx + biterr(bits_orig, bits_rx);
            err_eq = err_eq + biterr(bits_orig, bits_eq);
            totBits = totBits + length(bits_orig);
        end
    end

    bitErrorRate_rx(idxEb) = err_rx / totBits;
    bitErrorRate_eq(idxEb) = err_eq / totBits;

    fprintf('Eb/No=%2d dB  ->  BER_RX=%.3e  |  BER_ZF=%.3e\n', EbNo_dB, bitErrorRate_rx(idxEb), bitErrorRate_eq(idxEb));
end

%% ------------------------- Guardado ---------------------------------
save('BERZeroForcemat.mat','bitErrorRate_eq');

%% ------------------------- Gráfica ----------------------------------
figure; semilogy(EbNo_dB_vec, bitErrorRate_rx,'r*-', EbNo_dB_vec, bitErrorRate_eq,'bo-','LineWidth',1.5);
grid on; xlabel('Eb/No (dB)'); ylabel('BER'); legend('Sin EQ','ZF diagonal','Location','southwest'); title('DPSK‑OFDM BER vs Eb/No');
