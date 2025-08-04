%% ====================================================================
%  DPSK‑OFDM 48×48 – DiagNet‑2D por Eb/No  (Opción‑A: estima Hdiag)  
%  * Entrada  = mapa 48×48×2  (Re/Im de Y)                            
%  * Objetivo = mapa 48×48×2  (Re/Im de H_diag)                       
%  * La ecualización Zero‑Forcing Y/Ĥ se hace explícitamente en test  
%  * Una red por Eb/No (0:2:16 dB).  Criterio while: targetErr/targetBits
% ====================================================================
clear; close all; clc; addpath('Helper');

%% ----------- 0. Parámetros generales -------------------------------
EbNo_dB_vec = 0:2:16;
M = 4;  N = 48;  bitsPerSym = log2(M);
framesPerEb = 800;   framesEval = 40;
resultsDir = 'DiagNet2D_H_est'; if ~exist(resultsDir,'dir'), mkdir(resultsDir); end

%% -------- 1. Cargar canal V2V --------------------------------------
load('../../Data/kaggle_dataset/v2v80211p_LOS.mat','vectReal32b'); H1 = vectReal32b;
load('../../Data/kaggle_dataset/v2v80211p_NLOS.mat','vectReal32b'); H2 = vectReal32b;

BER_rx = zeros(size(EbNo_dB_vec)); BER_eq = zeros(size(EbNo_dB_vec));

for idxEb = 1:numel(EbNo_dB_vec)
    EbNo_dB = EbNo_dB_vec(idxEb);
    SNR_dB  = EbNo_dB + 10*log10(bitsPerSym);
    fprintf('\n=== Eb/No %2d dB  (SNR %.1f dB) ===\n', EbNo_dB, SNR_dB);

    %% ---- 2. Dataset (48×48×2)  Target = Hdiag ---------------------
    Xcell = cell(framesPerEb,1);  Ycell = cell(framesPerEb,1); chIdx = 1;
    for f = 1:framesPerEb
        bits_tx = randi([0 1], N, N*bitsPerSym);
        Xsym = zeros(N,N);
        for k = 1:N, Xsym(k,:) = dpskmod(bit2int(bits_tx(k,:).',bitsPerSym),M,pi/4); end
        Y = zeros(N,N); Hdiag = zeros(N,N);
        for n = 1:N
            Hmat = H1(:,:,chIdx)+H2(:,:,chIdx);
            Hdiag(:,n) = diag(Hmat);
            txT = ifft(Xsym(:,n),N);
            nv  = 10^(0.1*(10*log10(var(txT))-SNR_dB));
            Y(:,n) = Hmat*fft(txT + sqrt(nv/2)*(randn(size(txT))+1j*randn(size(txT))),N);
            chIdx = chIdx + 1; if chIdx>10000, chIdx = 1; end
        end
        Xcell{f} = cat(3, real(Y), imag(Y));            % Predictor
        Ycell{f} = cat(3, real(Hdiag), imag(Hdiag));    % Ground‑truth H
    end
    fprintf('  Dataset listo (%d frames)\n', framesPerEb);

    %% ---- 3. Convertir a 4‑D y split --------------------------------
    Xall = cat(4,Xcell{:});  Yall = cat(4,Ycell{:});
    idx = randperm(framesPerEb); nTr = round(0.8*framesPerEb);
    XTrain = Xall(:,:,:,idx(1:nTr));   YTrain = Yall(:,:,:,idx(1:nTr));
    XVal   = Xall(:,:,:,idx(nTr+1:end)); YVal = Yall(:,:,:,idx(nTr+1:end));

    %% ---- 4. CNN‑2D para estimar Hdiag ------------------------------
    layers = [
        imageInputLayer([48 48 2],"Normalization","zscore","Name","in")
        convolution2dLayer(3,32,'Padding','same'); batchNormalizationLayer; tanhLayer
        convolution2dLayer(3,64,'Padding','same'); batchNormalizationLayer; tanhLayer
        convolution2dLayer(3,64,'Padding','same','DilationFactor',2); batchNormalizationLayer; tanhLayer
        convolution2dLayer(3,64,'Padding','same'); batchNormalizationLayer; tanhLayer
        convolution2dLayer(1,128,'Padding','same'); batchNormalizationLayer; tanhLayer; dropoutLayer(0.2)
        convolution2dLayer(1,64,'Padding','same');  batchNormalizationLayer; tanhLayer; dropoutLayer(0.2)
        convolution2dLayer(1,2,'Padding','same')
        regressionLayer];

    opts = trainingOptions('adam','MaxEpochs',30,'MiniBatchSize',32,'Shuffle','every-epoch', ...
        'ValidationData',{XVal,YVal},'ValidationFrequency',ceil(nTr/32),'Verbose',true);

    fprintf('  Entrenando DiagNetH_%ddB ...\n', EbNo_dB);
    Diag2D = trainNetwork(XTrain,YTrain,layers,opts);
    save(fullfile(resultsDir,sprintf('DiagNetH_%ddB.mat',EbNo_dB)),'Diag2D');

    %% ---- 5. Evaluación BER ----------------------------------------
    targetErr=1e3; targetBits=1e7; err_rx=0; err_eq=0; bitsTot=0; chIdxEval=1;
    while err_rx<targetErr && bitsTot<targetBits
        bits_tx = randi([0 1], N, N*bitsPerSym);
        Xsym=zeros(N,N); for k=1:N, Xsym(k,:)=dpskmod(bit2int(bits_tx(k,:).',bitsPerSym),M,pi/4); end
        Y=zeros(N,N);
        for n=1:N
            Hmat=H1(:,:,chIdxEval)+H2(:,:,chIdxEval);
            txT=ifft(Xsym(:,n),N);
            nv=10^(0.1*(10*log10(var(txT))-SNR_dB));
            Y(:,n)=Hmat*fft(txT + sqrt(nv/2)*(randn(size(txT))+1j*randn(size(txT))),N);
            chIdxEval=chIdxEval+1; if chIdxEval>10000, chIdxEval=1; end
        end
        % Predicción de Ĥ y ZF explícito
        Hhat_pred = predict(Diag2D, cat(3,real(Y),imag(Y)));
        Hhat = complex(Hhat_pred(:,:,1), Hhat_pred(:,:,2));
        Xhat = Y ./ (Hhat + eps);

        % Contar errores
        for k=1:N
            sym_rx=dpskdemod(Y(k,:),M,pi/4);
            sym_eq=dpskdemod(Xhat(k,:),M,pi/4);
            b_org=bits_tx(k,:).';
            err_rx = err_rx + biterr(b_org,int2bit(sym_rx.',bitsPerSym));
            err_eq = err_eq + biterr(b_org,int2bit(sym_eq.',bitsPerSym));
            bitsTot = bitsTot + numel(b_org);
        end
    end
    BER_rx(idxEb)=err_rx/bitsTot; BER_eq(idxEb)=err_eq/bitsTot;
    fprintf('  BER_RX=%.2e  BER_EQ=%.2e\n', BER_rx(idxEb), BER_eq(idxEb));
end

%% ---- 6. Guardar y graficar ----------------------------------------
save(fullfile(resultsDir,'BER_DiagNetH_perEb.mat'),'BER_rx','BER_eq','EbNo_dB_vec');
figure; semilogy(EbNo_dB_vec,BER_rx,'r*-',EbNo_dB_vec,BER_eq,'bo-','LineWidth',1.5);
xlabel('Eb/No (dB)'); ylabel('BER'); grid on;
legend('Sin EQ','DiagNet‑ZF (est. H)'); title('BER vs Eb/No (CNN‑2D estima H)');
