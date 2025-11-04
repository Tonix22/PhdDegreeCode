clear all;

%load .mat files where Channels are saved
loadmatlab;
addpath('.') % add current path to directory

FastRun = false;

% QAM DATA
modorder   = 4;  %constelation size

SNRVECT    = 45:-2:5; % SNR range with step of

% Vectors where BER is saved
BER_MSE    = zeros(numel(SNRVECT),1);
BER_LMSE   = zeros(numel(SNRVECT),1);
BER_NearML = zeros(numel(SNRVECT),1);
BER_Diag  = zeros(numel(SNRVECT),1);
% Testing data set range
frames     = 4000;

% Constellation points for Near ML
conste = qammod(0:modorder-1, modorder,'gray','UnitAveragePower', true);
% Index vector for Near ML
index  = 1:size(LOS, 1); 

for SNR=SNRVECT
    %Total errors for each model
    errors_MSE    = 0;
    errors_LMSE   = 0;
    errors_NearML = 0;
    errors_Diag  = 0;
    %Channel indices
    LOS_cnt       = 1;
    NLOS_cnt      = 1;
    %Iterations for progres bar
    nIterations   = 4000;
    %ProgressBar
    hWaitBar = waitbar(0, ['SNR: ', num2str(SNR)]);

    for i = 16001:1:20000 % 4000 iterations
        %Progress
        progress = (i-16001) / nIterations;
        ML_curr_err    = errors_NearML/(48*4*(i-16001));
        LMMSE_curr_err = errors_LMSE/(48*4*(i-16001));
        waitbar(progress, hWaitBar, sprintf('SNR: %d, NML BER: %.5f Total: %.1f%%', SNR, ML_curr_err , progress * 100));

        %Alternated betwen LOS and NLOS channel
        if(bitand(i,1))
             H = LOS(:,:,LOS_cnt);
             LOS_cnt = LOS_cnt+1;
        end
        if(bitand(i,0))
             H = NLOS(:,:,NLOS_cnt);
             NLOS_cnt = NLOS_cnt+1;
        end
        %Tx data generation
        txbits = randi([0 1],size(LOS,1)*log2(modorder),1);
        %X      = qammod(txbits, modorder, 'gray', 'InputType', 'bit','UnitAveragePower', true);
        X      = qammod(txbits, modorder, 'gray', 'InputType', 'bit','UnitAveragePower', true);
        %Channel matrix multiply
        Y   = H*X;

        % Signal Power
        Ps  = sum(abs(Y).^2) / length(Y);
        % Noise power
        Pn  = Ps / (10^(SNR/10));
        % Generate noise
        n   = sqrt(Pn/2)* complex(randn(size(Y)), randn(size(Y)));
        y_n = Y+n;

        %MMSE
        X_hat_MSE  = inv(H'*H)*H'*y_n; %MSE
        %LMMSE
        X_hat_LMSE = inv(H'*H+eye(48)*Pn)*H'*y_n; %LMSE

        % Near ML
        R = H;
        [X_hat_NearML, nodos] = Near_ML(y_n, R, conste, index); % Call Near ML function

        % One-tap (per-tone) ZF usando solo la diagonal de H
        Hdiag = diag(H);
        % Evitar división por ~0
        Hdiag_safe = Hdiag;
        Hdiag_safe(abs(Hdiag_safe) < 1e-12) = 1e-12;
        X_hat_Diag = y_n ./ Hdiag_safe;
        
        %qam demod
        rxbits_MSE    = qamdemod(X_hat_MSE, modorder, 'gray', 'OutputType', 'bit','UnitAveragePower', true);
        rxbits_LMSE   = qamdemod(X_hat_LMSE, modorder, 'gray', 'OutputType', 'bit','UnitAveragePower', true);
        rxbits_NearML = qamdemod(X_hat_NearML, modorder, 'gray', 'OutputType', 'bit','UnitAveragePower', true); % Demodulate Near ML
        rxbits_Diag = qamdemod(X_hat_Diag, modorder, 'gray', 'OutputType', 'bit','UnitAveragePower', true);

        %Error Calculation
        errors_MSE    = sum(abs(txbits-rxbits_MSE))+errors_MSE;
        errors_LMSE   = sum(abs(txbits-rxbits_LMSE))+errors_LMSE;
        errors_NearML = sum(abs(txbits-rxbits_NearML))+errors_NearML;
        errors_Diag = sum(abs(txbits - rxbits_Diag)) + errors_Diag;

    end
    % Close the waitbar when the computation is finished
    close(hWaitBar);

    BER_MSE(SNR==SNRVECT)    = errors_MSE/(numel(txbits)*frames);
    BER_LMSE(SNR==SNRVECT)   = errors_LMSE/(numel(txbits)*frames);
    BER_NearML(SNR==SNRVECT) = errors_NearML/(numel(txbits)*frames);
    BER_Diag(SNR==SNRVECT) = errors_Diag/(numel(txbits)*frames);
    if FastRun == true
       break;
    end
end


figure1 = figure;
semilogy(SNRVECT, BER_MSE)
hold on;
semilogy(SNRVECT, BER_LMSE)
semilogy(SNRVECT, BER_NearML)
semilogy(SNRVECT, BER_Diag) 

title('BER vs SNR');
legend('LS/ZF', 'LMMSE', 'NearML', 'Diag one-tap')

% Create folder if it doesn't exist
outDir = 'plots';
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

% Timestamp for file name
t = datetime('now');
t.Format = 'MMM_dd_yyyy-HH_mm_ss';
name = strcat(outDir, '/GolderMatlab_Ber_', string(t));

% Save as JPG
jpg = strcat(name, '.jpg');
saveas(figure1, jpg)

% ==== Guardar resultados en CSV con encabezado y SNR ascendente ====
outDirCSV = 'csv_results';
if ~exist(outDirCSV, 'dir'), mkdir(outDirCSV); end

save_ber_csv(fullfile(outDirCSV,'BER_ZF.csv'),        SNRVECT, BER_MSE);
save_ber_csv(fullfile(outDirCSV,'BER_LMMSE.csv'),     SNRVECT, BER_LMSE);
save_ber_csv(fullfile(outDirCSV,'BER_NearML.csv'),    SNRVECT, BER_NearML);
save_ber_csv(fullfile(outDirCSV,'BER_DiagOneTap.csv'),SNRVECT, BER_Diag);

% ---- Función local para escribir con el formato exacto ----
function save_ber_csv(filepath, SNRVECT, BER)
    % Ordenar por SNR ascendente
    [SNR_sorted, idx] = sort(SNRVECT(:), 'ascend');
    BER_sorted = BER(idx);

    fid = fopen(filepath, 'w');
    if fid == -1, error('No se pudo abrir %s', filepath); end
    fprintf(fid, 'SNR_dB,BER\n');
    for k = 1:numel(SNR_sorted)
        % SNR con 1 decimal (5.0, 7.0, ...) y BER en decimal (sin notación científica)
        fprintf(fid, '%.1f,%.16f\n', SNR_sorted(k), BER_sorted(k));
    end
    fclose(fid);
end









