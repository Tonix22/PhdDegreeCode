clear; clc;

% Load your channel data (expects LOS and NLOS in workspace)
loadmatlab;
addpath('.')

% --- Config ---
modorder = 4;                 % QAM constellation size
SNRVECT  = 35:-2:5;           % dB values
%SNRVECT  = [15];
frames   = 1e5;               % frames per SNR
rng(1);                       % reproducibility
outdir   = 'datasetsTest';    % output folder
if ~exist(outdir,'dir'), mkdir(outdir); end

M         = size(LOS,1);      % e.g., 48
bitsPerSym = log2(modorder);

nLOS    = size(LOS,3);
nNLOS   = size(NLOS,3);
LOS_ix  = 1;
NLOS_ix = 1;

for SNR = SNRVECT
    % Storage for this SNR
    X_all       = complex(zeros(M, frames));
    y_n_all     = complex(zeros(M, frames));   % matched-filter output: H' * y_n
    H_all       = complex(zeros(M, M, frames));
    isLOS       = false(frames,1);             % true if LOS, false if NLOS

    % Nuevos: bits por frame y clases QPSK por subportadora
    txbits_all  = false(M*bitsPerSym, frames); % lógico para ahorrar espacio
    txcls_all   = zeros(M, frames, 'uint8');   % 0..modorder-1 (Gray)

    fprintf('Generating X, y_n, H for SNR = %d dB ...\n', SNR);

    for k = 1:frames
        % Alterna entre LOS y NLOS
        if bitand(k,1)
            H = LOS(:,:,LOS_ix);
            isLOS(k) = true;
            LOS_ix = LOS_ix + 1;
            if LOS_ix > nLOS, LOS_ix = 1; end
        else
            H = NLOS(:,:,NLOS_ix);
            isLOS(k) = false;
            NLOS_ix = NLOS_ix + 1;
            if NLOS_ix > nNLOS, NLOS_ix = 1; end
        end

        % ----- TX vector X -----
        txbits = randi([0 1], M*bitsPerSym, 1, 'uint8');            % bits (Gray)
        X      = qammod(txbits, modorder, 'gray', ...
                        'InputType', 'bit', 'UnitAveragePower', true);

        % Clases QPSK coherentes con el mapeo Gray de qammod
        txcls  = qamdemod(X, modorder, 'gray', 'OutputType', 'integer'); % 0..3

        % Canal
        Y = H * X;

        % Potencia de señal y ruido para la SNR deseada
        Ps = sum(abs(Y).^2) / M;
        Pn = Ps / (10^(SNR/10));

        % AWGN
        n   = sqrt(Pn/2) * complex(randn(M,1), randn(M,1));
        y_n = Y + n;

        % Guardar frame
        X_all(:,k)     = X;
        y_n_all(:,k)   = H' * y_n;       % matched filter (H^H y_n)
        H_all(:,:,k)   = H;

        txbits_all(:,k) = logical(txbits);   % bits usados (compactos)
        txcls_all(:,k)  = uint8(txcls);      % clase por subportadora (0..3)
    end

    % Guardar por SNR (H es grande -> usar -v7.3)
    fname = fullfile(outdir, sprintf('XyH_SNR_%ddB.mat', SNR));
    save(fname, 'X_all', 'y_n_all', 'H_all', 'isLOS', 'SNR', 'modorder', ...
                'txbits_all', 'txcls_all', 'bitsPerSym', '-v7.3');
    fprintf('Saved: %s\n', fname);
end

fprintf('Done. Files saved in "%s".\n', outdir);
