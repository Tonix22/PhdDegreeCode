clear; clc;

% Load your channel data (expects LOS and NLOS in workspace)
loadmatlab;
addpath('.')

% --- Config ---
modorder = 4;                 % QAM constellation size
%SNRVECT  = 35:-2:5;           % dB values
SNRVECT  = [15];
frames   = 1e5;               % frames per SNR
rng(1);                       % reproducibility
outdir   = 'datasetsTest';    % output folder
if ~exist(outdir,'dir'), mkdir(outdir); end

M       = size(LOS,1);        % e.g., 48
nLOS    = size(LOS,3);
nNLOS   = size(NLOS,3);
LOS_ix  = 1;
NLOS_ix = 1;

for SNR = SNRVECT
    % Storage for this SNR
    X_all    = complex(zeros(M, frames));
    y_n_all  = complex(zeros(M, frames));
    H_all    = complex(zeros(M, M, frames));
    isLOS    = false(frames,1);  % true if LOS, false if NLOS

    fprintf('Generating X, y_n, H for SNR = %d dB ...\n', SNR);

    for k = 1:frames
        % Alternate between LOS and NLOS
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
        txbits = randi([0 1], M*log2(modorder), 1);
        X      = qammod(txbits, modorder, 'gray', 'InputType', 'bit', 'UnitAveragePower', true);

        % Channel
        Y = H * X;

        % Noise power from SNR
        Ps = sum(abs(Y).^2) / length(Y);
        Pn = Ps / (10^(SNR/10));

        % AWGN
        n   = sqrt(Pn/2) * complex(randn(size(Y)), randn(size(Y)));
        y_n = Y + n;

        % Save this frame
        %X_hat_LMSE = inv(H'*H+eye(48)*Pn)*H'*y_n; %LMSE
        X_all(:,k)     = X;
        y_n_all(:,k)   = H'*y_n;
        H_all(:,:,k)   = H;
    end

    % Save per-SNR .mat (H is large -> use -v7.3)
    fname = fullfile(outdir, sprintf('XyH_SNR_%ddB.mat', SNR));
    save(fname, 'X_all', 'y_n_all', 'H_all', 'isLOS', 'SNR', 'modorder', '-v7.3');
    fprintf('Saved: %s\n', fname);
end

fprintf('Done. Files saved in "%s".\n', outdir);
