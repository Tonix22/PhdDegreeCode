close all;
clc;
addpath('Libraries');
%% Parameter System
SNR_dB_Range = 10:5:35;         % Range of SNR values in dB
M = 4;                   % Modulation order (QPSK)
FFTSize = 48;            % FFT size for OFDM
Retransmitions = 3;      % Number of retransmision
k = log2(M);             % Bits per symbol (log base 2 of modulation order)
numSC = 48;              % Number of subcarriers
numBitSymbol = numSC * k; % Total number of bits per OFDM symbol
samplesPerSNR = 10000;
H = load('../../Data/kaggle_dataset/v2v80211p_LOS.mat').vectReal32b;

for SNR_dB = SNR_dB_Range

    mimoSignal = zeros(samplesPerSNR,Retransmitions+1,FFTSize);
    Tx         = zeros(samplesPerSNR, numSC);

    for s = 1:samplesPerSNR
        signalTx = generateRandomData(M, numSC);
        Tx(s,:) = signalTx;
        %% Stack Generation procedure
        for i = 1:Retransmitions
            [~,DPSKsignalRx]= processChannelAndTransmit(signalTx, M, FFTSize, SNR_dB, numSC,H);
            mimoSignal(s,i,:) = DPSKsignalRx;
        end

        for i = 1:FFTSize
            % Extract the complex symbols for all retransmissions
            symbols = mimoSignal(s,1:Retransmitions,i);  % Complex numbers
            % Compute the Circular Mean (Mean Angle)
            mean_angle = angle(mean(exp(1j * angle(symbols))));  % Mean of unit vectors
            mimoSignal(s,1:Retransmitions,i) = angle(symbols);
            mimoSignal(s,end,i) = mean_angle;
            mimoSignal(s,:,i) = (mimoSignal(s,:,i) + pi) / (2 * pi);
        end
        %{
        figure;
        imagesc(squeeze(mimoSignal(s,:,:))); % Display the phase as an image
        colormap('jet'); % Use a colormap for better visualization
        colorbar; % Show the color scale
        title('Phase of the Complex Signal');
        xlabel('Columns');
        ylabel('Rows');
        saveas(gcf, 'phase_plot.png');
        return;
        %}
    end

    %% Save mimoSignal using Python inside MATLAB
    filename = sprintf("MIMODataSet/Signal_SNR_Rx_%d.npy", SNR_dB);
    py.numpy.save(filename , py.numpy.array(mimoSignal))
    disp(['Saved ' filename ' successfully.']);

    filename = sprintf("MIMODataSet/Signal_SNR_Tx_%d.npy", SNR_dB);
    py.numpy.save(filename , py.numpy.array(Tx))
    disp(['Saved ' filename ' successfully.']);

end