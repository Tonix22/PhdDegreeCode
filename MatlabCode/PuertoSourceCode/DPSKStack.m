close all;
clc;
addpath('Libraries');
%% Parameter System
SNR_dB_Range = 5:5:35;         % Range of SNR values in dB
M = 4;                   % Modulation order (QPSK)
FFTSize = 64;            % FFT size for OFDM
k = log2(M);             % Bits per symbol (log base 2 of modulation order)
numSC = 64;              % Number of subcarriers
numBitSymbol = numSC * k; % Total number of bits per OFDM symbol
samplesPerSNR = 10000;


for SNR_dB = SNR_dB_Range

    mimoSignal = zeros(samplesPerSNR,FFTSize,FFTSize);
    Tx         = zeros(samplesPerSNR, numSC);

    for s = 1:samplesPerSNR
        signalTx = generateRandomData(M, numSC);
        Tx(s,:) = signalTx;
        %% Stack Generation procedure
        for i = 1:FFTSize
            [~,DPSKsignalRx]= processChannelAndTransmit(signalTx, M, FFTSize, SNR_dB, numSC);
            mimoSignal(s,i,:) = angle(DPSKsignalRx);
        end

        figure;
        imagesc(squeeze(mimoSignal(s,:,:))); % Display the phase as an image
        colormap('jet'); % Use a colormap for better visualization
        colorbar; % Show the color scale
        title('Phase of the Complex Signal');
        xlabel('Columns');
        ylabel('Rows');
        saveas(gcf, 'phase_plot.png');
        return;
        
    end

    %% Save mimoSignal using Python inside MATLAB
    filename = sprintf("MIMODataSet/Signal_SNR_Rx_%d.npy", SNR_dB);
    py.numpy.save(filename , py.numpy.array(mimoSignal))

    filename = sprintf("MIMODataSet/Signal_SNR_Tx_%d.npy", SNR_dB);
    py.numpy.save(filename , py.numpy.array(Tx))
    disp(['Saved ' filename ' successfully.']);

end



% Plot the phase as an image
%figure;
%imagesc(mimoSignal); % Display the phase as an image
%colormap('jet'); % Use a colormap for better visualization
%colorbar; % Show the color scale
%title('Phase of the Complex Signal');
%xlabel('Columns');
%ylabel('Rows');
%saveas(gcf, 'phase_plot.png');
