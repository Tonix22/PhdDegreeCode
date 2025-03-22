close all;
clc;

% If needed, add your function paths:
addpath('../Libraries');
% addpath('BinaryClassifier');

%% Parameter System
SNR_dB_Range = 10:5:35;   % You can adjust as needed
M            = 4;         % QPSK
FFTSize      = 48;        % OFDM size
numSC        = 48;        % Number of subcarriers
MaxNumOfSymbols = 10000;   % Number of samples to generate per SNR

% Load channel data (example from your existing code)
H = load('../../../Data/kaggle_dataset/v2v80211p_LOS.mat').vectReal32b;

%% Main loop over SNR values
for SNR_dB = SNR_dB_Range
    
    % Preallocate for a single SNR batch
    % dataSet shape = (MaxNumOfSymbols, 4, 48, 48)
    % labelSet shape = (MaxNumOfSymbols, 48)
    dataSet  = zeros(MaxNumOfSymbols, 4, FFTSize, FFTSize, 'single');
    labelSet = zeros(MaxNumOfSymbols, FFTSize, 'single');
    
    for s = 1 : MaxNumOfSymbols
        
        %-------------------------------------------------
        % 1) Generate random 48-symbol ground truth
        %    rawSymbols ∈ {0,1,2,3} of length 48
        %-------------------------------------------------
        rawSymbols = generateRandomData(M, numSC);  
        labelSet(s, :) = single(rawSymbols);
        
        %-------------------------------------------------
        % 2) Transmit & receive through channel
        %-------------------------------------------------
        [~, signalEstimate] = processChannelAndTransmit(...
            rawSymbols, M, FFTSize, SNR_dB, numSC);
        
        % signalEstimate is the received 48-subcarrier complex signal
        % Per your script, you convert angle to [0..1] range:
        vector = (angle(signalEstimate) + pi) / (2 * pi);  % shape = (48,)
        
        [ch1, ch2, ch3, ch4] = computeCorrChannels(vector);
        
        %-------------------------------------------------
        % 5) Assign to dataSet
        %  dataSet(s, channel, row, col)
        %-------------------------------------------------
        dataSet(s, 1, :, :) = single(ch1);
        dataSet(s, 2, :, :) = single(ch2);
        dataSet(s, 3, :, :) = single(ch3);
        dataSet(s, 4, :, :) = single(ch4);
        
        % (No plotting now — purely data generation)
        
    end % end for each sample (MaxNumOfSymbols)
    
    %-------------------------------------------------
    % 6) Save to NumPy .npy
    %    E.g., one file for data, one file for labels
    %    Adjust folder name / filenames as needed
    %-------------------------------------------------
    dataFilename  = sprintf('Data/CorrData_SNR_%ddB_data.mat', SNR_dB);
    
    save(dataFilename, 'dataSet', 'labelSet', '-v7.3');
    disp(['Saved data file: ' dataFilename]);
    
end
