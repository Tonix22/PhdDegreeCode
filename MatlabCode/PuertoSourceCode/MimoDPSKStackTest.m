%% DPSKStack_NN_BER.m
% This script generates DPSK stacked signals over a range of SNRs,
% feeds each row of the stack into a trained neural network, and computes
% the resulting Bit Error Rate (BER).

close all;
clc;
addpath('Libraries');  % Ensure your helper functions are accessible

%% 1. System Parameters
SNR_dB    = 35:5:35;       % SNR range in dB
M         = 4;            % Modulation order (e.g., QPSK)
FFTSize   = 64;           % FFT size for OFDM and stack height
numSC     = 64;           % Number of subcarriers (also length of transmitted symbol vector)
k         = log2(M);      % Bits per symbol
numBitSymbol = numSC * k;   % Total bits per OFDM symbol
errorThreshold = 1000;    % Stop when at least this many bit errors are observed


%% 2. Setup Python and Load the Neural Network
% Set environment variable if you run into library conflicts
setenv('LD_PRELOAD', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6');
%if that doesnt works run this linux command before running matlab 
% export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
% matlab

%% Add the Python file directory to Python's search path
mypythonpath = '/home/tonix/Documents/PhdDegreeCode/PythonCode/DeepLearning/MIMOSolution';
insert(py.sys.path, int32(0), mypythonpath);

% Create an instance of the ModelLoader class from Python
% Note: The constructor parameters must match those expected by your Python class.
model_loader = py.ModelLoader.ModelLoader("/home/tonix/Documents/PhdDegreeCode/PythonCode/DeepLearning/MIMOSolution/model_MIMO_DPSK_35.pth", py.float(1e-3), "cuda");

%% 3. Initialize BER Storage
ber = zeros(1, length(SNR_dB));

%% 4. Main Loop: Process each SNR value
for idx = 1:length(SNR_dB)
    currentSNR = SNR_dB(idx);
    numError = 0;  % Accumulated bit errors
    numBits = 0;   % Accumulated total bits processed
    
    fprintf('Processing SNR = %d dB...\n', currentSNR);
    
    % Continue processing samples until errorThreshold is reached
    while numError < errorThreshold
        % Generate random transmitted symbols (1 x numSC)
        signalTx = generateRandomData(M, numSC);
        % signalTx is assumed to contain symbol labels (e.g., 1,2,3,4) for each subcarrier
        
        % Generate DPSK stack sample (dimensions: FFTSize x FFTSize)
        % For each of the FFTSize rows, simulate a channel transmission.
        sampleStack = zeros(FFTSize, FFTSize);
        for row = 1:FFTSize
            % Simulate channel transmission for the current row.
            % Here, processChannelAndTransmit returns (among other outputs) the
            % received DPSK signal. The SNR used in the channel simulation is currentSNR.
            [~, DPSKsignalRx] = processChannelAndTransmit(signalTx, M, FFTSize, currentSNR, numSC);
            % Normalize the phase to the interval [0,1]
            sampleStack(row, :) = (angle(DPSKsignalRx)/(2*pi) + 1) / 2;
        end
        
        % Use the neural network to predict the transmitted symbols from the stack.
        % Here we assume each row of the stack corresponds to one subcarrier.
        predictedSymbols = zeros(1, FFTSize);
        for row = 1:FFTSize
            % Extract the current row (1 x 64 vector)
            rowData = sampleStack(row, :);
            % Reshape to match the network input dimensions [1, 1, FFTSize]
            sampleInputMat = reshape(rowData, [1, 1, FFTSize]);
            
            % Convert the MATLAB array to a Python tensor
            sampleInputPy = py.torch.tensor(...
            py.numpy.array(sampleInputMat, pyargs('dtype', py.numpy.float32)), ...
            pyargs('device', 'cuda') );
            
            % Run inference using the neural network
            outputPy = model_loader.predict(sampleInputPy);
            
            % Convert the output (a PyTorch tensor) to a MATLAB array
            outputArray = double(outputPy.cpu().numpy());
            % Use argmax to obtain the predicted symbol (MATLAB indexes from 1)
            [~, pred] = max(outputArray);
            predictedSymbols(row) = pred-1;
        end
        
        % Compare the network predictions with the ground truth.
        % (Assuming that each element in signalTx corresponds to the correct subcarrier.)
        symbolErrors = sum(predictedSymbols ~= signalTx');
        % Count bit errors (each symbol error represents k bit errors)
        numError = numError + symbolErrors * k;
        % Total bits processed in this sample (numSC * k)
        numBits = numBits + numBitSymbol;
    end
    
    % Compute the BER for the current SNR value
    ber(idx) = numError / numBits;
    fprintf('SNR = %d dB, BER = %.4e\n', currentSNR, ber(idx));
end

%% 5. Plot BER vs. SNR
%% Plot Results
berTheoretical = calculateTheoreticalBER(SNR_dB, M, numBitSymbol); % Optional calculation of theoretical BER
plotBER(SNR_dB, ber, M, numBitSymbol, 'DPSK_SNR', berTheoretical);
saveBERToCSV(SNR_dB, ber, 'DPSK_SNR.csv');