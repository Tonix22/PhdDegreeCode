%% DPSKStack_NN_BER.m
% This script generates DPSK stacked signals over a range of SNRs,
% feeds each row of the stack into a trained neural network, and computes
% the resulting Bit Error Rate (BER).

close all;
clc;
addpath('Libraries');  % Ensure your helper functions are accessible

%% 1. System Parameters
SNR_dB    = 10:5:30;       % SNR range in dB
M         = 4;            % Modulation order (e.g., QPSK)
FFTSize   = 48;           % FFT size for OFDM and stack height
Retransmitions = 3;       % Number of retransmision
numSC     = 48;           % Number of subcarriers (also length of transmitted symbol vector)
k         = log2(M);      % Bits per symbol
numBitSymbol = numSC * k;   % Total bits per OFDM symbol
errorThreshold = 10000;    % Stop when at least this many bit errors are observed
H = load('../../Data/kaggle_dataset/v2v80211p_LOS.mat').vectReal32b;

%% 2. Setup Python and Load the Neural Network
% Set environment variable if you run into library conflicts
setenv('LD_PRELOAD', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6');
%if that doesnt works run this linux command before running matlab 
% export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
% run MimoDPSKStackTest.m
% matlab

%% Add the Python file directory to Python's search path
mypythonpath = '/home/tonix/Documents/PhdDegreeCode/PythonCode/DeepLearning/MIMOSolution';
insert(py.sys.path, int32(0), mypythonpath);

%% 3. Initialize BER Storage
ber = zeros(1, length(SNR_dB));

%% 4. Main Loop: Process each SNR value
for idx = 1:length(SNR_dB)
    currentSNR = SNR_dB(idx);
    % Create an instance of the ModelLoader class from Python
    % Note: The constructor parameters must match those expected by your Python class.
    pthFile = "/home/tonix/Documents/PhdDegreeCode/PythonCode/DeepLearning/MIMOSolution/model_MIMO_DPSK_"+currentSNR+".pth";
    model_loader = py.ModelLoader.ModelLoader(pthFile, py.float(1e-3), "cuda");

    numError = 0;  % Accumulated bit errors
    numBits = 0;   % Accumulated total bits processed
    
    fprintf('Processing SNR = %d dB...\n', currentSNR);
    
    % Continue processing samples until errorThreshold is reached
    while (numError < errorThreshold && numBits < 1e6 )
        % Generate random transmitted symbols (1 x numSC)
        signalTx = generateRandomData(M, numSC);
        % signalTx is assumed to contain symbol labels (e.g., 1,2,3,4) for each subcarrier
        
        % Generate DPSK stack sample (dimensions: FFTSize x FFTSize)
        % For each of the FFTSize rows, simulate a channel transmission.
        sampleStack = zeros(Retransmitions+1, FFTSize);
        for row = 1:Retransmitions
            % Simulate channel transmission for the current row.
            % Here, processChannelAndTransmit returns (among other outputs) the
            % received DPSK signal. The SNR used in the channel simulation is currentSNR.
            [~, DPSKsignalRx] = processChannelAndTransmit(signalTx, M, FFTSize, currentSNR, numSC,H);
            % Normalize the phase to the interval [0,1]
            sampleStack(row, :) = DPSKsignalRx;
        end

        for i = 1:FFTSize
            % Extract the complex symbols for all retransmissions
            symbols = sampleStack(1:Retransmitions,i);  % Complex numbers
            % Compute the Circular Mean (Mean Angle)
            mean_angle = angle(mean(exp(1j * angle(symbols))));  % Mean of unit vectors
            sampleStack(1:Retransmitions,i) = angle(symbols);
            sampleStack(end,i) = mean_angle;
            sampleStack(:,i) = (sampleStack(:,i) + pi) / (2 * pi);
        end

        % Use the neural network to predict the transmitted symbols from the stack.
        % Here we assume each row of the stack corresponds to one subcarrier.
        predictedSymbols = zeros(1, FFTSize);
        outputArray = zeros(4, FFTSize);
        for col = 1:FFTSize
            % Extract the current row (1 x 64 vector)
            colData = sampleStack(:, col);
            % Reshape to match the network input dimensions [1, 1, FFTSize]
            sampleInputMat = reshape(colData, [1, 1, Retransmitions+1]);
            
            % Convert the MATLAB array to a Python tensor
            sampleInputPy = py.torch.tensor(...
            py.numpy.array(sampleInputMat, pyargs('dtype', py.numpy.float32)), ...
            pyargs('device', 'cuda') );
            
            % Run inference using the neural network
            outputPy = model_loader.predict(sampleInputPy);
            
            % Convert the output (a PyTorch tensor) to a MATLAB array
            outputArray(:,col) = double(outputPy.cpu().numpy());
            % Use argmax to obtain the predicted symbol (MATLAB indexes from 1)
            [~, pred] = max(outputArray(:,col));
            predictedSymbols(col) = pred-1;
        end
        % Compare the network predictions with the ground truth.
        % (Assuming that each element in signalTx corresponds to the correct subcarrier.)
        symbolErrors = sum(predictedSymbols ~= signalTx');
        % Count bit errors (each symbol error represents k bit errors)
        numError = numError + symbolErrors * k;
        % Total bits processed in this sample (numSC * k)
        numBits = numBits + numBitSymbol;
        if(mod(numBits,10000) == 0)
            fprintf('numError = %d, numBits = %.4e\n', numError, numBits);
        end
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