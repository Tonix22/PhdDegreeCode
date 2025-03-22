%% 1. Setup Parameters and Read Image
clc; clear; close all;

% -- Image path (use your own dataset path)
imgPath = '/home/tonix/Documents/PhdDegreeCode/Data/Picture/Cascade.jpeg';  % Example image in MATLAB
imgData = imread(imgPath);

% Convert to grayscale if needed
if size(imgData,3) == 3
    imgData = rgb2gray(imgData);
end

% Convert image to a 1D array of bits
% Each pixel: 8 bits
imgVec  = imgData(:);
imgBits = de2bi(imgVec, 8, 'left-msb');  % size: [numPixels, 8]
imgBits = imgBits.';                     % transpose → each column is a pixel
imgBits = imgBits(:);                    % convert to column vector

% -- Communication parameters
N       = 3;          % Number of bits per PSK symbol, e.g. 3 bits => 8-PSK
M       = 2^N;        % PSK order (M = 8 for 8-PSK)
numSubc = 48;         % Number of data subcarriers in OFDM (after removing pilots)
SNRdB   = 20;         % SNR in dB for AWGN channel

% -- OFDM parameters
cpLen   = 16;         % Cyclic prefix length (example)
% NOTE: Typically you'd have a total IFFT size of 64 (like in 802.11), 
% but we'll keep it simple and just show 48 subcarriers + no pilot carriers. 
fftSize = numSubc;    % We'll do an IFFT of size = 48 for simplicity in this example

%% 2. PSK Modulation Preparation

% We need to ensure the total number of bits is divisible by N and by numSubc
% so that we can form complete OFDM symbols with exactly one PSK symbol per subcarrier.

% Zero-pad the bitstream if necessary
totalBits = length(imgBits);
bitsNeeded = ceil(totalBits/(N*numSubc))*(N*numSubc);
padBits = bitsNeeded - totalBits;
imgBitsPadded = [imgBits; randi([0 1], padBits, 1)];  % random bits for padding

% Reshape to [N, numberOfSymbols]
numSymbolsTotal = bitsNeeded / N;    % total PSK symbols
bitsPerSymbol   = reshape(imgBitsPadded, N, numSymbolsTotal);

% 3. PSK Modulation
symTxAll = zeros(1, numSymbolsTotal);  % pre-allocate
pskMod = comm.PSKModulator(M, 'BitInput', true, 'PhaseOffset', 0);

% Loop or vectorize. We can do it in a loop for clarity:
for k = 1:numSymbolsTotal
    symBits = bitsPerSymbol(:,k);
    symTxAll(k) = pskMod(symBits);
end

% Now we have a stream of PSK symbols: symTxAll

%% 4. OFDM Transmitter
% We'll group the PSK symbols into blocks of 'numSubc'
% Each block forms one OFDM symbol (48 subcarriers).

numOFDMsymbols = numSymbolsTotal / numSubc; 
symTxAll_matrix = reshape(symTxAll, numSubc, numOFDMsymbols);

% IFFT for each block (size = fftSize = 48)
ofdmSymbols_time = ifft(symTxAll_matrix, fftSize, 1);  % ifft along the rows

% Add cyclic prefix
% cpLen = 16 -> take last 16 samples from each block
ofdmSymbols_withCP = [ofdmSymbols_time(end-cpLen+1:end,:); ofdmSymbols_time];

% Serialize the OFDM symbols
txSignal = ofdmSymbols_withCP(:);

%% 5. Transmit Through AWGN
rxSignal = awgn(txSignal, SNRdB, 'measured');

%% 6. OFDM Receiver
% Reshape received signal to [fftSize + cpLen, numOFDMsymbols]
rxSignal_matrix = reshape(rxSignal, fftSize+cpLen, numOFDMsymbols);

% Remove cyclic prefix
rxSignal_noCP = rxSignal_matrix(cpLen+1:end, :);

% FFT to get back to frequency domain
rxSymbols_matrix = fft(rxSignal_noCP, fftSize, 1);

% rxSymbols_matrix should be size [48, numOFDMsymbols]
% We'll assume the subcarrier mapping is direct (no pilot subcarriers).
symRxAll = rxSymbols_matrix(:);

%% 7. PSK Demodulation
pskDemod = comm.PSKDemodulator(M, 'BitOutput', true, 'PhaseOffset', 0);

% Demodulate each symbol
rxBitsAll = zeros(N * length(symRxAll), 1);
idx = 1;
for k = 1:length(symRxAll)
    rxSym = symRxAll(k);
    demodBits = pskDemod(rxSym);
    rxBitsAll(idx:idx+N-1) = demodBits;
    idx = idx + N;
end

% We added padding, so let’s remove the extra bits
rxBits = rxBitsAll(1:totalBits);

%% 8. Compute BER
numErrors = sum(rxBits ~= imgBits);
BER = numErrors / totalBits;

fprintf('SNR = %.1f dB\n', SNRdB);
fprintf('Number of errors = %d\n', numErrors);
fprintf('Bit Error Rate   = %g\n', BER);

%% 9. (Optional) Reconstruct Image from Received Bits
% This is just to see how the decoded bits compare in an image sense.
% If BER is small, the image should be mostly intact.

rxBits8 = reshape(rxBits, 8, []).';
rxImg   = uint8(bin2dec(num2str(rxBits8)));
rxImg   = reshape(rxImg, size(imgData));

figure; 
subplot(1,2,1); imshow(imgData); title('Original Image');
subplot(1,2,2); imshow(rxImg);  title('Received Image (Decoded)');
