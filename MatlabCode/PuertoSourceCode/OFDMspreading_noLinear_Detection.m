clc;
clear all;
close all;
addpath('./Libraries/');

%% parameter 
% QAM modulation order
M = 4; 
fftSize = 48;

k = log2(M);
constellation = qammod(0:M-1,M,"gray");

%% channel V2V
channel = load('../../Data/kaggle_dataset/v2v80211p_LOS.mat');

H = channel.vectReal32b;
channelCont=1;
%% system model 

SNR = 5:5:45; % Range of SNR values, in dB.
EffEsp=(log2(M)*(48)/(64));

LMMSE = Equalizers("LMMSE", true, @LMMSE);
OSIC  = Equalizers("OSIC", false, @OSIC_Det);
NearML = Equalizers("NearML", false, @QRM_Det4b);

EqCollection = [LMMSE, OSIC, NearML];

for i = 1:length(EqCollection)
    currentEqualizer = EqCollection(i);
    disp(currentEqualizer.name);
    berEst = [];

    for n= 1:length(SNR)
        % Reset the error and bit counters
        numErrs = 0;
        numBits = 0;
        while numErrs < 1e3 && numBits < 1e6
            % Generate binary data and convert to symbols
            tx = randi([0 1],fftSize*log2(M),1);
            % QAM modulate using 'Gray' symbol mapping
            qpskSig = qammod(tx,M,"gray","InputType","bit","UnitAveragePower",true);
            
            %Process Channel
            G = H(:,:,channelCont);
            channelCont = channelCont +1;
            if channelCont == 10000
                channelCont = 1;
            end
                  
            TxSig = qpskSig;
            TxSig = fft(qpskSig,fftSize); 

            RxSignal = G*TxSig;

            H1 = ifft(G,fftSize);
            H1 = fft(H1.');
            H1 = H1.';

            %Multipy input by chanel in frequency domain.
            RxSignal = awgn(RxSignal,SNR(n),"measured");
            rxSig=ifft(RxSignal,fftSize);
            NoiseVar=10^(-SNR(n)/10);
            NoiseAmpl=sqrt(NoiseVar);
            nstd = sqrt( (1 /( 10^(SNR(n)/10 ) ) )/2 );

            if currentEqualizer.isLinear
                if currentEqualizer.name == "LMMSE"
                    rxSig = currentEqualizer.handler(H1,NoiseVar,rxSig,48);
                end
            else
                %[Q, R] = qr(G);
                %yp = Q'*rxSig;
                
                if currentEqualizer.name == "OSIC"
                    rxSig = currentEqualizer.handler(H1,rxSig);
                    
                elseif currentEqualizer.name == "NearML"
                    [yp, R, orden] = MMSESortedQRC(H1,nstd,rxSig,48,0);
                    rxSig  = currentEqualizer.handler(yp,R,constellation,orden);
                end
            end

            rx = qamdemod(rxSig,M, 'OutputType', 'bit');  % Demodulate
            % Calculate the number of bit errors
            nErrors = biterr(tx,rx);
            % Increment the error and bit counters
            numErrs = numErrs + nErrors;
            numBits = numBits + fftSize*log2(M);
        end
        
        berEst(n) = numErrs/numBits
    end

    currentEqualizer.PlotResult(SNR,berEst);

end

