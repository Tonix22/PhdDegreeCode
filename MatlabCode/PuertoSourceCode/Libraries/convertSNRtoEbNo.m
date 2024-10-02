function EbNo_dB = convertSNRtoEbNo(SNR_dB, numBitSymbol)
    % Ensure inputs are of type double
    SNR_dB = double(SNR_dB);
    numBitSymbol = double(numBitSymbol);
    % Convert SNR (in dB) to Eb/No (in dB)
    SNR_linear = 10.^(SNR_dB / 10);         % Convert SNR from dB to linear scale
    EbNo_linear = SNR_linear / numBitSymbol; % Calculate Eb/No in linear scale
    EbNo_dB = 10 * log10(EbNo_linear);      % Convert Eb/No to dB
end
