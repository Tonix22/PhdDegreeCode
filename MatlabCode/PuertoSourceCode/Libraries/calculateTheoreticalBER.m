function berTheoretical = calculateTheoreticalBER(SNR_dB, M, numBitSymbol)
    EbNo_dB = convertSNRtoEbNo(SNR_dB, numBitSymbol); % Convert SNR to Eb/No
    berTheoretical = berawgn(EbNo_dB, 'dpsk', M, 'nondiff'); % Theoretical BER
end
